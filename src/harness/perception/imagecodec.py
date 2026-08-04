"""PNG без внешних зависимостей и уменьшение кадра перед отправкой.

Зачем свой кодек. Кадр надо отправить в сервис описания, а Pillow в проекте нет и
тащить его ради одной функции не хочется: PNG пишется в тридцать строк на `zlib`,
который в стандартной библиотеке. Заодно это снимает вопрос «а что если Pillow
изменит поведение» — здесь нечему меняться.

PNG, а не JPEG, по существу: JPEG теряет данные, а именно на побитовой
одинаковости неподвижных пикселей держится и разделение слоёв, и кэш ответов.
Отправлять модели одно, а хранить другое — верный способ получить расхождение,
которое найдётся через месяц.

Уменьшение кадра — главный рычаг экономии на бесплатных тарифах: они считают
токены, а токенов у картинки тем больше, чем она крупнее. Уменьшение здесь
целочисленное усреднение по блокам, а не выборка каждого N-го пикселя: выборка
теряет мелкие надписи, а они и есть то, что мы просим описать.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib

import numpy as np


def png_bytes(image: np.ndarray, *, level: int = 6) -> bytes:
    """Закодировать кадр в PNG. Принимает (h,w) серый или (h,w,3) цветной."""
    if image.dtype != np.uint8:
        raise ValueError(f"кадр должен быть uint8, пришёл {image.dtype}")
    if image.ndim == 2:
        colour_type, planes = 0, 1
        data = image
    elif image.ndim == 3 and image.shape[2] == 3:
        colour_type, planes = 2, 3
        data = image
    elif image.ndim == 3 and image.shape[2] == 4:
        colour_type, planes = 6, 4
        data = image
    else:
        raise ValueError(f"не умею кодировать форму {image.shape}")

    height, width = data.shape[:2]
    stride = width * planes
    raw = bytearray()
    flat = np.ascontiguousarray(data).reshape(height, stride)
    for row in flat:
        raw.append(0)                 # фильтр 0: без предсказания
        raw.extend(row.tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), level))
            + chunk(b"IEND", b""))


def data_uri(image: np.ndarray, *, level: int = 6) -> str:
    """`data:image/png;base64,...` — то, что понимают все сервисы описания."""
    return "data:image/png;base64," + base64.b64encode(png_bytes(image, level=level)).decode()


def base64_png(image: np.ndarray, *, level: int = 6) -> str:
    """Только base64, без префикса: так просят Gemini и Ollama."""
    return base64.b64encode(png_bytes(image)).decode()


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    """Уменьшить кадр так, чтобы большая сторона не превышала `max_side`.

    Усреднение по блокам целочисленным коэффициентом. Дробного масштабирования
    нет намеренно: оно требует интерполяции, а интерполяция размывает мелкие
    надписи — то самое, что мы и просим описать.
    """
    if max_side <= 0:
        raise ValueError("max_side должен быть положительным")
    h, w = image.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / max_side)))
    if factor == 1:
        return image
    hh, ww = (h // factor) * factor, (w // factor) * factor
    cut = image[:hh, :ww]
    if cut.ndim == 2:
        blocks = cut.reshape(hh // factor, factor, ww // factor, factor)
        return blocks.mean(axis=(1, 3)).round().astype(np.uint8)
    planes = cut.shape[2]
    blocks = cut.reshape(hh // factor, factor, ww // factor, factor, planes)
    return blocks.mean(axis=(1, 3)).round().astype(np.uint8)


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Вырезать окно внимания. Границы обрезаются по кадру, а не додумываются."""
    top, left, height, width = box
    top = max(0, int(top))
    left = max(0, int(left))
    return image[top:top + int(height), left:left + int(width)]


def frame_fingerprint(image: np.ndarray) -> str:
    """Отпечаток кадра для кэша ответов.

    Побитовый: два побитово одинаковых кадра обязаны дать один ответ, а два
    различающихся хоть на пиксель — считаться разными. Это не «похожесть», а
    именно тождество: приблизительный кэш выдавал бы ответ про другой кадр, и
    поймать это было бы почти невозможно.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(str(image.shape).encode())
    h.update(str(image.dtype).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()
