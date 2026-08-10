"""Хранилище крупных блоков: кадры и звук.

Из 0.3: «кадры — отдельным хранилищем, метаданные — индексом. Не по файлу на
кадр». Из 0.4: «промотка к произвольному моменту» должна открываться за доли
секунды.

Устройство: несколько шардов, в каждый только дозапись; рядом индекс в JSONL,
одна строка на блок. Индекс читается один раз при открытии, дальше доступ —
`seek` по смещению. Чтение блока не зависит от количества блоков.

Инвариант 1 распространяется и сюда: шарды открываются в режиме `"ab"`, и
единственная операция записи — «в конец». Функции «переписать блок» нет.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

MAGIC = b"HBLB1"          # чтобы битый шард не читался как валидный
HEADER_LEN = len(MAGIC) + 4  # magic + длина полезной части (uint32 BE)


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Ссылка на блок. Это и есть то, что лежит в записи журнала.

    `enc` — как блок закодирован. `"raw"` — массив как есть. `"delta"` — разность
    с предыдущим блоком по модулю 256 со сдвигом на 128; `base` указывает на
    ближайший опорный кадр, от которого разности идут подряд.
    """

    blob_id: int
    shard: int
    offset: int
    nbytes: int          # размер сжатого блока на диске
    shape: tuple[int, ...]
    dtype: str
    enc: str = "raw"
    base: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {"blob_id": self.blob_id, "shard": self.shard, "offset": self.offset,
             "nbytes": self.nbytes, "shape": list(self.shape), "dtype": self.dtype}
        if self.enc != "raw":
            d["enc"] = self.enc
            d["base"] = self.base
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BlobRef:
        return cls(int(d["blob_id"]), int(d["shard"]), int(d["offset"]), int(d["nbytes"]),
                   tuple(int(x) for x in d["shape"]), str(d["dtype"]),
                   str(d.get("enc", "raw")),
                   None if d.get("base") is None else int(d["base"]))


@dataclass(slots=True)
class Cost:
    """Куда ушло время и сколько вышло байтов. `TASK-17`, пункт 2.

    Заведено потому, что частота живой записи оказалась 6.5 кадр/с против 32.5 в
    `selftest` на той же машине, и объяснения не было ни у кого: «узкое место где-то в
    записи на диск» — не диагноз. Разбивка снимается **на машине оператора**, а не
    выводится из замера в контейнере: там другой экран, другой диск и другой процессор.

    Счётчики дешёвые нарочно: `perf_counter_ns` стоит десятки наносекунд, а кадр идёт
    десятки миллисекунд. Измерение, из-за которого замедлилось измеряемое, — испорченное
    измерение.
    """

    calls: int = 0
    keyframes: int = 0
    delta_ns: int = 0            # вычисление разности с предыдущим кадром
    compress_ns: int = 0         # zlib.compress
    write_ns: int = 0            # запись в шард и строка индекса
    bytes_in: int = 0            # сколько отдали сжатию
    bytes_out: int = 0           # сколько легло на диск

    def add(self, other: "Cost") -> None:
        self.calls += other.calls
        self.keyframes += other.keyframes
        self.delta_ns += other.delta_ns
        self.compress_ns += other.compress_ns
        self.write_ns += other.write_ns
        self.bytes_in += other.bytes_in
        self.bytes_out += other.bytes_out

    @property
    def total_ns(self) -> int:
        return self.delta_ns + self.compress_ns + self.write_ns

    def as_dict(self) -> dict[str, Any]:
        n = max(1, self.calls)
        return {
            "calls": self.calls, "keyframes": self.keyframes,
            "delta_ms_per_call": round(self.delta_ns / n / 1e6, 3),
            "compress_ms_per_call": round(self.compress_ns / n / 1e6, 3),
            "write_ms_per_call": round(self.write_ns / n / 1e6, 3),
            "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
            "ratio": round(self.bytes_in / self.bytes_out, 2) if self.bytes_out else None,
        }


def _encode_delta(frame: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Разность кадров по модулю 256 со сдвигом 128, в арифметике uint8.

    Считается **в uint8**, а не через int16: арифметика uint8 в numpy переполняется по
    модулю 256 сама, то есть даёт ровно то же число, а обходов массива делает вдвое
    меньше и памяти просит вчетверо меньше.

    Это не украшательство. `TASK-17`, пункт 2: на кадре 1920×1080 прежняя запись через
    int16 стоила 11.6 мс — больше, чем само сжатие на уровне 6 (10.2 мс), — и обе стадии
    идут в основном потоке, то есть прямо ограничивают частоту записи. Замер:
    `docs/measurements/record_cost.json`.

    Тождественность двух записей проверяется тестом на всех 65536 парах значений: если бы
    она была «скорее всего верна», кадры расходились бы при чтении в редких пикселях, и
    это был бы худший вид поломки — незаметный.
    """
    return (frame - prev + np.uint8(128))


def _decode_delta(prev: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Обратное к `_encode_delta`, тоже в uint8."""
    return (delta + prev - np.uint8(128))


class BlobStore:
    """Дозаписываемое хранилище массивов с индексом.

    Открывается в одном из режимов: `"a"` — писать и читать, `"r"` — только
    читать. В режиме `"r"` шарды не открываются на запись вообще, поэтому
    воспроизведение записанной сессии не может её испортить.
    """

    def __init__(self, root: str | Path, mode: Literal["a", "r"] = "r",
                 shard_bytes: int = 64 * 1024 * 1024, compress_level: int = 6) -> None:
        self.root = Path(root)
        self.mode = mode
        self.shard_bytes = int(shard_bytes)
        self.compress_level = int(compress_level)
        self._index: list[BlobRef] = []
        self._index_fh = None
        self._shard_fh = None
        self._shard_no = 0
        self._shard_size = 0
        self._read_cache: dict[int, Any] = {}
        self.cost = Cost()

        if mode == "a":
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise StoreError(f"нет хранилища: {self.root}")

        self._load_index()
        if mode == "a":
            self._index_fh = (self.root / "index.jsonl").open("a", encoding="utf-8")
            self._open_tail_shard()

    # --- индекс -------------------------------------------------------------

    def _load_index(self) -> None:
        path = self.root / "index.jsonl"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ref = BlobRef.from_dict(json.loads(line))
                except Exception as e:
                    raise StoreError(f"{path}:{lineno}: индекс битый: {e}") from e
                if ref.blob_id != len(self._index):
                    raise StoreError(
                        f"{path}:{lineno}: blob_id {ref.blob_id}, ожидался {len(self._index)}. "
                        "Индекс дозаписывается строго по порядку; пропуск означает правку файла"
                    )
                self._index.append(ref)

    def _open_tail_shard(self) -> None:
        shards = sorted(self.root.glob("shard-*.bin"))
        self._shard_no = int(shards[-1].stem.split("-")[1]) if shards else 0
        path = self._shard_path(self._shard_no)
        self._shard_size = path.stat().st_size if path.exists() else 0
        if self._shard_size >= self.shard_bytes:
            self._shard_no += 1
            self._shard_size = 0
        self._shard_fh = self._shard_path(self._shard_no).open("ab")

    def _shard_path(self, no: int) -> Path:
        return self.root / f"shard-{no:05d}.bin"

    # --- запись -------------------------------------------------------------

    def append(self, array: np.ndarray, *, enc: str = "raw",
               base: int | None = None, shape: tuple[int, ...] | None = None,
               dtype: str | None = None) -> BlobRef:
        if self.mode != "a":
            raise StoreError("хранилище открыто только на чтение")
        if not isinstance(array, np.ndarray):
            raise StoreError(f"ожидался numpy-массив, пришло {type(array).__name__}")
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        raw = array.tobytes()
        t0 = time.perf_counter_ns()
        payload = zlib.compress(raw, self.compress_level)
        t1 = time.perf_counter_ns()
        self.cost.calls += 1
        self.cost.compress_ns += t1 - t0
        self.cost.bytes_in += len(raw)
        record = MAGIC + len(payload).to_bytes(4, "big") + payload
        self.cost.bytes_out += len(record)

        if self._shard_size and self._shard_size + len(record) > self.shard_bytes:
            self._shard_fh.close()
            self._shard_no += 1
            self._shard_size = 0
            self._shard_fh = self._shard_path(self._shard_no).open("ab")

        offset = self._shard_size
        t2 = time.perf_counter_ns()
        self._shard_fh.write(record)
        self._shard_size += len(record)

        ref = BlobRef(len(self._index), self._shard_no, offset, len(payload),
                      tuple(shape or array.shape), str(dtype or array.dtype),
                      enc, base)
        self._index.append(ref)
        self._index_fh.write(json.dumps(ref.as_dict(), separators=(",", ":")) + "\n")
        self.cost.write_ns += time.perf_counter_ns() - t2
        return ref

    def flush(self) -> None:
        if self._shard_fh:
            self._shard_fh.flush()
        if self._index_fh:
            self._index_fh.flush()

    # --- чтение -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def ref(self, blob_id: int) -> BlobRef:
        try:
            return self._index[blob_id]
        except IndexError:
            raise StoreError(f"нет блока {blob_id}: в хранилище {len(self._index)}") from None

    def read(self, blob_id: int | BlobRef) -> np.ndarray:
        ref = blob_id if isinstance(blob_id, BlobRef) else self.ref(blob_id)
        if ref.enc != "raw":
            raise StoreError(
                f"блок {ref.blob_id} закодирован как {ref.enc!r}: читать его сырым "
                "нельзя, иначе вернутся разности вместо данных. "
                "Используйте хранилище, которое умеет эту кодировку (FrameStore)")
        return self._read_stored(ref)

    def _read_stored(self, ref: BlobRef) -> np.ndarray:
        """Достать то, что лежит на диске, без раскодирования разностей."""
        fh = self._read_cache.get(ref.shard)
        if fh is None:
            path = self._shard_path(ref.shard)
            if not path.exists():
                raise StoreError(f"нет шарда {path}")
            fh = path.open("rb")
            self._read_cache[ref.shard] = fh
        if self.mode == "a":
            self.flush()
        fh.seek(ref.offset)
        head = fh.read(HEADER_LEN)
        if len(head) != HEADER_LEN or not head.startswith(MAGIC):
            raise StoreError(f"блок {ref.blob_id}: на смещении {ref.offset} нет заголовка блока")
        nbytes = int.from_bytes(head[len(MAGIC):], "big")
        if nbytes != ref.nbytes:
            raise StoreError(f"блок {ref.blob_id}: индекс обещает {ref.nbytes} байт, в шарде {nbytes}")
        raw = zlib.decompress(fh.read(nbytes))
        return np.frombuffer(raw, dtype=np.dtype(ref.dtype)).reshape(ref.shape)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self._index)):
            yield self.read(i)

    def bytes_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("shard-*.bin"))

    # --- жизненный цикл -----------------------------------------------------

    def close(self) -> None:
        for fh in self._read_cache.values():
            fh.close()
        self._read_cache.clear()
        if self._shard_fh:
            self._shard_fh.close()
            self._shard_fh = None
        if self._index_fh:
            self._index_fh.close()
            self._index_fh = None

    def __enter__(self) -> BlobStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FrameStore(BlobStore):
    """Кадры экрана: опорный кадр целиком, остальные — разностью от предыдущего.

    Почему разностью. Захват экрана — цифровой и без потерь: если в кадре ничего
    не изменилось, он побитово совпадает с предыдущим. Разность тогда состоит из
    одного повторяющегося значения и сжимается почти в ноль, тогда как покадровое
    сжатие каждый раз платит за всю картинку заново. Ради этого и существует
    критерий 0.3 «час записи занимает вменяемое место».

    Разность обратима точно: `(cur - prev + 128) mod 256`, обратно
    `(delta + prev - 128) mod 256`. Никакой потери, никакого приближения.

    Чего это не заменяет: при 1080p и 30 кадрах в секунду даже так выходит
    слишком много (замеры — в `docs/ARCHITECTURE-HARNESS.md`, раздел «Сколько
    занимает запись»). Для живой записи нужен видеокодек; здесь его нет,
    потому что проверить его на этой машине нечем — ffmpeg не установлен, а
    непроверенный кодек в хранилище опыта опаснее отсутствующего.

    `keyframe_interval` — через сколько кадров ставить опорный. Он же ограничивает
    цену промотки: чтобы собрать произвольный кадр, читается не больше
    `keyframe_interval` блоков.
    """

    def __init__(self, *args: Any, keyframe_interval: int = 30, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.keyframe_interval = max(1, int(keyframe_interval))
        self._prev_frame: np.ndarray | None = None
        self._prev_id: int | None = None
        self._cache_id: int | None = None
        self._cache_frame: np.ndarray | None = None

    def append_frame(self, frame: np.ndarray) -> BlobRef:
        if frame.dtype != np.uint8:
            raise StoreError(f"кадр должен быть uint8, пришёл {frame.dtype}")
        if frame.ndim not in (2, 3):
            raise StoreError(f"кадр должен быть (h,w) или (h,w,c), пришло {frame.shape}")
        if frame.ndim == 3 and frame.shape[2] not in (3, 4):
            raise StoreError(f"каналов должно быть 3 или 4, пришло {frame.shape[2]}")

        blob_id = len(self._index)
        same_shape = (self._prev_frame is not None
                      and self._prev_frame.shape == frame.shape)
        keyframe = (blob_id % self.keyframe_interval == 0) or not same_shape

        if keyframe:
            self.cost.keyframes += 1
            ref = self.append(frame)
            self._base_id = blob_id
        else:
            # Вычисление разности считается отдельно от сжатия: это две разные работы, и
            # смешав их, нельзя ответить, что именно съело кадр. На 1080p разность — это
            # три обхода массива в 2 МБ (int16, вычитание, обратно в uint8), и обход не
            # бесплатен.
            t0 = time.perf_counter_ns()
            delta = _encode_delta(frame, self._prev_frame)
            self.cost.delta_ns += time.perf_counter_ns() - t0
            ref = self.append(delta, enc="delta", base=self._base_id)
        self._prev_frame = frame.copy()
        self._prev_id = blob_id
        return ref

    _base_id: int = 0

    def read(self, blob_id: int | BlobRef) -> np.ndarray:
        ref = blob_id if isinstance(blob_id, BlobRef) else self.ref(blob_id)
        if ref.enc == "raw":
            return self._read_stored(ref)
        if ref.enc != "delta":
            raise StoreError(f"неизвестная кодировка кадра: {ref.enc!r}")

        # Последовательное чтение — самый частый случай: если предыдущий кадр уже
        # собран, достаточно применить одну разность.
        if self._cache_id == ref.blob_id - 1 and self._cache_frame is not None:
            out = self._apply(self._cache_frame, self._read_stored(ref))
        else:
            base = ref.base if ref.base is not None else 0
            out = self._read_stored(self.ref(base))
            for i in range(base + 1, ref.blob_id + 1):
                out = self._apply(out, self._read_stored(self.ref(i)))
        self._cache_id, self._cache_frame = ref.blob_id, out
        return out

    @staticmethod
    def _apply(prev: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _decode_delta(prev, delta)

    def encoding_stats(self) -> dict[str, int]:
        raw = sum(1 for r in self._index if r.enc == "raw")
        return {"frames": len(self._index), "keyframes": raw,
                "deltas": len(self._index) - raw,
                "keyframe_interval": self.keyframe_interval,
                "bytes": self.bytes_on_disk()}


class AudioStore(BlobStore):
    """Блоки звука. Стерео обязательно: из разницы каналов считается пеленг."""

    def append_block(self, block: np.ndarray, *, channels: int = 2) -> BlobRef:
        if block.dtype != np.int16:
            raise StoreError(f"звук должен быть int16, пришёл {block.dtype}")
        if block.ndim != 2 or block.shape[1] != channels:
            raise StoreError(
                f"блок звука должен быть (samples, {channels}), пришло {block.shape}. "
                "Моно ломает пеленг и потому не принимается"
            )
        return self.append(block)
