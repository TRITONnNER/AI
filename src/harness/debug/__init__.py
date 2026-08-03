"""Поток исследователя. Агентская сторона сюда не импортируется — инвариант 12."""

from .channel import DebugChannel, DebugChannelError

__all__ = ["DebugChannel", "DebugChannelError"]
