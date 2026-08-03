"""Инъекция ввода: маска, СТОП, сторожевой таймер."""

from .base import InjectionSink, InjectionUnavailable
from .mask import InputMask
from .stop import StopSwitch
from .watchdog import Watchdog

__all__ = ["InjectionSink", "InjectionUnavailable", "InputMask", "StopSwitch", "Watchdog"]
