"""Источники кадров и звука."""

from .base import BackendUnavailable, CaptureSource, Frame, describe_backends

__all__ = ["BackendUnavailable", "CaptureSource", "Frame", "describe_backends"]
