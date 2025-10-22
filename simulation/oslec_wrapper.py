"""
Python ctypes bindings for the OSLEC echo canceller.

The wrapper keeps the original C sources untouched and exposes a small,
idiomatic API for simulation and analysis workflows.
"""
from __future__ import annotations

import ctypes
import enum
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LOG = logging.getLogger(__name__)


class OSLECError(RuntimeError):
    """Raised when the underlying OSLEC library cannot be used."""


class AdaptionMode(enum.IntFlag):
    """Bit masks mirroring the adaption flags from oslec.h."""

    USE_ADAPTION = 0x01
    USE_NLP = 0x02
    USE_CNG = 0x04
    USE_CLIP = 0x08
    USE_TX_HPF = 0x10
    USE_RX_HPF = 0x20
    DISABLE = 0x40

    @classmethod
    def default(cls) -> "AdaptionMode":
        return (
            cls.USE_ADAPTION
            | cls.USE_NLP
            | cls.USE_CNG
            | cls.USE_CLIP
            | cls.USE_TX_HPF
            | cls.USE_RX_HPF
        )


def _default_library_path() -> Path:
    candidate = Path(__file__).resolve().parent / "liboslec.so"
    if not candidate.exists():
        raise OSLECError(
            "liboslec.so not found. Build it with `make` inside the "
            "`simulation/` directory before running the simulations."
        )
    return candidate


_lib = None


def _load_library(path: Optional[Path] = None) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    library_path = Path(path) if path else _default_library_path()
    _lib = ctypes.CDLL(str(library_path))

    _lib.oslec_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _lib.oslec_create.restype = ctypes.c_void_p

    _lib.oslec_free.argtypes = [ctypes.c_void_p]
    _lib.oslec_free.restype = None

    _lib.oslec_adaption_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.oslec_adaption_mode.restype = None

    _lib.oslec_flush.argtypes = [ctypes.c_void_p]
    _lib.oslec_flush.restype = None

    _lib.oslec_update.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int16,
        ctypes.c_int16,
    ]
    _lib.oslec_update.restype = ctypes.c_int16

    _lib.oslec_hpf_tx.argtypes = [ctypes.c_void_p, ctypes.c_int16]
    _lib.oslec_hpf_tx.restype = ctypes.c_int16

    return _lib


def _to_int16(value: float) -> int:
    clipped = max(min(int(round(value)), 32767), -32768)
    return ctypes.c_int16(clipped).value


class OSLEC:
    """High-level manager for an OSLEC instance."""

    def __init__(
        self,
        taps: int,
        adaption_mode: AdaptionMode = AdaptionMode.default(),
        library_path: Optional[Path] = None,
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be a positive integer")

        self._lib = _load_library(library_path)

        state = self._lib.oslec_create(int(taps), int(adaption_mode))
        if not state:
            raise OSLECError("oslec_create() returned NULL")

        self._state = ctypes.c_void_p(state)
        self._closed = False
        LOG.debug(
            "Created OSLEC instance: taps=%d adaption_mode=0x%02x",
            taps,
            int(adaption_mode),
        )

    def __enter__(self) -> "OSLEC":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise OSLECError("OSLEC instance closed")

    def close(self) -> None:
        if not self._closed and self._state:
            self._lib.oslec_free(self._state)
            self._state = ctypes.c_void_p()
            self._closed = True
            LOG.debug("Freed OSLEC instance")

    def flush(self) -> None:
        self._ensure_open()
        self._lib.oslec_flush(self._state)
        LOG.debug("Flushed OSLEC state")

    def set_adaption_mode(self, adaption_mode: AdaptionMode) -> None:
        self._ensure_open()
        self._lib.oslec_adaption_mode(self._state, int(adaption_mode))
        LOG.debug("Set adaption mode to 0x%02x", int(adaption_mode))

    def update(self, tx: int, rx: int) -> int:
        self._ensure_open()
        tx_i16 = _to_int16(tx)
        rx_i16 = _to_int16(rx)
        return int(self._lib.oslec_update(self._state, tx_i16, rx_i16))

    def hpf_tx(self, tx: int) -> int:
        self._ensure_open()
        tx_i16 = _to_int16(tx)
        return int(self._lib.oslec_hpf_tx(self._state, tx_i16))

    def process(
        self, tx_samples: Sequence[int], rx_samples: Sequence[int]
    ) -> np.ndarray:
        """Process aligned transmit and receive sequences sample by sample."""
        self._ensure_open()

        if len(tx_samples) != len(rx_samples):
            raise ValueError("tx_samples and rx_samples must share a length")

        output = np.empty(len(tx_samples), dtype=np.int16)
        LOG.debug("Processing %d samples through OSLEC", len(tx_samples))
        for idx, (tx, rx) in enumerate(zip(tx_samples, rx_samples)):
            output[idx] = self.update(tx, rx)
        return output

    def adaptive_process(
        self,
        tx_stream: Iterable[int],
        rx_stream: Iterable[int],
    ) -> Iterable[int]:
        """
        Lazily process streams of samples.

        This is useful when driving the canceller with generators that produce
        very long sequences.
        """
        self._ensure_open()
        LOG.debug("Starting adaptive processing stream")
        for tx, rx in zip(tx_stream, rx_stream):
            yield self.update(tx, rx)


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """
    Convert floating point audio in [-1.0, 1.0) to int16 PCM.
    """
    scaled = np.clip(samples, -1.0, 0.999969482421875) * 32768.0
    return scaled.astype(np.int16)


def pcm16_to_float(samples: Sequence[int]) -> np.ndarray:
    """
    Convert int16 PCM samples to floating point in [-1.0, 1.0).
    """
    return np.asarray(samples, dtype=np.float32) / 32768.0
