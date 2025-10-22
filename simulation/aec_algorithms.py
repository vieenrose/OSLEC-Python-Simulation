"""
Reference acoustic echo cancellation algorithms for benchmarking.

All algorithms expose a simple class interface with a ``process`` method that
accepts transmit (`tx`) and microphone (`mic`) signals and returns the residual
echo-cancelled signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from oslec_wrapper import OSLEC


class BaseAEC:
    def reset(self) -> None:
        raise NotImplementedError

    def process(self, tx: Sequence[float], mic: Sequence[float]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NLMSParams:
    taps: int
    mu: float = 0.6
    epsilon: float = 1e-6


class NLMSAEC(BaseAEC):
    """Standard NLMS implementation for reference."""

    def __init__(self, params: NLMSParams):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros(self.params.taps, dtype=np.float64)
        self.buffer = np.zeros(self.params.taps, dtype=np.float64)

    def process(self, tx: Sequence[float], mic: Sequence[float]) -> np.ndarray:
        tx_arr = np.asarray(tx, dtype=np.float64)
        mic_arr = np.asarray(mic, dtype=np.float64)
        output = np.empty_like(tx_arr)
        for idx in range(len(tx_arr)):
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = tx_arr[idx]
            y = np.dot(self.weights, self.buffer)
            err = mic_arr[idx] - y
            norm = np.dot(self.buffer, self.buffer) + self.params.epsilon
            step = (self.params.mu / norm) * err
            self.weights += step * self.buffer
            output[idx] = err
        return output


@dataclass
class IPNLMSParams:
    taps: int
    mu: float = 0.5
    alpha: float = 0.5
    epsilon: float = 1e-6


class IPNLMSAEC(BaseAEC):
    """
    Improved Proportionate NLMS (IPNLMS).
    """

    def __init__(self, params: IPNLMSParams):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros(self.params.taps, dtype=np.float64)
        self.buffer = np.zeros(self.params.taps, dtype=np.float64)

    def process(self, tx: Sequence[float], mic: Sequence[float]) -> np.ndarray:
        tx_arr = np.asarray(tx, dtype=np.float64)
        mic_arr = np.asarray(mic, dtype=np.float64)
        output = np.empty_like(tx_arr)
        alpha = self.params.alpha
        taps = self.params.taps
        epsilon = self.params.epsilon
        mu = self.params.mu

        for idx in range(len(tx_arr)):
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = tx_arr[idx]
            y = np.dot(self.weights, self.buffer)
            err = mic_arr[idx] - y

            abs_w = np.abs(self.weights)
            sum_abs = np.sum(abs_w) + epsilon
            g = (1 - alpha) / (2 * taps) + (1 + alpha) * abs_w / (2 * sum_abs)
            norm = np.dot(g * self.buffer, self.buffer) + epsilon
            step_vec = (mu * err * g * self.buffer) / norm
            self.weights += step_vec
            output[idx] = err
        return output


@dataclass
class MDFParams:
    taps: int
    block_size: int = 128
    mu: float = 0.5
    epsilon: float = 1e-6


class MDFAEC(BaseAEC):
    """
    Partitioned Block Frequency-Domain NLMS (simplified MDF).
    """

    def __init__(self, params: MDFParams):
        if params.block_size & (params.block_size - 1):
            raise ValueError("block_size must be a power of two")
        self.params = params
        self.num_partitions = int(
            np.ceil(params.taps / params.block_size)
        )
        self.fft_len = params.block_size * 2
        self.reset()

    def reset(self) -> None:
        self.freq_bins = self.fft_len // 2 + 1
        self.freq_weights = np.zeros(
            (self.num_partitions, self.freq_bins), dtype=np.complex128
        )
        self.buffer_blocks = np.zeros(
            (self.num_partitions, self.freq_bins), dtype=np.complex128
        )
        self.time_buffer = np.zeros(self.params.block_size, dtype=np.float64)
        self.error_buffer = np.zeros(self.params.block_size, dtype=np.float64)

    def process(self, tx: Sequence[float], mic: Sequence[float]) -> np.ndarray:
        tx_arr = np.asarray(tx, dtype=np.float64)
        mic_arr = np.asarray(mic, dtype=np.float64)
        block = self.params.block_size
        out = np.empty_like(tx_arr)
        fft_len = self.fft_len
        mu = self.params.mu
        epsilon = self.params.epsilon

        for start in range(0, len(tx_arr), block):
            end = min(start + block, len(tx_arr))
            frame_len = end - start
            tx_block = np.zeros(block, dtype=np.float64)
            mic_block = np.zeros(block, dtype=np.float64)
            tx_block[:frame_len] = tx_arr[start:end]
            mic_block[:frame_len] = mic_arr[start:end]

            # Update circular buffer of input blocks
            self.buffer_blocks[1:] = self.buffer_blocks[:-1]
            x_fft_in = np.concatenate([tx_block, np.zeros(block)])
            x_freq = np.fft.rfft(x_fft_in)
            self.buffer_blocks[0, :] = x_freq

            # Estimate echo
            y_freq = np.sum(self.freq_weights * self.buffer_blocks, axis=0)
            y_time = np.fft.irfft(y_freq, n=fft_len)
            y_block = y_time[block:fft_len]

            err_block = mic_block - y_block[:block]
            out[start:end] = err_block[:frame_len]

            # Adaptive update
            err_fft_in = np.concatenate([np.zeros(block), err_block])
            err_fft = np.fft.rfft(err_fft_in)
            energy = np.sum(
                np.abs(self.buffer_blocks) ** 2, axis=0
            ) + epsilon
            update = mu * err_fft * np.conj(self.buffer_blocks[0]) / energy
            self.freq_weights[0] += update

        return out


class OSLECWrapper(BaseAEC):
    """Adapter to measure the existing OSLEC implementation."""

    def __init__(self, taps: int):
        self.taps = taps

    def reset(self) -> None:
        pass

    def process(self, tx: Sequence[float], mic: Sequence[float]) -> np.ndarray:
        tx_pcm = (np.asarray(tx) * 32768.0).astype(np.int16)
        mic_pcm = (np.asarray(mic) * 32768.0).astype(np.int16)
        with OSLEC(self.taps) as canceller:
            residual = canceller.process(tx_pcm, mic_pcm)
        return residual.astype(np.float64) / 32768.0
