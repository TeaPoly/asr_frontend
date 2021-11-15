#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Model definitions Compute mel-filterbanks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import math

'''
Some code borrowed from open source code in TORCHAUDIO.COMPLIANCE.KALDI. 
https://pytorch.org/audio/stable/compliance.kaldi.html#torchaudio.compliance.kaldi.fbank
'''

def log_compression(inputs: torch.Tensor,
                    log_offset: float = 1.) -> torch.Tensor:
    """Compress an inputs tensor with using a logarithm."""
    return torch.log(inputs + log_offset)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: torch.Tensor) -> torch.torch.Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def get_mel_banks(num_bins: int,
                  window_length_padded: int,
                  sample_freq: float,
                  low_freq: float,
                  high_freq: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        (Tensor, Tensor): The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """
    assert num_bins > 3, 'Must have at least 3 mel bins'
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), \
        ('Bad values in options: low-freq {} and high-freq {} vs. nyquist {}'.format(low_freq, high_freq, nyquist))

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * \
        mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * \
        mel_freq_delta  # size(num_bins, 1)

    # center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
    bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))

    return bins


class MelFilterbanks(nn.Module):
    """Computes mel-filterbanks."""

    def __init__(self,
                 n_filters: int = 80,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 window_len: float = 32.,
                 window_stride: float = 10.,
                 compression_fn=log_compression,
                 min_freq: float = 100.0,
                 max_freq: float = 6800.0):
        """Constructor of a MelFilterbanks frontend.

        Args:
          n_filters: the number of mel_filters.
          sample_rate: sampling rate of input waveforms, in samples.
          n_fft: number of frequency bins of the spectrogram.
          window_len: size of the window, in seconds.
          window_stride: stride of the window, in seconds.
          compression_fn: a callable, the compression function to use.
          min_freq: minimum frequency spanned by mel-filters (in Hz).
          max_freq: maximum frequency spanned by mel-filters (in Hz).
        """
        super().__init__()

        self._n_filters = n_filters
        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._window_len = int(sample_rate * window_len // 1000)
        self._window_stride = int(sample_rate * window_stride // 1000)
        self._compression_fn = compression_fn
        self._min_freq = min_freq
        self._max_freq = max_freq if max_freq else sample_rate / 2.
        self._preemph = 0.97

        assert (self._max_freq <= self._sample_rate //
                2), (self._max_freq, self._sample_rate // 2)

        self.mel_filters = get_mel_banks(self._n_filters,
                                         self._n_fft,
                                         self._sample_rate,
                                         self._min_freq,
                                         self._max_freq).T

        self.window = torch.hamming_window(
            self._window_len, dtype=self.mel_filters.dtype)

    def forward(
            self,
            inputs: torch.Tensor,
            lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mel-filterbanks of a batch of waveforms.

        Args:
            inputs (torch.Tensor): input audio of shape (batch, samples).

        Returns:
            torch.Tensor: (Log) Mel-filterbanks, (batch, frames, bins).
        """
        # Step 1: Expands signal into frames.
        # shape: [batch, time, _frame_size]
        framed_signal = inputs.unfold(1, self._window_len, self._window_stride)

        # Step 2: Remove DC offset.
        remove_dc_framed_signal = framed_signal - \
            torch.unsqueeze(torch.mean(framed_signal, dim=-1), -1)

        # Step 3: Pre-emphasis.
        prev_frame = torch.cat(
            [torch.unsqueeze(remove_dc_framed_signal[:, :, 0], -1),
             remove_dc_framed_signal[:, :, 0:-1]],
            dim=-1)
        preemphasized = remove_dc_framed_signal - self._preemph * prev_frame

        # Step 4: Apply window fn.
        windowed_signal = preemphasized * self.window.to(device=inputs.device)

        # Step 5: FFT.
        real_frequency_spectrogram = torch.fft.rfft(
            windowed_signal, n=self._n_fft)

        magnitude_spectrogram = torch.square(
            torch.abs(real_frequency_spectrogram))

        # Step 6: Linear scale spectrograms to the mel scale.
        # Shape of magnitude_spectrogram is num_frames x (fft_size/2+1)
        # Mel_weight is [num_spectrogram_bins, num_mel_bins]
        # Weight matrix implemented in the magnitude domain.
        mel_spectrogram = torch.matmul(
            magnitude_spectrogram[:, :, :self._n_fft//2],
            self.mel_filters.to(device=inputs.device)
        )

        hlens = None
        if isinstance(lens, torch.Tensor):
            size = (lens-self._window_len)/self._window_stride+1
            hlens = size.to(dtype=lens.dtype)

        if not isinstance(self._compression_fn, type(None)):
            return self._compression_fn(mel_spectrogram), hlens

        return mel_spectrogram, hlens

    def inference(
        self,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """ Inference

        Args:
            xs (torch.Tensor): (T, eunits)

        Returns:
            Tensor: The sequences of encoder states(T, eunits).
        """
        xs = xs.unsqueeze(0)
        ilens = torch.tensor([xs.size(1)])

        return self.forward(xs, ilens)[0][0]

