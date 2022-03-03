#!/usr/bin/env python3
# Copyright 2022 Lucky Wong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch


class Pcen(torch.nn.Module):
    """Trainable per-channel energy normalization (PCEN).

      This applies a fixed or learnable normalization by an exponential moving
      average smoother, and a compression.
      See https://arxiv.org/abs/1607.05666 for more details.
    """

    def __init__(self,
                 feat_dim: int,
                 alpha: float = 0.96,
                 smooth_coef: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6):
        """PCEN constructor.

        Args:
          feat_dim: int, feature dims
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
        """
        super().__init__()

        self.floor = floor

        self.smooth = torch.nn.Parameter(torch.Tensor(feat_dim))
        torch.nn.init.constant_(self.smooth, coeff_init)

        # The AGC strength (or gain normalization strength) is controlled by the parameter α ∈ [0, 1]
        self.alpha = torch.nn.Parameter(torch.Tensor(feat_dim))
        torch.nn.init.constant_(self.alpha, alpha)

        # A stabilized root compression to further reduce the dynamic range offset δ and exponent r
        self.delta = torch.nn.Parameter(torch.Tensor(feat_dim))
        torch.nn.init.constant_(self.delta, delta)

        self.root = torch.nn.Parameter(torch.Tensor(feat_dim))
        torch.nn.init.constant_(self.root, root)

    def apply_iir(self, x):
        """Implements a first order Infinite Impulse Response (IIR) forward filter initialized using the input values.
        :param x (torch.tensor): batch of (mel-) spectrograms. shape: [..., Frequency, Time]
        :return M: Low-pass filtered version of the input spectrograms.
        """
        s = torch.clamp(self.smooth, min=0.0, max=1.0)

        M = [x[..., 0]]
        for t in range(1, x.size(-1)):
            m = (1. - s) * M[-1] + s * x[..., t]
            M.append(m)
        M = torch.stack(M, dim=-1)

        return M

    def forward(self,
                xs: torch.Tensor,
                xs_mask: torch.Tensor
                ):
        """
        :param xs:      Input tensor (#batch, time, idim).
        :param xs_mask: Input mask (#batch, 1, time).
        :return:
        """
        alpha = torch.min(self.alpha, torch.ones(
            self.alpha.size(), device=self.alpha.device))
        root = torch.max(self.root, torch.ones(
            self.root.size(), device=self.root.device))

        # exchange the temporal dimension and the feature dimension
        xs = xs.transpose(1, 2)

        # mask batch padding
        if mask_pad is not None:
            xs.masked_fill_(~mask_pad, 0.0)

        xs = self.apply_iir(xs)

        # mask batch padding
        if mask_pad is not None:
            xs.masked_fill_(~mask_pad, 0.0)

        ema_smoother = xs.transpose(1, 2)

        one_over_root = 1. / root

        xs = ((xs / (self.floor + ema_smoother)**alpha + self.delta)**one_over_root
              - self.delta**one_over_root)

        return xs
