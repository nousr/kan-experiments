from typing import Tuple
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange

from .spline import curve2coef, coef2curve


@beartype
@dataclass
class KANLayerArguments:
    """
    Arguments for a single KAN Layer
    """

    in_features: int
    out_features: int
    order: int = 3
    init_noise: float = 0.1
    grid_intervals: int = 5
    grid_eps: float = 0
    grid_max: int = 1
    grid_min: int = -1
    spline_scale: float = 1.0
    spline_trainable: bool = True
    residual_func: Callable = torch.nn.SiLU()
    residual_scale: float = 1.0
    residual_scale_trainable: bool = True
    device: torch.device = None
    dtype: torch.dtype = None

    @property
    def size(self):
        return self.in_features * self.out_features


class KANLayer(nn.Module):
    """
    A single KAN Layer.

    Args:
        layer_args: KanLayerArguments
    """

    @beartype
    def __init__(self, layer_args: KANLayerArguments):
        super().__init__()
        self.layer_args = layer_args

        self.grid = torch.einsum(
            "i,j->ij",
            torch.ones(
                self.layer_args.size,
            ),
            torch.linspace(
                self.layer_args.grid_min,
                self.layer_args.grid_max,
                steps=self.layer_args.grid_intervals + 1,
            ),
        )
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)

        noises = (
            (
                torch.rand(
                    self.layer_args.size,
                    self.grid.size(1),
                )
                - 1 / 2
            )
            * self.layer_args.init_noise
            / self.layer_args.grid_intervals
        )

        self.coef = torch.nn.Parameter(
            curve2coef(
                self.grid,
                noises,
                self.grid,
                self.layer_args.order,
            )
        )
        self.residual_scale = torch.nn.Parameter(
            torch.full(
                (self.layer_args.size,),
                self.layer_args.residual_scale,
            )
        ).requires_grad_(self.layer_args.residual_scale_trainable)

        self.spline_scale = torch.nn.Parameter(
            torch.full(
                (self.layer_args.size,),
                self.layer_args.spline_scale,
            )
        ).requires_grad_(self.layer_args.spline_trainable)

        self.residual_func = self.layer_args.residual_func

        self.mask = torch.nn.Parameter(
            torch.ones(
                self.layer_args.size,
            )
        ).requires_grad_(False)
        self.weight_sharing = torch.arange(
            self.layer_args.size,
            dtype=torch.int32,
        )
        self.lock_counter = nn.Parameter(
            torch.tensor([0], dtype=torch.int32),
            requires_grad=False,
        )
        self.lock_id = torch.zeros(self.layer_args.size)

    @beartype
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, o, i, s = (
            x.size(0),
            self.layer_args.out_features,
            self.layer_args.in_features,
            self.layer_args.size,
        )

        pre_activation = torch.einsum(
            "ij,k->ikj",
            x,
            torch.ones(
                self.layer_args.out_features,
            ),
        )

        x = rearrange(
            pre_activation,
            "b o i -> (o i) b",
            b=b,
            o=o,
            i=i,
        )

        base = self.residual_func(x)
        base = rearrange(base, "s b -> b s", s=s, b=b)
        y = coef2curve(
            x_eval=x,
            grid=self.grid[self.weight_sharing],
            coef=self.coef[self.weight_sharing],
            k=self.layer_args.order,
        )
        y = rearrange(y, "s b -> b s", s=s, b=b)

        post_spline = rearrange(y.clone(), "b (o i) -> b o i", b=b, o=o, i=i)

        y = (
            self.residual_scale.unsqueeze(dim=0) * base
            + self.spline_scale.unsqueeze(dim=0) * y
        )
        y = self.mask.unsqueeze(dim=0) * y
        y = rearrange(y, "b (o i) -> b o i", b=b, o=o, i=i)

        post_activation = y.clone()

        y = y.sum(dim=2)

        return y, pre_activation, post_activation, post_spline
