import logging
from typing import ClassVar, Generic, TypeVar

import torch
from torch import nn

logger = logging.getLogger(__name__)


Prepared = TypeVar("Prepared")  # represents the prepared condition input type.


class TokenizedText:
    tokens: torch.Tensor

    _pool: ClassVar[list["TokenizedText"]] = []
    _pool_max: ClassVar[int] = 32

    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens

    @classmethod
    def acquire(cls, tokens: torch.Tensor) -> "TokenizedText":
        if cls._pool:
            instance = cls._pool.pop()
            instance.tokens = tokens
            return instance
        return cls(tokens)

    @classmethod
    def release(cls, instance: "TokenizedText") -> None:
        if len(cls._pool) < cls._pool_max:
            cls._pool.append(instance)


class BaseConditioner(nn.Module, Generic[Prepared]):
    """Base model for all conditioner modules.

    Args:
        dim (int): internal dim of the model.
        output_dim (int): Output dim of the conditioner.
        force_linear (bool, optional): Force linear projection even when `dim == output_dim`.
        output_bias (bool): if True, the output projection will have a bias.
        learn_padding (bool): if True, the padding value will be learnt, zero otherwise.
    """

    def __init__(
        self, dim: int, output_dim: int, output_bias: bool = False, force_linear: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        assert force_linear or dim != output_dim
        assert not output_bias

    def forward(self, inputs: TokenizedText) -> torch.Tensor:
        return self._get_condition(inputs)
