"""Timm feature backbone with explicit stage-level freezing controls."""

from __future__ import annotations

from typing import Iterable

from ...core import register
from .timm_model import TimmModel


@register()
class SelectiveTimmModel(TimmModel):
    """TimmModel variant used by controlled backbone-transfer experiments."""

    def __init__(
        self,
        *args,
        freeze_all=False,
        trainable_prefixes=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freeze_all = bool(freeze_all)
        self.trainable_prefixes = tuple(
            self._parameter_prefix(prefix) for prefix in (trainable_prefixes or ())
        )
        self.out_channels = list(self.channels)
        self.out_strides = list(self.strides)
        self._configure_trainability()

    def _is_trainable(self, name: str) -> bool:
        if self.freeze_all:
            return False
        if not self.trainable_prefixes:
            return True
        return any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in self.trainable_prefixes
        )

    def _configure_trainability(self) -> None:
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self._is_trainable(name))

    @staticmethod
    def _parameter_prefix(prefix: str) -> str:
        if prefix.startswith("layers."):
            first, index, *rest = prefix.split(".")
            prefix = f"{first}_{index}"
            if rest:
                prefix += "." + ".".join(rest)
        return prefix

    @staticmethod
    def _top_level_prefixes(names: Iterable[str]) -> set[str]:
        return {name.split(".")[0] for name in names}

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            return self
        if self.freeze_all:
            self.model.eval()
            return self
        if not self.trainable_prefixes:
            return self

        trainable_roots = self._top_level_prefixes(self.trainable_prefixes)
        for name, module in self.model.named_children():
            if name not in trainable_roots:
                module.eval()
        return self
