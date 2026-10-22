"""Explicit device execution sizes; never changes CPU proofs or quadrature."""
from dataclasses import dataclass

from ..validation import integer_at_least


@dataclass(frozen=True)
class DeviceBatchConfig:
    tile_size: int = 32768
    screen_batch_tiles: int = 8

    def __post_init__(self):
        for name in ('tile_size', 'screen_batch_tiles'):
            object.__setattr__(self, name, integer_at_least(getattr(self, name), name))

    @property
    def policy_id(self):
        return f'device-batch-fused-v1-tile{self.tile_size}-screen{self.screen_batch_tiles}'
