from pathlib import Path
from typing import Literal

import torch


def get_checkpoint_path(
    directory: Path,
    mode: Literal['last', 'best'] = 'best',
) -> Path:
    if mode == 'last':
        return directory.joinpath('last.ckpt')
    return max(directory.glob('epoch=*-step=*.ckpt'))


def get_checkpoint_epoch(path: Path) -> int:
    return torch.load(path, map_location='cpu')['epoch']
