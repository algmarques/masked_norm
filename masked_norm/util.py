"""
Utilities module
"""

from torch import Tensor


def unsqueeze_as_(dst: Tensor, src: Tensor) -> None:
    """
    Unsqueezes singleton dim axis according to the shape of a source tensor.
    """

    diff = len(src.shape) - len(dst.shape)

    if not diff > 0:
        raise ValueError(
            "the shape of 'dst' must be larger than the shape of 'src'"
        )

    for i in range(diff):
        dst.unsqueeze_(-1)


def get_factory_key(key: str) -> bool:
    """
    Filtering procedure for selecting factory kwargs.
    """

    if key == "device":
        return True

    if key == "dtype":
        return True

    return False
