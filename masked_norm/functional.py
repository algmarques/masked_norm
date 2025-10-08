"""
Functional module

Contains the functional implementation of masked normalization.
"""

from __future__ import annotations
from typing import Optional

from torch import Tensor

from .validation import validate_masked_norm
from .validation import validate_affine_masked_norm
from .util import unsqueeze_as_


def masked_norm(
    inpt: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Masked normalization procedure.

    Normalizes elements of the input specified by a mask. If no mask is
    passed to the routine, normalization is performed along the last axis.
    If either by chance or design a selection of samples yields null
    variance, the normalization over such selection is ignored, and the
    values are passed along unaltered.
    """

    validate_masked_norm(inpt, mask)

    mean = inpt.mean(dim=-1, keepdim=True)
    var = inpt.var(dim=-1, keepdim=True)

    var_mask = (var != 0.0)
    unsqueeze_as_(mask, var_mask)
    mask |= var_mask

    norm = inpt[mask] - mean[mask]
    norm = norm / var[mask].sqrt()

    inpt.masked_scatter_(mask, norm)

    return inpt


def affine_masked_norm(
    inpt: Tensor,
    mask: Optional[Tensor],
    weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    """
    Affine masked normalization procedure.

    Normalizes elements of the input specified by a mask. If no mask is
    passed to the routine, normalization is performed along the last axis.
    If either by chance or design a collection of samples yields null
    variance, the normalization over such collection is ignored, and the
    values are passed along unaltered. After normalization an affine
    transformation is applied along the normalized dimensions.
    """

    validate_affine_masked_norm(inpt, mask, weight, bias)

    return weight * masked_norm(inpt, mask) + bias
