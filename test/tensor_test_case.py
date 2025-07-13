"""
"""

from __future__ import annotations

from unittest import TestCase

from torch import Tensor


class TensorTestCase(TestCase):
    """
    Test Case with special equality assertion for torch Tensor
    """

    def assertEqTensor(
        self: TensorTestCase,
        x: Tensor | None,
        y: Tensor | None,
        msg: str | None = None
    ) -> None:
        """
        Tensor equality assertion method
        """

        cond = (x == y)
        if isinstance(cond, Tensor):
            cond = bool(cond.all())

        self.assertTrue(cond, msg)
