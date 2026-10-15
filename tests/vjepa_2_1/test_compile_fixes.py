# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``compile_fixes.patch_inductor_multiple_of``: the exact ``sympy.cancel`` fallback for Inductor's
``SizeVarAllocator.statically_known_multiple_of`` (the ``InductorError: CantSplit`` workaround).

The end-to-end crash only reproduces with GPU Triton codegen, so this pins the decision itself:
- the expressions from the failing cluster runs (4502603 / 4505759) are rejected by stock Inductor and
  accepted with the patch;
- everything that is not provably a multiple stays rejected (the patch must never let Inductor emit
  indexing for a split that does not divide);
- Inductor's integer-denominator path and its already-true answers are untouched;
- installation is idempotent.
Symbols are made like Inductor's size symbols (``SymT.SIZE``, integer, nonnegative).
"""

import unittest

import sympy
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor.sizevars import SizeVarAllocator
from torch.utils._sympy.symbol import SymT, make_symbol

from app.vjepa_2_1.compile_fixes import exact_multiple_of, patch_inductor_multiple_of


def size_symbol(idx):
    return make_symbol(SymT.SIZE, idx, integer=True, nonnegative=True)


s3, s69, s71, s89 = (size_symbol(i) for i in (3, 69, 71, 89))

# (numerator, denominator) straight from the cluster logs: 48 x 384 x (n_ctxt + n_pred) vs 48 x (n_ctxt + n_pred)
FAILING_SPLITS = [
    (18432 * s3 + 18432 * s89, 48 * s3 + 48 * s89),  # jobs 4502603 / 4503470 / 4505144
    (18432 * s69 + 18432 * s71, 48 * s69 + 48 * s71),  # job 4505759 (resumed)
]
# must stay False: not divisible, or divisible only for some values of the symbols
NOT_PROVABLY_MULTIPLE = [
    (100 * s3 + 7, 48 * s3),
    (18432 * s3 + 18432 * s89, 48 * s3 + 96 * s89),  # quotient (384*s3 + 384*s89)/(s3 + 2*s89)
    (s3 * s89, s3 + s89),
    (s3, 2 * s3 + 1),
    (48 * s3 + 24, 48 * s3),  # 1 + 1/(2*s3)
    (s3, 2),  # integer denominator: Inductor's own exact check
]


class CompileFixTestBase(unittest.TestCase):
    def setUp(self):
        self._original = SizeVarAllocator.statically_known_multiple_of
        while hasattr(self._original, "_vjepa_original"):  # another test module may have patched already
            self._original = self._original._vjepa_original
        SizeVarAllocator.statically_known_multiple_of = self._original
        self.sizevars = SizeVarAllocator(ShapeEnv())

    def tearDown(self):
        SizeVarAllocator.statically_known_multiple_of = self._original


class ExactMultipleOfTest(CompileFixTestBase):
    def test_failing_splits_are_exact_multiples(self):
        for num, den in FAILING_SPLITS:
            with self.subTest(num=num):
                self.assertTrue(exact_multiple_of(num, den))
                self.assertEqual(sympy.cancel(num / den), 384)

    def test_non_multiples_are_rejected(self):
        for num, den in NOT_PROVABLY_MULTIPLE:
            with self.subTest(num=num, den=den):
                self.assertFalse(exact_multiple_of(num, den))

    def test_degenerate_inputs(self):
        self.assertFalse(exact_multiple_of(s3, 0))
        self.assertTrue(exact_multiple_of(384, 48))
        self.assertTrue(exact_multiple_of(s3 * s89 + s3, s3))  # s89 + 1
        many = sum(size_symbol(100 + i) for i in range(25))
        self.assertFalse(exact_multiple_of(2 * many, many))  # over the expensive-sympy symbol limit


class PatchedSizeVarsTest(CompileFixTestBase):
    def test_stock_inductor_rejects_the_failing_splits(self):
        """Documents the bug: if this starts failing, the torch version fixed it and the patch can go."""
        for num, den in FAILING_SPLITS:
            with self.subTest(num=num):
                self.assertFalse(self.sizevars.statically_known_multiple_of(num, den))

    def test_patch_accepts_failing_splits_and_nothing_else(self):
        self.assertTrue(patch_inductor_multiple_of())
        for num, den in FAILING_SPLITS:
            with self.subTest(num=num):
                self.assertTrue(self.sizevars.statically_known_multiple_of(num, den))
        for num, den in NOT_PROVABLY_MULTIPLE:
            with self.subTest(num=num, den=den):
                self.assertFalse(self.sizevars.statically_known_multiple_of(num, den))

    def test_patch_keeps_every_stock_answer_that_was_true(self):
        cases = [(48 * s3, 48 * s3), (18432 * s3, 48 * s3), (48 * s3, 16), (384, 48), (s3 * s89, s3)]
        stock = [self.sizevars.statically_known_multiple_of(n, d) for n, d in cases]
        patch_inductor_multiple_of()
        patched = [self.sizevars.statically_known_multiple_of(n, d) for n, d in cases]
        for (n, d), before, after in zip(cases, stock, patched):
            with self.subTest(num=n, den=d):
                if before:
                    self.assertTrue(after)

    def test_install_is_idempotent(self):
        self.assertTrue(patch_inductor_multiple_of())
        installed = SizeVarAllocator.statically_known_multiple_of
        self.assertTrue(patch_inductor_multiple_of())
        self.assertIs(SizeVarAllocator.statically_known_multiple_of, installed)
        self.assertIs(installed._vjepa_original, self._original)


if __name__ == "__main__":
    unittest.main()
