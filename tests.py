from unittest import TestCase

import numpy as np
from hypothesis import given, strategies as st

from aitkens import accelerate


class TestAitkens(TestCase):
    @given(
        st.floats(min_value=-1e9, max_value=1e9),
        st.floats(min_value=-1, max_value=1).filter(
            lambda initial: abs(initial > 1e-6)
        ),
        st.floats(min_value=1e-2, max_value=1e8),
    )
    def test_geometric_decay(self, limit, initial, rate):
        """Geometrically (exponentially) converging sequences converge
        instantly when accelerated.
        """
        lst = limit + initial * np.exp(-rate * np.array([0, 1, 2]))
        accelerated_lst = accelerate(lst)
        np.testing.assert_allclose(accelerated_lst, [limit], atol=1/rate**2)

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_handles_constant_sequence(self, val):
        lst = [val, val, val]
        accelerated_lst = accelerate(lst)
        self.assertEqual(accelerated_lst, [val])

    def test_central_differences_have_expected_lengths(self):
        xs = np.random.rand(8)
        axs = accelerate(xs, direction='central')
        self.assertTupleEqual((6,), axs.shape)

    def test_forward_differences_have_expected_lengths(self):
        xs = np.random.rand(8)
        axs = accelerate(xs, direction='forward')
        self.assertTupleEqual((6,), axs.shape)
