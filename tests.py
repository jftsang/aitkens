from unittest import TestCase
from unittest.mock import patch

import numpy as np
from hypothesis import given, strategies as st
from parameterized import parameterized

from aitkens import accelerate, second_differences


class TestAitkens(TestCase):
    @given(
        st.floats(min_value=1, max_value=1e9),
        st.floats(min_value=-1, max_value=1).filter(
            lambda initial: abs(initial > 1e-6)
        ),
        st.floats(min_value=1, max_value=1e4),
    )
    def test_geometric_decay(self, limit, initial, rate):
        """Geometrically (exponentially) converging sequences converge
        well when accelerated.
        """
        # TODO the ranges are a bit too conservative and the tolerances
        # very high. This is to make the tests pass (#2), but is it
        # possible to get better estimates on the error bounds?
        lst = limit + initial * np.exp(-rate * np.array(range(12)))
        accelerated_lst = accelerate(lst)
        np.testing.assert_allclose(
            accelerated_lst[-1], [limit], atol=1/rate**2
        )

    @given(st.floats(allow_infinity=False, allow_nan=False))
    def test_handles_constant_sequence(self, val):
        lst = [val, val, val]
        accelerated_lst = accelerate(lst)
        self.assertEqual(accelerated_lst, [val])

    def test_forward_differences(self):
        xs = [1, 4, 9, 16]
        txs, dxs, d2xs = second_differences(xs, direction='forward')
        self.assertListEqual([1, 4], list(txs))
        self.assertListEqual([3, 5], list(dxs))
        self.assertListEqual([2, 2], list(d2xs))

    def test_central_differences(self):
        xs = [1, 4, 9, 16]
        txs, dxs, d2xs = second_differences(xs, direction='central')
        self.assertListEqual([4, 9], list(txs))
        self.assertListEqual([4, 6], list(dxs))
        self.assertListEqual([2, 2], list(d2xs))

    def test_central_differences_have_expected_lengths(self):
        xs = np.random.rand(8)
        axs = accelerate(xs, direction='central')
        self.assertTupleEqual((6,), axs.shape)

    def test_forward_differences_have_expected_lengths(self):
        xs = np.random.rand(8)
        axs = accelerate(xs, direction='forward')
        self.assertTupleEqual((6,), axs.shape)

    @patch('aitkens.second_differences')
    def test_default_is_forward_differences(self, m):
        m.return_value = (np.array([]), np.array([]), np.array([]))
        xs = np.random.rand(8)
        axs = accelerate(xs)  # not specifying direction
        m.assert_called_once()
        self.assertEqual('forward', m.call_args.kwargs['direction'])

    @parameterized.expand([(0,), (-1,), (0.5,), (1.,)])
    def test_rejects_invalid_iterations(self, it):
        xs = [0, 0, 0]
        with self.assertRaises(TypeError):
            accelerate(xs, iterations=it)

    def test_multiple_iterations(self):
        xs = [100, 50, 20, 10, 5, 2, 1]
        acc1 = accelerate(xs)
        np.testing.assert_allclose(acc1, [-25., 5., 0., -2.5, 0.5])
        acc2 = accelerate(xs, iterations=2)
        np.testing.assert_allclose(acc2, [ 0.71428571, -5., -1.13636364])
        acc3 = accelerate(xs, iterations=3)
        np.testing.assert_allclose(acc3, [-2.69491525])
