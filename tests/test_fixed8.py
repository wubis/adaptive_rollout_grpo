import unittest

from fixed8 import group_advantages


class GroupAdvantagesTests(unittest.TestCase):
    def test_mixed_group_uses_population_standard_deviation(self):
        advantages = group_advantages([1, 1, 0, 0, 0, 0, 0, 0])
        expected = [1.732047] * 2 + [-0.577349] * 6

        self.assertEqual(len(advantages), 8)
        for index, (actual, target) in enumerate(zip(advantages, expected)):
            with self.subTest(index=index):
                self.assertAlmostEqual(actual, target, places=6)

    def test_uniform_groups_have_zero_advantages(self):
        for reward in (0, 1):
            with self.subTest(reward=reward):
                self.assertEqual(group_advantages([reward] * 8), [0.0] * 8)

    def test_rejects_wrong_group_size(self):
        with self.assertRaises(ValueError):
            group_advantages([0] * 7)

    def test_rejects_nonbinary_reward(self):
        with self.assertRaises(ValueError):
            group_advantages([0] * 7 + [2])
