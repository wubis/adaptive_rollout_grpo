from importlib.util import find_spec
import unittest

from verifier_v2 import extract_answer, grade_answer
from verifier import grade_final_answer, parse_answer


@unittest.skipUnless(find_spec("math_verify"), "Install pinned Math-Verify")
class ExtractionV2Tests(unittest.TestCase):
    def test_requested_variable_and_later_correction(self):
        problem = "Given a root, what is the value of $b$?"
        for text in (
            "Solving gives $b = 3$.",
            "Solving gives b = 3.",
            r"\boxed{-3}" + "\nThus, $b=3$.",
            "The calculation starts with 9. Correcting the sign, we find $b=3$.",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text, problem)[0], "3")

    def test_boundary_and_conflicting_conclusions(self):
        problem = "What is the value of $b$?"
        self.assertEqual(extract_answer(r"\boxed{3}" + "\n\nSolve the equation and write out the steps.\n"
                                        + r"\boxed{8}", problem)[0], "3")
        self.assertEqual(extract_answer(r"\boxed{3}" + "\n\n2. Evaluate another expression.\n"
                                        + r"\boxed{8}", problem)[0], "3")
        self.assertEqual(extract_answer(r"\boxed{3}" + "\nFinal answer: $-3$.", problem)[0], "-3")
        text = r"\boxed{15}" + "\n\n1. Simplify the expression inside parentheses:\nWork.\n" + r"\boxed{17}"
        self.assertEqual(extract_answer(text, "Compute the ceiling.")[0], "17")
        text = r"\boxed{15}" + "\n\n1. Simplify the expression inside parentheses.\n2. Apply the ceiling.\n" + r"\boxed{17}"
        self.assertEqual(extract_answer(text, "Compute the ceiling.")[0], "17")
        text = r"\boxed{?}" + "\n\n1. Compute the expression.\n" + r"\boxed{17}"
        self.assertEqual(extract_answer(text, "Compute the ceiling.")[0], "17")

    def test_negative_cases(self):
        for problem, text, cut in (
            ("Find the sum.", "Thus, $b=3$.", False),
            ("What is the value of $b$?", "$a=3$.", False),
            ("What is the value of $b$?", "$b=3$. We still need to add the other term.", False),
            ("What is the value of $b$?", "Final answer: $b=3", True),
            ("What is the value of $b$?", "b=3", True),
            ("What is the value of $b$?", "$b=3.", True),
            ("What is the value of $b$?", r"\(b=3.", True),
            ("What is the value of $b$?", "b=3 or b=4.", False),
            ("What is the value of $b$?", "$b=3$ or $b=4$.", False),
            ("Find the sum.", "The sum is $3$. We still need to add the remaining terms.", False),
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text, problem, truncated=cut)[0])

    def test_requested_quantity_overrides_early_box(self):
        text = r"\boxed{8}" + "\nSince B is (6,4), the sum of the coordinates is \\(6 + 4 = 10\\).\n\nLet's verify this solution."
        self.assertEqual(extract_answer(text, "What is the sum of the coordinates?")[0], "10")


@unittest.skipUnless(find_spec("math_verify"), "Install pinned Math-Verify")
class GradingV2Tests(unittest.TestCase):
    def test_extraction_does_not_see_gold_and_historical_rule_remains(self):
        text = r"\boxed{2}" + "\nFinal answer: $3$."
        for gold, reward in (("3", 1), ("2", 0)):
            grade = grade_answer(text, "Compute the sum.", parse_answer(gold))
            self.assertEqual((grade.answer, grade.reward), ("3", reward))
        self.assertEqual(grade_final_answer(text, "Compute the sum.", parse_answer("2")).reward, 1)
