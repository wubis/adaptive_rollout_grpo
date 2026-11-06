from importlib.util import find_spec
import unittest

from verifier import _explicit_answer, grade_boxed, grade_final_answer, parse_answer


class ExplicitAnswerTests(unittest.TestCase):
    def test_requested_quantity_in_prose(self):
        self.assertEqual(
            _explicit_answer("Thus, the intersection is at the ordered pair $(x,y)=(3,-1)$.",
                             "Find the intersection of the lines.", truncated=False),
            "(3,-1)",
        )

    def test_intermediate_value_for_another_quantity(self):
        self.assertIsNone(_explicit_answer("Therefore, the discriminant is 4.", "Find the roots.", truncated=False))

    def test_intermediate_value_for_requested_quantity(self):
        text = "Start with the first term. The sum is 3. We still need to add the remaining terms."
        self.assertIsNone(_explicit_answer(text, "Find the sum of the terms.", truncated=False))

    def test_closed_tuple_at_cap_is_complete(self):
        self.assertEqual(_explicit_answer("Final answer: (3,-1)", "Find the intersection.", True), "(3,-1)")

    def test_unfinished_or_ambiguous_answer(self):
        for text, truncated in (
            ("Final answer: (3, -", False),
            ("Final answer: 3 +", False),
            ("Final answer: 3 or 4.", False),
            ("Final answer: $17", True),
            ("Final answer: 1", True),
            ("The roots are", True),
        ):
            with self.subTest(text=text):
                self.assertIsNone(_explicit_answer(text, "Find the roots.", truncated))


@unittest.skipUnless(find_spec("math_verify"), "Install the pinned Math-Verify for scoring tests")
class VerifierTests(unittest.TestCase):
    def test_regression_correct_center_in_prose(self):
        problem = "Give the center of the circle as an ordered pair."
        gold = parse_answer("(3,-1)")
        for text, truncated in (
            ("Therefore, the center of the circle is (3, -1).", False),
            ("I found the center of the circle to be (3,-1).\nThe derivation starts with", True),
            ("The center of the circle is (3, -1).\n\n1) A group of companies earned what total?\n"
             r"The answer is \boxed{75}.", True),
        ):
            with self.subTest(text=text):
                self.assertEqual(grade_boxed(text, gold).reward, 0)
                grade = grade_final_answer(text, problem, gold, truncated=truncated)
                self.assertEqual((grade.source, grade.reward), ("explicit", 1))

    def test_wrong_answer_is_extracted_without_using_gold(self):
        text = "Final answer: $12$."
        for gold_value, reward in (("17", 0), ("12", 1)):
            grade = grade_final_answer(text, "Compute the sum.", parse_answer(gold_value))
            self.assertEqual((grade.answer, grade.reward), ("12", reward))
            self.assertEqual(grade.status, "correct" if reward else "incorrect")

    def test_valid_box_takes_precedence_including_when_wrong(self):
        for trailing in (r"\boxed{}", r"\boxed{", "Question 2: What is 1+2?\n" + r"\boxed{3}"):
            with self.subTest(trailing=trailing):
                text = r"\boxed{2} Final answer: 3." + "\n" + trailing
                grade = grade_final_answer(text, "Compute the sum.", parse_answer("3"))
                self.assertEqual((grade.source, grade.answer, grade.reward), ("boxed", "2", 0))

    def test_ignores_placeholder_boxes_and_keeps_math_verify_equivalence(self):
        for box in (r"\boxed{}", r"\boxed{\phantom{100}}", r"\boxed{\text{Answer}}"):
            with self.subTest(box=box):
                text = box + "\n" + r"Final answer: $\frac{1}{2}$."
                grade = grade_final_answer(text, "Compute the ratio.", parse_answer("0.5"))
                self.assertEqual((grade.source, grade.answer, grade.reward), ("explicit", r"\frac{1}{2}", 1))

    def test_later_problem_cannot_supply_an_answer(self):
        for heading in ("Problem:", "**Question 2:**", "New problem:", "2) A group bought how many tickets?"):
            with self.subTest(heading=heading):
                for answer in (r"\boxed{17}", "Final answer: 17."):
                    grade = grade_final_answer(
                        f"We need to compute the sum.\n\n{heading} Compute another sum.\n{answer}",
                        "Compute the sum.", parse_answer("17"),
                    )
                    self.assertEqual((grade.status, grade.reward), ("unparseable", 0))

    def test_later_problem_cannot_override_original_box(self):
        grade = grade_final_answer(
            r"\boxed{17}" + "\n\nProblem: Compute 2+2.\n" + r"\boxed{4}",
            "Compute the sum.", parse_answer("17"),
        )
        self.assertEqual((grade.source, grade.answer, grade.reward), ("boxed", "17", 1))

    def test_inline_problem_cannot_override_original_answer(self):
        grade = grade_final_answer(
            r"Final answer: 17. Problem: Compute 2+2. \boxed{4}",
            "Compute the sum.", parse_answer("17"),
        )
        self.assertEqual((grade.source, grade.answer, grade.reward), ("explicit", "17", 1))

    def test_polynomial_fallback_accepts_explicit_conclusions(self):
        for problem, text, gold_value in (
            ("What is $p(t)$?", "Therefore, the polynomial $p(t)$ is $2t^3-t+7$.", "7-t+2t^3"),
            ("Expand the product.", r"Final answer: \(2t^{3}-t+7\).", "2t^3-t+7"),
            ("Expand the product.", "The answer is $-4z+3$.", "3-4z"),
        ):
            with self.subTest(text=text):
                gold = parse_answer(gold_value)
                self.assertEqual(grade_final_answer(text, problem, gold).status, "unparseable")
                grade = grade_final_answer(text, problem, gold, allow_polynomial=True)
                self.assertEqual((grade.source, grade.status), ("polynomial", "correct"))

    def test_polynomial_fallback_rejects_ambiguous_or_incomplete_claims(self):
        gold = parse_answer("2t^3-t+7")
        for text, truncated in (
            ("Therefore, the polynomial $q(t)$ is $2t^3-t+7$.", False),
            ("Therefore, the polynomial $p(t)$ is $2x^3-x+7$.", False),
            ("We derived $2t^3-t+7$.", False),
            ("Final answer: $2t^3-t+7$. That was only the first term.", False),
            ("Final answer: $2t^3-t+7$ or $2t^3-t-7$.", False),
            ("Final answer: $2t^3-t+7=7-t+2t^3$.", False),
            ("Final answer: $2t^3-t+s+7$.", False),
            ("Final answer: $2t^3-t^{-1}+7$.", False),
            ("Final answer: $2t^3-t+7", True),
            ("Final answer: $2t^3-t+7$", True),
            ("Question 2: Find another polynomial.\nFinal answer: $2t^3-t+7$.", False),
        ):
            with self.subTest(text=text):
                grade = grade_final_answer(
                    text, "What is $p(t)$?", gold, truncated=truncated, allow_polynomial=True,
                )
                self.assertEqual((grade.status, grade.reward), ("unparseable", 0))

    def test_polynomial_equivalence_and_precedence(self):
        problem = "What is $p(t)$?"
        gold = parse_answer("2t^3-t+7")
        final = "Final answer: $2t^3-t+7$."
        for text, source, status in (
            (final + "\nQuestion 2: Find the next answer.\n" + r"\boxed{3}", "polynomial", "correct"),
            (r"\boxed{3}" + "\n" + final, "boxed", "incorrect"),
            ("Final answer: $2t^3-t-7$.", "polynomial", "incorrect"),
        ):
            with self.subTest(text=text):
                grade = grade_final_answer(text, problem, gold, truncated=True, allow_polynomial=True)
                self.assertEqual((grade.source, grade.status), (source, status))
