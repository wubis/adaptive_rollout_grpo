from dataclasses import dataclass
import re


BOX = re.compile(r"\\boxed\s*\{")
BOX_PLACEHOLDER = re.compile(
    r"(?:\\[hv]?phantom\s*\{(?:[^{}]|\{[^{}]*\})*\}|\\text\s*\{\s*(?:final\s+)?answer\s*\})",
    re.IGNORECASE,
)
NEW_PROBLEM = re.compile(
    r"(?:^|(?<=[.!?])[ \t]+)[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?(?:"
    r"(?:problem|question|exercise|user|human)(?:[ \t]+\d+)?[ \t]*:"
    r"|(?:problem|question|exercise)[ \t]+\d+[.)]"
    r"|(?:next|new|another)[ \t]+(?:problem|question|exercise)\b"
    r"|solve the (?:problem|question) and put the final answer\b)"
    r"|^[ \t]*(?:(?:what|which|how|who|when|where|why)\b|\d+[.)][ \t]+)[^\n]*\?",
    re.IGNORECASE | re.MULTILINE,
)
REQUEST = re.compile(r"\b(?:find|give|compute|calculate|determine|state|what (?:is|are))\s+(?:the\s+)?", re.IGNORECASE)
CLAIM = re.compile(
    r"(?:^[ \t]*|(?<=[.!?])[ \t]+)"
    r"(?P<conclusion>(?:therefore|thus|hence|so|in conclusion|simplifying),?[ \t]+)?"
    r"(?:(?:we find that|I found)[ \t]+)?(?:\*\*)?"
    r"(?:"
    r"(?P<label>(?:(?:the|my|our)[ \t]+)?(?:final[ \t]+)?answer)"
    r"(?:[ \t]*\*\*)?(?:[ \t]*[:=]|[ \t]+(?:is|equals)\b)"
    r"|the[ \t]+(?P<subject>[a-z]+(?:[ \t]+[a-z]+){0,8}?)[ \t]+(?:is|equals|to be)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
SCALAR = rf"(?:[+-]?\\(?:d|t)?frac\s*\{{\s*[+-]?\d+\s*\}}\s*\{{\s*[+-]?\d+\s*\}}|{NUMBER}(?:\s*/\s*[+-]?\d+)?)"
VALUE = rf"(?:{SCALAR}|\(\s*{SCALAR}(?:\s*,\s*{SCALAR})+\s*\))"
ASSIGNMENT = r"(?:[a-zA-Z]|\(\s*[a-zA-Z](?:\s*,\s*[a-zA-Z])+\s*\))\s*=\s*"
EXPRESSION = re.compile(
    rf"(?:\$\$\s*(?:{ASSIGNMENT})?(?P<display>{VALUE})\s*\$\$"
    rf"|\$\s*(?:{ASSIGNMENT})?(?P<inline>{VALUE})\s*\$"
    rf"|\\\(\s*(?:{ASSIGNMENT})?(?P<parens>{VALUE})\s*\\\)"
    rf"|\\\[\s*(?:{ASSIGNMENT})?(?P<brackets>{VALUE})\s*\\\]"
    rf"|(?:{ASSIGNMENT})?(?P<plain>{VALUE}))"
    r"(?=[ \t]*(?:[.!?](?!\d)|\r?\n|\Z))"
)
ANSWER_FORMAT = re.compile(r"^(?:\*\*\s*)?(?:indeed\s+)?(?:(?:at\s+)?(?:the\s+)?ordered pair\s+)?", re.IGNORECASE)


@dataclass(frozen=True)
class Grade:
    reward: int
    status: str
    answer: str | None = None
    source: str | None = None


def _box_contents(text, start):
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip() or None
    return None


def boxed_answer(text):
    boxes = list(BOX.finditer(text))
    return _box_contents(text, boxes[-1].end()) if boxes else None


def parse_answer(answer):
    from math_verify import LatexExtractionConfig, parse

    return parse(
        f"${answer}$",
        extraction_config=[LatexExtractionConfig()],
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


def _grade(answer, source, parsed, gold):
    from math_verify import verify

    if not parsed:
        return Grade(0, "unparseable", answer, source)
    if verify(gold, parsed):
        return Grade(1, "correct", answer, source)
    return Grade(0, "incorrect", answer, source)


def grade_boxed(text, gold):
    """Replay the smoke verifier's last-box rule."""
    answer = boxed_answer(text)
    return _grade(answer, "boxed" if answer else None, parse_answer(answer) if answer else [], gold)


def score_answer(text, gold):
    grade = grade_boxed(text, gold)
    return grade.reward, grade.status


def _explicit_answer(solution, problem, truncated):
    request = REQUEST.search(problem)
    target = " ".join(problem[request.end():].lower().split()) if request else ""
    last_claim = None
    for claim in CLAIM.finditer(solution):
        if claim["label"]:
            last_claim = claim
        else:
            subject = " ".join(claim["subject"].lower().split())
            if subject not in {"value", "number", "quantity", "expression"} and re.match(
                rf"{re.escape(subject)}\b", target
            ):
                last_claim = claim
    if last_claim is None:
        return None

    payload = ANSWER_FORMAT.sub("", solution[last_claim.end():].lstrip(), count=1)
    expression = EXPRESSION.match(payload)
    if expression is None:
        return None
    answer = expression[expression.lastgroup]
    suffix = payload[expression.end():].strip()
    if last_claim["subject"] and not last_claim["conclusion"] and solution[:last_claim.start()].strip():
        if suffix not in {"", ".", "!", "?"}:
            return None
    bare_scalar = expression.lastgroup == "plain" and not answer.endswith((")", "}"))
    if truncated and bare_scalar and not suffix:
        return None
    return answer


def grade_final_answer(text, problem, gold, *, truncated=False):
    """Grade one bounded solution, with boxed answers taking precedence."""
    boundary = NEW_PROBLEM.search(text)
    solution = text[:boundary.start()] if boundary else text
    for box in reversed(list(BOX.finditer(solution))):
        answer = _box_contents(solution, box.end())
        if not answer or BOX_PLACEHOLDER.fullmatch(answer):
            continue
        parsed = parse_answer(answer)
        if parsed:
            return _grade(answer, "boxed", parsed, gold)

    answer = _explicit_answer(solution, problem, truncated=truncated and boundary is None)
    return _grade(answer, "explicit" if answer else None, parse_answer(answer) if answer else [], gold)
