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

FUNCTION = r"\$[a-zA-Z]\((?P<variable>[a-zA-Z])\)\$"
POLYNOMIAL_CLAIM = re.compile(
    rf"^[ \t]*(?i:therefore|thus|hence|so),?[ \t]+(?i:the polynomial)[ \t]+"
    rf"(?P<function>{FUNCTION})[ \t]+(?i:is|equals)[ \t]+",
    re.MULTILINE,
)
POLYNOMIAL_PAYLOAD = re.compile(
    r"(?:\$(?P<inline>[^$\n]+)\$|\\\((?P<parens>[^\\\n]+)\\\))\s*(?P<period>\.)?\s*\Z"
)
MONOMIAL = r"(?:[0-9]*[ \t]*[a-zA-Z](?:[ \t]*\^[ \t]*(?:[1-9][0-9]?|\{[1-9][0-9]?\}))?|[0-9]+)"
POLYNOMIAL = re.compile(rf"[+-]?[ \t]*{MONOMIAL}(?:[ \t]*[+-][ \t]*{MONOMIAL})*")


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


def _explicit_polynomial(solution, problem, truncated):
    request = REQUEST.search(problem)
    target = re.match(FUNCTION, problem[request.end():]) if request else None
    if target is None and not re.search(r"\b(?:polynomial|expand)\b", problem, re.IGNORECASE):
        return None

    claims = [claim for claim in CLAIM.finditer(solution) if claim["label"]]
    if target:
        claims.extend(claim for claim in POLYNOMIAL_CLAIM.finditer(solution)
                      if claim["function"] == target[0])
    if not claims:
        return None
    claim = max(claims, key=lambda match: match.start())
    payload = POLYNOMIAL_PAYLOAD.fullmatch(solution[claim.end():].lstrip())
    if payload is None or (truncated and not payload["period"]):
        return None
    answer = (payload["inline"] or payload["parens"]).strip()
    if not POLYNOMIAL.fullmatch(answer):
        return None
    variables = set(re.findall(r"[a-zA-Z]", answer))
    if len(variables) != 1 or (target and variables != {target["variable"]}):
        return None
    return answer


def grade_final_answer(text, problem, gold, *, truncated=False, allow_polynomial=False):
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

    cut_off = truncated and boundary is None
    answer = _explicit_answer(solution, problem, truncated=cut_off)
    if answer:
        return _grade(answer, "explicit", parse_answer(answer), gold)
    if allow_polynomial:
        answer = _explicit_polynomial(solution, problem, truncated=cut_off)
        if answer:
            return _grade(answer, "polynomial", parse_answer(answer), gold)
    return Grade(0, "unparseable")
