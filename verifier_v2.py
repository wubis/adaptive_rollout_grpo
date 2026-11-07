"""Select the latest explicit answer to the original question, before grading."""

import re

from verifier import (
    BOX, BOX_PLACEHOLDER, NEW_PROBLEM, REQUEST, SCALAR, VALUE,
    Grade, _box_contents, _explicit_answer, _explicit_polynomial, _grade, parse_answer,
)


VERSION = "final-answer-v2"
TASK = re.compile(
    r"\n[ \t]*\n[ \t]*(?:\d+[.)][ \t]+)?"
    r"(?:Solve|Find|Evaluate|Factor|Expand|Simplify|Compute|Calculate|Determine)\b[^\n]*",
)
TARGET = re.compile(
    r"(?:\bvalue of|\bsolve for|\bfind|\bdetermine|\bwhat is)\s+(?:the\s+)?"
    r"(?:\$(?P<dollar>[a-zA-Z])\$|\\\((?P<parens>[a-zA-Z])\\\))",
    re.IGNORECASE,
)
MATH = re.compile(
    r"\$\$(?P<display>[^$]+)\$\$|\$(?P<inline>[^$\n]+)\$"
    r"|\\\((?P<parens>[^\n]+?)\\\)|\\\[(?P<brackets>[^\n]+?)\\\]",
)
NUMERIC_CHAIN = re.compile(rf"(?:{SCALAR}\s*[+*/-]\s*)*{SCALAR}(?:\s*=\s*{SCALAR})+")
VERIFICATION = re.compile(r"^(?:[.!]\s*)?(?:Let's|Let us)\s+(?:verify|check|confirm|implement)\b", re.I)
CONCLUSION = re.compile(r"\b(?:therefore|thus|hence|so|finally|we find|solving.*?gives)\b", re.I)


def requested_variable(problem):
    matches = list(TARGET.finditer(problem))
    return (matches[-1]["dollar"] or matches[-1]["parens"]) if matches else None


def complete_value(payload):
    """Recognize a complete scalar or tuple, including a numeric equality chain."""
    payload = payload.strip()
    if re.fullmatch(VALUE, payload):
        return payload
    if NUMERIC_CHAIN.fullmatch(payload):
        return payload.rsplit("=", 1)[1].strip()
    return None


def explicit_candidates(solution, problem, truncated):
    candidates = []
    variable = requested_variable(problem)
    request = REQUEST.search(problem)
    target = problem[request.end():] if request else ""
    noun = re.match(r"(sum|product|difference|ratio|distance|area|perimeter|center)\b", target, re.I)

    answer = _explicit_answer(solution, problem, truncated)
    tail = solution[solution.rfind(answer) + len(answer):] if answer else ""
    if answer and not re.search(r"\b(?:still|remaining|not final|need to)\b", tail, re.I):
        candidates.append((solution.rfind(answer), answer, "explicit"))
    polynomial = _explicit_polynomial(solution, problem, truncated)
    if polynomial:
        candidates.append((solution.rfind(polynomial), polynomial, "polynomial"))

    for match in MATH.finditer(solution):
        payload = match[match.lastgroup].strip()
        suffix = solution[match.end():].strip()
        prefix = solution[max(0, solution.rfind("\n", 0, match.start())):match.start()]
        terminal = suffix in ("", ".", "!")
        verification = bool(VERIFICATION.match(suffix))
        if variable:
            assignment = re.fullmatch(rf"{re.escape(variable)}\s*=\s*(.+)", payload)
            answer = complete_value(assignment[1]) if assignment else None
            if answer and not re.search(r"\bor\s*$", prefix) and (terminal or CONCLUSION.search(prefix)):
                candidates.append((match.start(), answer, "assignment"))
        if noun and (terminal or verification):
            subject = re.search(rf"\bthe\s+{noun[1]}\b[^.\n]*\b(?:is|equals)\s*$", prefix, re.I)
            answer = complete_value(payload)
            if subject and answer:
                candidates.append((match.start(), answer, "quantity"))

    # Plain assignments must end a sentence or the original solution.
    if variable:
        plain = re.compile(rf"(?<![\w\\$(]){re.escape(variable)}\s*=\s*(?P<value>{VALUE})(?=[.!](?!\d)|\s*\Z)")
        for match in plain.finditer(solution):
            # Delimited assignments were checked above, including their closure.
            if any(math.start() <= match.start() < math.end() for math in MATH.finditer(solution)):
                continue
            prefix = solution[max(0, solution.rfind("\n", 0, match.start())):match.start()]
            suffix = solution[match.end():].strip()
            if not re.search(r"\bor\s*$", prefix) and (
                (suffix in ("", ".", "!") and not (truncated and not suffix)) or CONCLUSION.search(prefix)
            ):
                candidates.append((match.start(), match["value"], "assignment"))
    return candidates


def extract_answer(text, problem, *, truncated=False):
    boundary = NEW_PROBLEM.search(text)
    solution = text[:boundary.start()] if boundary else text
    cut_off = truncated and boundary is None
    # A command after a completed answer can start a new, unmarked exercise.
    for task in TASK.finditer(solution):
        command = task[0].strip()
        numbered_outline = re.match(r"\d+[.)]", command) and not re.search(r"\$|\\\(|=|\banother\b", command)
        if command.endswith(":") or numbered_outline:
            continue
        prefix = solution[:task.start()]
        boxes = [_box_contents(prefix, box.end()) for box in BOX.finditer(prefix)]
        has_box = any(answer and not BOX_PLACEHOLDER.fullmatch(answer) and parse_answer(answer) for answer in boxes)
        if has_box or explicit_candidates(prefix, problem, False):
            solution = prefix
            cut_off = False
            break

    candidates = explicit_candidates(solution, problem, cut_off)
    for box in BOX.finditer(solution):
        answer = _box_contents(solution, box.end())
        if answer and not BOX_PLACEHOLDER.fullmatch(answer) and parse_answer(answer):
            candidates.append((box.start(), answer, "boxed"))
    if not candidates:
        return None, None
    _, answer, source = max(candidates, key=lambda item: item[0])
    return answer, source


def grade_answer(text, problem, gold, *, truncated=False):
    answer, source = extract_answer(text, problem, truncated=truncated)
    return _grade(answer, source, parse_answer(answer), gold) if answer else Grade(0, "unparseable")
