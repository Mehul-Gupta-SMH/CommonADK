"""Feature questions an agent must answer, and deterministic scoring of them.

An agent in commonadk has a persona (prose, in `skill.md`) and tools, and
returns free text. Nothing declares what SHAPE its answer takes, so
evaluating one means reading it -- and reading it means a judge, whether
that judge is a person or another model.

This module is the other option: declare the questions up front, constrain
each answer to a boolean or a named set, and score by `==`. No threshold,
no similarity, no judge. The idea is not new here; it is
`agent_gauntlet.Answer` generalised, whose docstring has said why since it
was written: "Structured, so grading needs no similarity threshold -- and
therefore no judge sneaking back in."

## Why this is its own spec, and its own file

The questions live in an `evaluation.yaml` beside the project rather than
in `agent.yaml`, and loading it is a separate call. An existing project is
unaffected and the file is optional: declaring an answer contract is a
thing you opt into per agent, not a tax on every agent.

## Why these field names and not OpenAI's

OpenAI's Decisions API (`POST /v1/decisions`, announced 2026-09-29) does
the same job at the model: developer-declared questions, answers drawn
from a fixed set, probabilities attached. It is the obvious backend for
this spec and the vocabulary below borrows its two useful kinds.

What this module deliberately does NOT do is mirror its wire format. As of
writing the endpoint is in limited preview, no official reference was
reachable, and three third-party documentation sources disagree about the
field names -- while at least one independent implementation of the same
idea rejects any unknown field with HTTP 400. A spec that mirrors an
unstable schema inherits every revision of it. So this is commonadk's own
stable shape, and the Decisions API is one thing that can answer it.

That is `pricing.AUTHORITY`'s lesson from the sibling project, which
learned it the expensive way: assuming an authority you have not verified
is how the wrong thing gets pinned.

## Why there is no `score` kind

The upstream API has three question kinds -- predicate, choice, and score.
This module has the first two.

A score is a number, and turning a number into a verdict needs a
threshold. A threshold is a judge with a parameter attached: move it and
the same answer passes or fails, which is the one property this module
exists to avoid. Adding `score` later is easy; it is left out so that the
first version cannot quietly reintroduce what it was built to remove.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from pydantic import ValidationError as PydanticValidationError

    from .models import Project

AnswerValue = Union[bool, str]
"""What an answer may be: a boolean, or one of a question's named choices.

Not `int`, not `float`. See the module docstring on `score`.
"""


class DecisionKind(str, Enum):
    """The two question kinds that can be scored by equality."""

    PREDICATE = "predicate"
    """Answered true or false."""

    CHOICE = "choice"
    """Answered with exactly one of a declared set of names."""


class DecisionQuestion(BaseModel):
    """One question, and the complete set of answers it accepts."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: DecisionKind
    instructions: str = ""
    """What the agent is told about this question. Prose, and the only
    prose here: it shapes the answer but never scores it."""

    choices: list[str] = Field(default_factory=list)
    """The allowed answers for a `choice` question, and empty otherwise."""

    @model_validator(mode="after")
    def _check_domain(self) -> "DecisionQuestion":
        if not self.name.strip():
            raise ValueError("a question needs a name")
        if self.kind is DecisionKind.PREDICATE:
            if self.choices:
                raise ValueError(
                    f"question '{self.name}': a predicate is answered true "
                    f"or false, so it must not declare choices"
                )
            return self
        # A choice question's domain has to be a real choice, and has to be
        # unambiguous: two identical options make an answer that matches
        # both, and one option makes a question that cannot discriminate.
        if len(self.choices) < 2:
            raise ValueError(
                f"question '{self.name}': a choice question needs at least "
                f"two choices, got {len(self.choices)}"
            )
        duplicates = sorted({c for c in self.choices
                             if self.choices.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"question '{self.name}': duplicate choices {duplicates}"
            )
        blank = [c for c in self.choices if not c.strip()]
        if blank:
            raise ValueError(f"question '{self.name}': a choice cannot be blank")
        return self

    def accepts(self, value: AnswerValue) -> bool:
        """Is `value` inside this question's declared domain?

        The whole contract in one method: an answer is scoreable only if it
        is one of the things the question said it could be.
        """
        if self.kind is DecisionKind.PREDICATE:
            return isinstance(value, bool)
        # `bool` is a subclass of `int`, not of `str`, so a stray True
        # cannot slip through as a choice name -- but be explicit anyway.
        return isinstance(value, str) and value in self.choices


class DecisionCase(BaseModel):
    """One input, and the answers the persona is supposed to give for it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expect: dict[str, AnswerValue] = Field(default_factory=dict)
    """Question name -> the expected answer.

    A case may expect a subset of the questions. A question it does not
    mention is not scored for that case, which is different from being
    expected to be absent.
    """


class DecisionSpec(BaseModel):
    """The questions one agent must answer, and the cases to score it on."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    persona: str = ""
    """Which persona this contract belongs to.

    Recorded rather than enforced: the persona lives in `skill.md` as prose
    and nothing here can check that the questions match it. Naming it is
    what makes a spec traceable to the thing it was written for.
    """

    questions: list[DecisionQuestion] = Field(default_factory=list)
    cases: list[DecisionCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_spec(self) -> "DecisionSpec":
        if not self.agent.strip():
            raise ValueError("a decision spec must name the agent it is for")
        if not self.questions:
            raise ValueError(
                f"agent '{self.agent}': a decision spec with no questions "
                f"scores nothing"
            )
        names = [q.name for q in self.questions]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate question name(s): {duplicates}")

        by_name = {q.name: q for q in self.questions}
        case_ids = [c.id for c in self.cases]
        dup_cases = sorted({c for c in case_ids if case_ids.count(c) > 1})
        if dup_cases:
            raise ValueError(f"duplicate case id(s): {dup_cases}")

        for case in self.cases:
            for question_name, expected in case.expect.items():
                question = by_name.get(question_name)
                if question is None:
                    raise ValueError(
                        f"case '{case.id}' expects an answer to "
                        f"'{question_name}', which is not a declared question"
                    )
                # Caught here rather than at scoring time on purpose: an
                # expectation outside the question's own domain can never
                # be met, so every run against it would score a failure
                # that is the spec's fault. A bar nothing can clear is not
                # a measurement.
                if not question.accepts(expected):
                    allowed = ("true/false" if question.kind
                               is DecisionKind.PREDICATE
                               else ", ".join(question.choices))
                    raise ValueError(
                        f"case '{case.id}' expects {expected!r} for "
                        f"'{question_name}', which is not in its domain "
                        f"({allowed})"
                    )
        return self

    def question(self, name: str) -> Optional[DecisionQuestion]:
        return next((q for q in self.questions if q.name == name), None)


class DecisionAnswer(BaseModel):
    """One answer an agent actually gave."""

    model_config = ConfigDict(extra="forbid")

    question: str
    value: Optional[AnswerValue] = None
    """`None` means the agent did not answer -- it abstained, or nothing
    parseable came back. Distinct from a wrong answer, and scored as
    such."""

    probability: Optional[float] = None
    """The answer's own stated probability, if the backend reported one."""

    probability_calibrated: bool = False
    """Whether `probability` is a CALIBRATED probability of correctness.

    Defaults to False, and that default is load-bearing. The Decisions API
    returns per-answer probabilities, and whether they are calibrated is
    contested: one source says production use requires calibrating
    thresholds yourself against labelled data, and an independent
    implementation of the same idea states outright that its values are not
    calibrated probabilities that the decision is correct.

    A Brier score computed over uncalibrated numbers is a figure that looks
    like calibration and is not. So the flag travels with the number, and a
    caller that wants to score calibration has to find it set -- see
    `calibratable`.
    """

    @property
    def answered(self) -> bool:
        return self.value is not None

    @property
    def calibratable(self) -> bool:
        """May this answer's probability be scored as a calibration?

        Only when a probability is present AND something declared it
        calibrated. Everything else is a number worth recording and not
        worth scoring.
        """
        return self.probability is not None and self.probability_calibrated


class DecisionOutcome(BaseModel):
    """One (case, question) cell's verdict."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    expected: AnswerValue
    given: Optional[AnswerValue] = None
    out_of_domain: bool = False
    """The agent answered, but with something the question never offered.

    Its own state rather than a wrong answer, because the remedy differs:
    a wrong answer is the agent's, an unparseable one is usually the output
    contract's.
    """

    @property
    def decidable(self) -> bool:
        """Was this scoreable at all?

        False when the agent did not answer. Those cells leave the rate's
        denominator rather than counting as misses -- an unanswered
        question is not a wrong one, and scoring it as wrong would make an
        abstaining agent indistinguishable from a confidently incorrect
        one.
        """
        return self.given is not None

    @property
    def correct(self) -> Optional[bool]:
        """`None` when undecidable, never False."""
        if not self.decidable:
            return None
        if self.out_of_domain:
            return False
        return self.given == self.expected


def score_case(
    spec: DecisionSpec,
    case: DecisionCase,
    answers: Sequence[DecisionAnswer],
) -> list[DecisionOutcome]:
    """Score one case's answers against what it expected.

    Only the questions the case expects are scored. An answer to a
    question the case said nothing about is not evidence for or against
    anything, and is dropped rather than counted.
    """
    given = {a.question: a for a in answers}
    outcomes: list[DecisionOutcome] = []
    for question_name, expected in case.expect.items():
        question = spec.question(question_name)
        answer = given.get(question_name)
        value = answer.value if answer is not None else None
        outcomes.append(DecisionOutcome(
            case_id=case.id,
            question=question_name,
            expected=expected,
            given=value,
            out_of_domain=bool(
                value is not None
                and question is not None
                and not question.accepts(value)
            ),
        ))
    return sorted(outcomes, key=lambda o: (o.case_id, o.question))


def accuracy_by_question(
    outcomes: Sequence[DecisionOutcome],
) -> dict[str, Optional[float]]:
    """Per-question accuracy over the decidable cells.

    Per question rather than one blended number, for the reason the sibling
    project refuses a composite scalar: an agent that is perfect on three
    questions and useless on a fourth is not 75% of an agent, and one
    column cannot say which it is.

    `None` for a question with no decidable cell -- not 0.0, which would
    read as "always wrong" about a question nobody managed to answer.
    """
    rates: dict[str, Optional[float]] = {}
    by_question: dict[str, list[DecisionOutcome]] = {}
    for outcome in outcomes:
        by_question.setdefault(outcome.question, []).append(outcome)
    for question, cells in by_question.items():
        decidable = [c for c in cells if c.decidable]
        rates[question] = (
            sum(1 for c in decidable if c.correct) / len(decidable)
            if decidable else None
        )
    return rates


def unanswered(outcomes: Sequence[DecisionOutcome]) -> list[DecisionOutcome]:
    """The cells that left the denominator, so a reader can see how many.

    A rate computed over two of twenty cells is arithmetically fine and
    worth almost nothing, and the only way to know is to be told how much
    was censored.
    """
    return [o for o in outcomes if not o.decidable]


# -- loading ---------------------------------------------------------------
#
# A separate call from `commonadk.load`, and a separate file from
# `agent.yaml`. An answer contract is something an agent opts into: a
# project with no `evaluation.yaml` is not an incomplete project, and
# loading one must not become a condition of loading the other.


def load_decisions(
    path: Union[str, "Path"],
    *,
    project: Optional["Project"] = None,
) -> list[DecisionSpec]:
    """Load `evaluation.yaml` -- one spec per agent.

    The file holds a list of specs, or a single mapping when there is only
    one agent to score. Both shapes are accepted; anything else is an
    error rather than a guess.

    Pass `project` to have the agent names checked against it. Worth doing:
    a spec naming an agent that does not exist scores nothing and reports
    nothing, so a typo in `agent:` is silent until somebody wonders why a
    column is empty. Without a project the names cannot be checked, and
    this says so by simply not checking rather than by assuming they are
    fine.

    Raises `commonadk.validation.ValidationError` with every problem found,
    the same contract `commonadk.load` has.
    """
    from pathlib import Path as _Path

    import yaml
    from pydantic import ValidationError as PydanticValidationError

    from .validation import ValidationError

    file = _Path(path)
    errors: list[str] = []

    if not file.is_file():
        raise ValidationError([f"evaluation spec not found: {file}"])

    try:
        data = yaml.safe_load(file.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValidationError([f"{file.name}: invalid YAML: {exc}"]) from None

    if isinstance(data, dict):
        raw_specs = [data]
    elif isinstance(data, list):
        raw_specs = data
    else:
        raise ValidationError([
            f"{file.name}: expected a spec mapping or a list of them, got "
            f"{type(data).__name__}"
        ])

    specs: list[DecisionSpec] = []
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            errors.append(
                f"{file.name}: entry {index} is a {type(raw).__name__}, "
                f"expected a mapping"
            )
            continue
        try:
            specs.append(DecisionSpec.model_validate(raw))
        except PydanticValidationError as exc:
            # Named by agent where possible: "entry 2" is useless in a file
            # with six of them.
            label = raw.get("agent") or f"entry {index}"
            errors.append(f"{file.name}: {label}: {_first_error(exc)}")

    agents = [s.agent for s in specs]
    duplicates = sorted({a for a in agents if agents.count(a) > 1})
    if duplicates:
        # Two contracts for one agent is two different bars, and nothing
        # downstream could say which one a result was scored against.
        errors.append(
            f"{file.name}: more than one spec for agent(s) {duplicates}"
        )

    if project is not None:
        known = set(project.agents)
        for spec in specs:
            if spec.agent not in known:
                errors.append(
                    f"{file.name}: spec names agent '{spec.agent}', which the "
                    f"project does not define (has: {', '.join(sorted(known))})"
                )

    if errors:
        raise ValidationError(errors)
    return specs


def _first_error(exc: "PydanticValidationError") -> str:
    """The first pydantic error, flattened to one line.

    Mirrors `loader._format_pydantic_error`'s job without importing it:
    these messages are read by a person fixing a YAML file, and pydantic's
    own repr buries the useful sentence.
    """
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        message = err.get("msg", "invalid")
        return f"{location}: {message}"
    return "invalid"
