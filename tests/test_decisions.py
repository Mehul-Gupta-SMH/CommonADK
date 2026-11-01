"""Declared answer contracts, scored by equality (no judge).

An agent returns free text, so evaluating it means reading it -- and
reading it means a judge, whether that judge is a person or another model.
This module's subject is the other option: declare the questions, constrain
each answer to a boolean or a named set, and score by `==`.

Every test here is about one of the three things that make such a score
trustworthy rather than merely exact:

  1. a bar nothing can clear is not a measurement (domain checks, at load
     time, so a mistyped expectation fails the spec rather than the agent),
  2. an unanswered question is not a wrong one (censoring, so an
     abstaining agent is distinguishable from a confidently incorrect one),
  3. a probability is not a calibration (the flag, so a Brier score cannot
     be computed over numbers nobody validated).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from commonadk.decisions import (
    DecisionAnswer,
    DecisionCase,
    DecisionKind,
    DecisionQuestion,
    DecisionSpec,
    accuracy_by_question,
    load_decisions,
    score_case,
    unanswered,
)
from commonadk.validation import ValidationError


def _spec(**overrides):
    base = dict(
        agent="triage",
        persona="support_triage",
        questions=[
            DecisionQuestion(name="is_urgent", kind=DecisionKind.PREDICATE),
            DecisionQuestion(
                name="category", kind=DecisionKind.CHOICE,
                choices=["billing", "bug", "feature"],
            ),
        ],
        cases=[DecisionCase(id="c1", input="x",
                            expect={"is_urgent": True, "category": "billing"})],
    )
    base.update(overrides)
    return DecisionSpec(**base)


# --- the domain is the contract -------------------------------------------


def test_a_predicate_is_answered_true_or_false_and_declares_no_choices():
    q = DecisionQuestion(name="is_urgent", kind=DecisionKind.PREDICATE)
    assert q.accepts(True) and q.accepts(False)
    assert not q.accepts("yes"), "a string is not a boolean"
    assert not q.accepts("true")

    with pytest.raises(ValueError, match="must not declare choices"):
        DecisionQuestion(name="is_urgent", kind=DecisionKind.PREDICATE,
                         choices=["yes", "no"])


def test_a_choice_question_needs_a_real_unambiguous_choice():
    with pytest.raises(ValueError, match="at least two choices"):
        DecisionQuestion(name="c", kind=DecisionKind.CHOICE, choices=["only"])
    with pytest.raises(ValueError, match="duplicate choices"):
        DecisionQuestion(name="c", kind=DecisionKind.CHOICE,
                         choices=["a", "b", "a"])
    with pytest.raises(ValueError, match="cannot be blank"):
        DecisionQuestion(name="c", kind=DecisionKind.CHOICE, choices=["a", " "])


def test_a_boolean_cannot_pass_as_a_choice_name():
    """`bool` is a subclass of `int`, not `str`, and this is the one place
    that distinction decides whether a wrong answer scores as right."""
    q = DecisionQuestion(name="category", kind=DecisionKind.CHOICE,
                         choices=["billing", "bug"])
    assert q.accepts("billing")
    assert not q.accepts(True)
    assert not q.accepts("nonsense")


def test_an_expectation_outside_the_domain_fails_the_SPEC_not_the_agent():
    """A bar nothing can clear is not a measurement.

    Caught at load time: every run against such a case would score a
    failure that is the spec's fault, and nothing downstream could tell
    that from the agent being wrong.
    """
    with pytest.raises(ValueError, match="not in its domain"):
        _spec(cases=[DecisionCase(id="c1", input="x",
                                  expect={"category": "refund"})])
    with pytest.raises(ValueError, match="not in its domain"):
        _spec(cases=[DecisionCase(id="c1", input="x",
                                  expect={"is_urgent": "yes"})])


def test_expecting_an_undeclared_question_is_refused():
    with pytest.raises(ValueError, match="not a declared question"):
        _spec(cases=[DecisionCase(id="c1", input="x",
                                  expect={"sentiment": True})])


def test_duplicate_names_are_refused_on_both_axes():
    with pytest.raises(ValueError, match="duplicate question name"):
        _spec(questions=[
            DecisionQuestion(name="dup", kind=DecisionKind.PREDICATE),
            DecisionQuestion(name="dup", kind=DecisionKind.PREDICATE),
        ], cases=[])
    with pytest.raises(ValueError, match="duplicate case id"):
        _spec(cases=[DecisionCase(id="same", input="a", expect={}),
                     DecisionCase(id="same", input="b", expect={})])


def test_a_spec_with_no_questions_scores_nothing_and_says_so():
    with pytest.raises(ValueError, match="scores nothing"):
        _spec(questions=[], cases=[])


def test_the_spec_forbids_unknown_keys():
    """commonadk's own YAML is strict -- a typo must not be ignored."""
    with pytest.raises(Exception):
        DecisionSpec.model_validate({
            "agent": "triage",
            "questions": [{"name": "q", "kind": "predicate"}],
            "quesitons": [],          # typo
        })


def test_there_is_no_score_kind():
    """Deliberate. A score needs a threshold to become a verdict, and a
    threshold is a judge with a parameter attached -- move it and the same
    answer passes or fails."""
    assert {k.value for k in DecisionKind} == {"predicate", "choice"}


# --- an unanswered question is not a wrong one ----------------------------


def test_an_unanswered_question_leaves_the_denominator():
    """Scoring it wrong would make an abstaining agent indistinguishable
    from a confidently incorrect one."""
    spec = _spec()
    outcomes = score_case(spec, spec.cases[0], [
        DecisionAnswer(question="is_urgent", value=True),
        # `category` simply absent.
    ])
    by_q = {o.question: o for o in outcomes}
    assert by_q["is_urgent"].correct is True
    assert by_q["category"].decidable is False
    assert by_q["category"].correct is None, "None, never False"

    rates = accuracy_by_question(outcomes)
    assert rates["is_urgent"] == 1.0
    assert rates["category"] is None, (
        "no decidable cell means no rate -- 0.0 would read as 'always wrong' "
        "about a question nobody answered"
    )
    assert len(unanswered(outcomes)) == 1


def test_an_explicit_abstention_reads_the_same_as_silence():
    spec = _spec()
    outcomes = score_case(spec, spec.cases[0], [
        DecisionAnswer(question="is_urgent", value=None),
        DecisionAnswer(question="category", value="billing"),
    ])
    by_q = {o.question: o for o in outcomes}
    assert by_q["is_urgent"].decidable is False
    assert by_q["is_urgent"].correct is None


def test_an_out_of_domain_answer_is_wrong_but_marked_apart():
    """Its own state, because the remedy differs: a wrong answer is the
    agent's, an unparseable one is usually the output contract's."""
    spec = _spec()
    outcomes = score_case(spec, spec.cases[0], [
        DecisionAnswer(question="is_urgent", value=True),
        DecisionAnswer(question="category", value="refund"),
    ])
    bad = next(o for o in outcomes if o.question == "category")
    assert bad.out_of_domain is True
    assert bad.decidable is True, "it answered -- just not with a real option"
    assert bad.correct is False


def test_an_answer_to_a_question_the_case_never_asked_is_dropped():
    """Not evidence for or against anything."""
    spec = _spec(cases=[DecisionCase(id="c1", input="x",
                                     expect={"is_urgent": True})])
    outcomes = score_case(spec, spec.cases[0], [
        DecisionAnswer(question="is_urgent", value=True),
        DecisionAnswer(question="category", value="billing"),
    ])
    assert [o.question for o in outcomes] == ["is_urgent"]


def test_rates_are_per_question_never_blended():
    """An agent perfect on three questions and useless on a fourth is not
    75% of an agent, and one column cannot say which it is."""
    spec = _spec(cases=[
        DecisionCase(id="a", input="x", expect={"is_urgent": True,
                                                "category": "billing"}),
        DecisionCase(id="b", input="y", expect={"is_urgent": True,
                                                "category": "bug"}),
    ])
    outcomes = []
    outcomes += score_case(spec, spec.cases[0], [
        DecisionAnswer(question="is_urgent", value=True),
        DecisionAnswer(question="category", value="billing"),
    ])
    outcomes += score_case(spec, spec.cases[1], [
        DecisionAnswer(question="is_urgent", value=True),
        DecisionAnswer(question="category", value="feature"),
    ])
    rates = accuracy_by_question(outcomes)
    assert rates["is_urgent"] == 1.0
    assert rates["category"] == 0.5
    assert "overall" not in rates and len(rates) == 2


# --- a probability is not a calibration -----------------------------------


def test_a_probability_is_not_calibratable_unless_declared_so():
    """The Decisions API returns per-answer probabilities and whether they
    are calibrated is contested -- one source says production use requires
    calibrating thresholds yourself, and an independent implementation of
    the same idea states its values are NOT calibrated probabilities of
    correctness.

    A Brier score over uncalibrated numbers looks like calibration and is
    not, so the flag travels with the number and defaults to off.
    """
    bare = DecisionAnswer(question="q", value=True, probability=0.91)
    assert bare.probability == 0.91
    assert bare.probability_calibrated is False
    assert bare.calibratable is False, "recorded, not scoreable"

    declared = DecisionAnswer(question="q", value=True, probability=0.91,
                              probability_calibrated=True)
    assert declared.calibratable is True

    # And a flag with no number behind it is not calibratable either.
    empty = DecisionAnswer(question="q", value=True,
                           probability_calibrated=True)
    assert empty.calibratable is False


# --- loading --------------------------------------------------------------


EVAL = """\
- agent: triage
  persona: support_triage
  questions:
    - name: is_urgent
      kind: predicate
    - name: category
      kind: choice
      choices: [billing, bug, feature]
  cases:
    - id: c1
      input: "charged twice"
      expect:
        is_urgent: true
        category: billing
"""


def test_a_list_of_specs_loads(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text(EVAL, encoding="utf-8")
    specs = load_decisions(path)
    assert len(specs) == 1
    assert specs[0].agent == "triage"
    assert specs[0].persona == "support_triage"
    assert len(specs[0].cases) == 1


def test_a_single_mapping_loads_too(tmp_path):
    """One agent is the common case and should not need a list."""
    path = tmp_path / "evaluation.yaml"
    path.write_text(EVAL.replace("- agent:", "agent:").replace("\n  ", "\n"),
                    encoding="utf-8")
    specs = load_decisions(path)
    assert len(specs) == 1 and specs[0].agent == "triage"


def test_two_specs_for_one_agent_are_refused(tmp_path):
    """Two contracts for one agent is two different bars, and nothing
    downstream could say which one a result was scored against."""
    path = tmp_path / "evaluation.yaml"
    path.write_text(EVAL + EVAL.replace("- agent: triage", "- agent: triage"),
                    encoding="utf-8")
    with pytest.raises(ValidationError) as caught:
        load_decisions(path)
    assert any("more than one spec" in e for e in caught.value.errors)


def test_a_missing_file_is_an_error_not_an_empty_list(tmp_path):
    with pytest.raises(ValidationError) as caught:
        load_decisions(tmp_path / "nope.yaml")
    assert any("not found" in e for e in caught.value.errors)


def test_a_bad_spec_is_named_by_its_agent(tmp_path):
    """'entry 2' is useless in a file with six of them."""
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        "- agent: triage\n"
        "  questions:\n"
        "    - name: category\n"
        "      kind: choice\n"
        "      choices: [only_one]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as caught:
        load_decisions(path)
    assert any("triage" in e for e in caught.value.errors)


def test_the_agent_name_is_checked_against_a_project_when_given(tmp_path):
    """A spec naming an agent that does not exist scores nothing and
    reports nothing, so the typo is silent until somebody wonders why a
    column is empty."""
    from commonadk.models import InteractionGraph, Project, ProjectConfig

    project = Project(
        config=ProjectConfig(name="p", default_model="anthropic/claude-x"),
        agents={},
        graph=InteractionGraph(),
    )
    path = tmp_path / "evaluation.yaml"
    path.write_text(EVAL, encoding="utf-8")

    # Without a project the names cannot be checked, and it does not
    # pretend to: loading succeeds.
    assert load_decisions(path)

    with pytest.raises(ValidationError) as caught:
        load_decisions(path, project=project)
    assert any("does not define" in e for e in caught.value.errors)


def test_a_scalar_file_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(ValidationError) as caught:
        load_decisions(path)
    assert any("expected a spec mapping" in e for e in caught.value.errors)
