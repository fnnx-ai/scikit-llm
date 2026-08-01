from __future__ import annotations

import pickle
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from skllm.classification import MultiLabelZeroShotClassifier, ZeroShotClassifier
from skllm.config import config_context
from skllm.core import Requirement, SampleError, UsageRecord
from skllm.ensemble import ConsensusClassifier
from tests.fakes import FakeBackend, FakeCall

TEXTS = ["buy cheap pills now", "lunch at noon?"]
CLASSES = ["ham", "spam"]


def members(*model_ids: str, **params: Any) -> list[ZeroShotClassifier]:
    """A panel telling itself apart by model id, so scripts can answer per member."""
    return [
        ZeroShotClassifier(model=f"fake::{model_id}", verbose=0, **params)
        for model_id in model_ids
    ]


def voting(
    ballots: Mapping[str, Sequence[str | None]],
) -> Callable[[FakeCall], Any]:
    """Answer per member and per sample, whichever order the calls arrive in.

    ``None`` stands for a member that fails on that sample, which its own
    ``"fill"`` policy turns into an abstention.
    """

    def answer(call: FakeCall) -> Any:
        index = next(index for index, text in enumerate(TEXTS) if text in call.prompt)
        vote = ballots[call.model][index]
        if vote is None:
            return RuntimeError("the provider rejected the request")
        return {"label": vote}

    return answer


@pytest.fixture
def panel(fake_backend: FakeBackend) -> ConsensusClassifier:
    """Three members: 2-1 for spam on the first text, 1-1 (one abstains) on the second."""
    fake_backend.structured_script = voting(
        {"a": ["spam", "spam"], "b": ["spam", "ham"], "c": ["ham", None]}
    )
    return ConsensusClassifier(members("a", "b", "c"), verbose=0).fit(None, CLASSES)


# --- voting ------------------------------------------------------------------


def test_the_majority_label_wins(panel: ConsensusClassifier) -> None:
    with pytest.warns(UserWarning):  # member c fills its second sample
        predictions = panel.predict(TEXTS)

    assert predictions[0] == "spam"


def test_a_tie_goes_to_the_earliest_listed_voting_member(
    panel: ConsensusClassifier,
) -> None:
    """The second text splits 1-1 between members a (spam) and b (ham); c abstains.

    ``ham`` sorts first, so a tie broken by class order would answer otherwise.
    """
    with pytest.warns(UserWarning):
        predictions = panel.predict(TEXTS)

    assert predictions[1] == "spam"


def test_probabilities_are_the_vote_fractions(panel: ConsensusClassifier) -> None:
    with pytest.warns(UserWarning):
        probabilities = panel.predict_proba(TEXTS)

    assert probabilities[0].tolist() == pytest.approx([1 / 3, 2 / 3])
    # Normalized over the two members that voted, not over all three.
    assert probabilities[1].tolist() == [0.5, 0.5]


def test_every_member_answers_the_whole_batch(
    panel: ConsensusClassifier, fake_backend: FakeBackend
) -> None:
    with pytest.warns(UserWarning):
        panel.predict(TEXTS)

    asked = [call.model for call in fake_backend.calls]
    assert sorted(asked) == ["a", "a", "b", "b", "c", "c"]


def test_members_are_asked_concurrently(fake_backend: FakeBackend) -> None:
    barrier = threading.Barrier(3, timeout=5)

    def meet(call: FakeCall) -> dict[str, str]:
        barrier.wait()
        return {"label": "spam"}

    fake_backend.structured_script = meet
    ensemble = ConsensusClassifier(members("a", "b", "c"), verbose=0).fit(None, CLASSES)

    assert list(ensemble.predict(["buy cheap pills now"])) == ["spam"]


def test_predictions_keep_the_input_order(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["spam", "ham"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    assert list(ensemble.predict(TEXTS)) == ["spam", "ham"]
    assert list(ensemble.predict(list(reversed(TEXTS)))) == ["ham", "spam"]


# --- weights -----------------------------------------------------------------


def test_weights_scale_the_votes(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting(
        {"a": ["ham", "ham"], "b": ["spam", "spam"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), weights=[3, 1], verbose=0).fit(
        None, CLASSES
    )

    predictions = ensemble.predict(TEXTS[:1])
    probabilities = ensemble.predict_proba(TEXTS[:1])

    assert list(predictions) == ["ham"]
    assert probabilities.tolist() == [[0.75, 0.25]]


def test_weighted_probabilities_ignore_the_weight_of_an_abstaining_member(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"a": ["ham", None], "b": ["spam", "spam"]})
    ensemble = ConsensusClassifier(members("a", "b"), weights=[3, 1], verbose=0).fit(
        None, CLASSES
    )

    with pytest.warns(UserWarning):
        probabilities = ensemble.predict_proba(TEXTS)

    assert probabilities.tolist() == [[0.75, 0.25], [0.0, 1.0]]


def test_weights_that_tie_only_approximately_still_go_to_member_order(
    fake_backend: FakeBackend,
) -> None:
    """0.1 + 0.2 != 0.3 in floats, but a panel weighted that way is tied."""
    fake_backend.structured_script = voting(
        {"a": ["ham"], "b": ["spam"], "c": ["spam"]}
    )
    ensemble = ConsensusClassifier(
        members("a", "b", "c"), weights=[0.3, 0.1, 0.2], verbose=0
    ).fit(None, CLASSES)

    assert list(ensemble.predict(TEXTS[:1])) == ["ham"]


def test_equal_weights_leave_the_tie_break_to_member_order(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["ham", "spam"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), weights=[2, 2], verbose=0).fit(
        None, CLASSES
    )

    assert list(ensemble.predict(TEXTS)) == ["spam", "ham"]


# --- failures ----------------------------------------------------------------


def test_a_sample_no_member_voted_for_is_filled_and_reported(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"a": ["spam", None], "b": ["spam", None]})
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    with pytest.warns(UserWarning, match="ConsensusClassifier: 1 of 2 samples"):
        predictions = ensemble.predict(TEXTS)

    assert list(predictions) == ["spam", None]
    assert ensemble.errors_ == (
        SampleError(
            index=1,
            error="no member voted: every one of them filled this sample under its own policy",
        ),
    )
    assert ensemble.usage_.last.failures == 1


def test_a_sample_without_votes_takes_the_ensemble_default_label(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"a": [None, None], "b": [None, None]})
    ensemble = ConsensusClassifier(
        members("a", "b"), default_label="unknown", verbose=0
    ).fit(None, CLASSES)

    with pytest.warns(UserWarning):
        predictions = ensemble.predict(TEXTS)

    assert list(predictions) == ["unknown", "unknown"]


def test_probabilities_of_a_sample_without_votes_are_zeros(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"a": ["spam", None], "b": ["spam", None]})
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    with pytest.warns(UserWarning):
        probabilities = ensemble.predict_proba(TEXTS)

    assert probabilities.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_a_member_that_raises_aborts_the_batch(fake_backend: FakeBackend) -> None:
    """A member's own error policy decides what its failures do."""
    fake_backend.structured_script = voting(
        {"a": ["spam", "spam"], "b": ["spam", None]}
    )
    ensemble = ConsensusClassifier(
        [*members("a"), *members("b", on_error="raise")], verbose=0
    ).fit(None, CLASSES)

    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        ensemble.predict(TEXTS)


def test_an_aborted_batch_reports_what_the_panel_spent(
    fake_backend: FakeBackend,
) -> None:
    """An abort must not leave the previous operation's report standing."""
    fake_backend.structured_script = voting({"a": ["spam", None]})
    member = members("a", max_concurrency=1)[0]
    ensemble = ConsensusClassifier([member], verbose=0).fit(None, CLASSES)

    with pytest.warns(UserWarning):
        ensemble.predict(TEXTS[1:])
    assert [error.index for error in ensemble.errors_] == [0]

    member.set_params(on_error="raise")
    ensemble.fit(None, CLASSES)  # the clones now abort where they used to fill
    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        ensemble.predict(TEXTS)

    assert ensemble.errors_ == ()
    assert ensemble.usage_.last.requests == 1  # what the aborted batch paid for
    assert ensemble.usage_.total.requests == 1


def test_the_ensemble_raise_policy_aborts_a_sample_without_votes(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"a": ["spam", None], "b": ["spam", None]})
    ensemble = ConsensusClassifier(members("a", "b"), on_error="raise", verbose=0).fit(
        None, CLASSES
    )

    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="got no vote"):
        ensemble.predict(TEXTS)


# --- fitting -----------------------------------------------------------------


def test_fit_leaves_the_constructor_members_unfitted(fake_backend: FakeBackend) -> None:
    panel = members("a", "b")
    ensemble = ConsensusClassifier(panel, verbose=0)

    ensemble.fit(None, CLASSES)

    assert ensemble.estimators is panel
    for member, fitted in zip(panel, ensemble.estimators_, strict=True):
        assert fitted is not member
        assert fitted.get_params() == member.get_params()
        check_is_fitted(fitted)
        with pytest.raises(NotFittedError):
            check_is_fitted(member)


def test_a_pre_fitted_member_is_refit_as_a_clone(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting({"a": ["spam", "spam"]})
    member = members("a")[0].fit(None, ["other", "labels"])

    ensemble = ConsensusClassifier([member], verbose=0).fit(None, CLASSES)

    assert member.classes_ == ["labels", "other"]
    assert ensemble.estimators_[0].classes_ == CLASSES
    assert ensemble.classes_ == CLASSES


def test_the_ensemble_derives_its_own_classes(fake_backend: FakeBackend) -> None:
    ensemble = ConsensusClassifier(members("a"), verbose=0).fit(
        None, ["spam", "ham", "spam"]
    )

    assert ensemble.classes_ == ["ham", "spam"]


def test_non_string_labels_are_rejected(fake_backend: FakeBackend) -> None:
    with pytest.raises(TypeError, match="labels must be strings"):
        ConsensusClassifier(members("a"), verbose=0).fit(None, [0, 1])


def test_members_are_fitted_on_the_same_data(fake_backend: FakeBackend) -> None:
    from skllm.classification import FewShotClassifier

    member = FewShotClassifier(model="fake::a", verbose=0)
    ensemble = ConsensusClassifier([member], verbose=0).fit(TEXTS, CLASSES)

    assert ensemble.estimators_[0].examples_ == (
        ("buy cheap pills now", "ham"),
        ("lunch at noon?", "spam"),
    )


# --- validation --------------------------------------------------------------


def test_an_empty_panel_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one classifier"):
        ConsensusClassifier([]).fit(None, CLASSES)


def test_a_multi_label_member_is_rejected() -> None:
    ensemble = ConsensusClassifier([MultiLabelZeroShotClassifier(model="fake::a")])

    with pytest.raises(TypeError, match="single-label"):
        ensemble.fit(None, CLASSES)


def test_a_foreign_member_is_rejected() -> None:
    with pytest.raises(TypeError, match="single-label"):
        ConsensusClassifier(["not an estimator"]).fit(None, CLASSES)  # type: ignore[list-item]


def test_a_panel_that_is_not_a_list_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        ConsensusClassifier(ZeroShotClassifier(model="fake::a")).fit(None, CLASSES)


def test_weights_must_match_the_panel() -> None:
    with pytest.raises(ValueError, match="one weight per member"):
        ConsensusClassifier(members("a", "b"), weights=[1]).fit(None, CLASSES)


@pytest.mark.parametrize("weight", [0, -1, "heavy", None])
def test_weights_must_be_positive_numbers(weight: Any) -> None:
    with pytest.raises(ValueError, match="positive number"):
        ConsensusClassifier(members("a", "b"), weights=[1, weight]).fit(None, CLASSES)


def test_default_label_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="default_label"):
        ConsensusClassifier(members("a"), default_label=1).fit(None, CLASSES)  # type: ignore[arg-type]


def test_an_invalid_error_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_error"):
        ConsensusClassifier(members("a"), on_error="ignore").fit(None, CLASSES)


# --- reporting ---------------------------------------------------------------


def test_usage_adds_up_what_the_members_spent(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["ham", "spam"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    ensemble.predict(TEXTS)

    members_ = ensemble.estimators_
    assert ensemble.usage_.last.requests == 4
    assert ensemble.usage_.last.input_tokens == sum(
        member.usage_.last.input_tokens for member in members_
    )
    assert ensemble.usage_.total == ensemble.usage_.last
    assert ensemble.errors_ == ()


def test_requirements_cover_every_member(fake_backend: FakeBackend) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="fake::a", alias="team_a"),
            ZeroShotClassifier(model="fake::b", alias="team_b"),
            ZeroShotClassifier(model="openai::gpt-5"),
        ],
        verbose=0,
    )

    assert ensemble.llm_requirements() == (
        Requirement("fake", "team_a"),
        Requirement("fake", "team_b"),
        Requirement("openai", None),
    )
    # Fitting adds the clones, which need exactly the same credentials.
    assert ensemble.fit(None, CLASSES).llm_requirements() == (
        Requirement("fake", "team_a"),
        Requirement("fake", "team_b"),
        Requirement("openai", None),
    )


# --- aliases -----------------------------------------------------------------


def aliased(*aliases: str, **params: Any) -> list[ZeroShotClassifier]:
    """A panel whose members are told apart by both model id and alias."""
    return [
        ZeroShotClassifier(model=f"fake::{alias}", alias=alias, verbose=0, **params)
        for alias in aliases
    ]


def test_each_member_authenticates_by_its_own_alias(fake_backend: FakeBackend) -> None:
    """Same provider, different keys — what per-provider resolution cannot express."""
    fake_backend.structured_script = voting({"fast": ["spam"], "careful": ["spam"]})
    ensemble = ConsensusClassifier(aliased("fast", "careful"), verbose=0).fit(
        None, CLASSES
    )

    with config_context(credentials={"fast": "sk-fast", "careful": "sk-careful"}):
        ensemble.predict(TEXTS[:1])

    assert {call.model: call.api_key for call in fake_backend.calls} == {
        "fast": "sk-fast",
        "careful": "sk-careful",
    }


def test_an_override_reaches_only_the_member_it_names(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting({"fast": ["spam"], "careful": ["spam"]})
    ensemble = ConsensusClassifier(
        aliased("fast", "careful", model_params={"temperature": 0.2}), verbose=0
    ).fit(None, CLASSES)

    with config_context(
        credentials={"fast": "sk-fast", "careful": "sk-careful"},
        overrides={"careful": {"temperature": 0.9}},
    ):
        ensemble.predict(TEXTS[:1])

    assert {call.model: call.params["temperature"] for call in fake_backend.calls} == {
        "fast": 0.2,
        "careful": 0.9,
    }


# --- sklearn compliance ------------------------------------------------------


def test_params_survive_a_clone(fake_backend: FakeBackend) -> None:
    ensemble = ConsensusClassifier(
        members("a", "b"), weights=[1, 2], default_label="unknown", max_concurrency=2
    )

    cloned = clone(ensemble)
    params = cloned.get_params()

    # The panel itself is cloned, so it compares member by member rather than
    # by identity, as every other param does.
    assert params == {**ensemble.get_params(), "estimators": params["estimators"]}
    assert vars(ensemble) == ensemble.get_params()
    assert [member.get_params() for member in cloned.estimators] == [
        member.get_params() for member in ensemble.estimators
    ]
    assert cloned.estimators[0] is not ensemble.estimators[0]


def test_an_unfitted_ensemble_raises() -> None:
    ensemble = ConsensusClassifier(members("a"))

    assert get_tags(ensemble).requires_fit

    with pytest.raises(NotFittedError):
        ensemble.predict(TEXTS)


def test_an_empty_batch_asks_nothing(
    panel: ConsensusClassifier, fake_backend: FakeBackend
) -> None:
    predictions = panel.predict([])
    probabilities = panel.predict_proba([])

    assert list(predictions) == []
    assert probabilities.shape == (0, 2)
    assert fake_backend.calls == []
    assert panel.usage_.total == panel.usage_.last == UsageRecord()


def test_pickling_keeps_the_panel_and_drops_the_member_keys(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["spam", "ham"]}
    )
    ensemble = ConsensusClassifier(
        members("a", "b", api_key="sk-secret"), verbose=0
    ).fit(None, CLASSES)

    payload = pickle.dumps(ensemble)
    restored = pickle.loads(payload)

    assert b"sk-secret" not in payload
    assert [member.api_key for member in restored.estimators_] == [None, None]
    assert restored.classes_ == CLASSES

    with config_context(fake_api_key="sk-new"):
        assert list(restored.predict(TEXTS)) == ["spam", "ham"]
    assert {call.api_key for call in fake_backend.calls} == {"sk-new"}


def test_a_context_credential_reaches_every_member(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["spam", "ham"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    with config_context(fake_api_key="sk-ctx"):
        ensemble.predict(TEXTS)

    assert {call.api_key for call in fake_backend.calls} == {"sk-ctx"}


def test_the_ensemble_scores_like_a_classifier(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = voting(
        {"a": ["spam", "ham"], "b": ["spam", "ham"]}
    )
    ensemble = ConsensusClassifier(members("a", "b"), verbose=0).fit(None, CLASSES)

    assert ensemble.score(TEXTS, np.array(["spam", "ham"])) == 1.0
