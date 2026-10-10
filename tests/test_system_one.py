"""Estimators on a System One model: typed questions in, probabilities out."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pytest

import skllm
from skllm import (
    ConsensusClassifier,
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelFewShotClassifier,
    MultiLabelZeroShotClassifier,
    RubricScorer,
    Summarizer,
    Vectorizer,
    ZeroShotClassifier,
)
from skllm.backends import BackendError, register_backend, registry
from skllm.prompts import CATEGORY_QUESTION, MEMBERSHIP_QUESTION, RUBRIC_QUESTION
from tests.fakes import FakeBackend, FakeCall, SystemOneFake
from tests.test_serialization import dependencies, manifest

MODEL = "typesafe::jev-latest"
SENTIMENTS = ["negative", "neutral", "positive"]
TOPICS = [["price"], ["delivery", "quality"], ["quality"]]
REVIEWS = ["Too expensive", "Late and broken", "Works great"]


@pytest.fixture
def jev(monkeypatch: pytest.MonkeyPatch) -> Iterator[SystemOneFake]:
    backend = SystemOneFake()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    register_backend("typesafe", backend)
    try:
        yield backend
    finally:
        registry._BACKENDS.pop("typesafe", None)


def choice(probabilities: Mapping[str, float]) -> dict[str, Any]:
    return {
        "category": {
            "type": "choice",
            "choice": max(probabilities, key=lambda label: probabilities[label]),
            "confidence": 0.9,
            "probabilities": dict(probabilities),
        }
    }


def nouls(probabilities: Mapping[str, float]) -> dict[str, Any]:
    return {
        label: {"type": "noul", "noul": probability}
        for label, probability in probabilities.items()
    }


def score(level: float) -> dict[str, Any]:
    return {"score": {"type": "score", "score": level, "confidence": 0.9}}


def by_state(answers: Mapping[str, Any]) -> Callable[[FakeCall], Any]:
    """Answer each call according to the text it asks about."""
    return lambda call: answers[call.state]


# --- single-label -------------------------------------------------------------


def test_a_zero_shot_classifier_asks_one_choice_about_the_text(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = choice({"negative": 0.1, "neutral": 0.2, "positive": 0.7})
    classifier = ZeroShotClassifier(model=MODEL, verbose=0).fit(None, SENTIMENTS)

    assert classifier.predict(["Works great"]).tolist() == ["positive"]

    (call,) = jev.calls
    assert call.kind == "evaluate"
    assert call.model == "jev-latest"
    assert call.state == "Works great"
    assert call.questions == {
        "category": {
            "type": "choice",
            "instructions": CATEGORY_QUESTION,
            "criteria": dict.fromkeys(SENTIMENTS),
        }
    }


def test_single_label_probabilities_are_the_models_own(jev: SystemOneFake) -> None:
    jev.answers_script = by_state(
        {
            "Too expensive": choice(
                {"negative": 0.8, "neutral": 0.15, "positive": 0.05}
            ),
            "Works great": choice({"negative": 0.0, "neutral": 0.3, "positive": 0.7}),
        }
    )
    classifier = ZeroShotClassifier(model=MODEL, verbose=0).fit(None, SENTIMENTS)

    probabilities = classifier.predict_proba(["Too expensive", "Works great"])

    np.testing.assert_allclose(probabilities, [[0.8, 0.15, 0.05], [0.0, 0.3, 0.7]])
    assert classifier.usage_.last.requests == 2


def test_predict_is_the_argmax_of_predict_proba_ties_to_the_earliest_class(
    jev: SystemOneFake,
) -> None:
    """The provider's own ``choice`` may break the tie differently."""
    jev.answers_script = choice({"negative": 0.1, "neutral": 0.45, "positive": 0.45})
    jev.answers_script["category"]["choice"] = "positive"
    classifier = ZeroShotClassifier(model=MODEL, verbose=0).fit(None, SENTIMENTS)

    assert classifier.predict(["Fine, I guess"]).tolist() == ["neutral"]


def test_a_failed_sample_is_filled_with_the_default_label_and_a_zero_row(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = by_state(
        {
            "Works great": choice({"negative": 0.0, "neutral": 0.1, "positive": 0.9}),
            "???": BackendError("boom"),
        }
    )
    classifier = ZeroShotClassifier(
        model=MODEL, default_label="neutral", verbose=0
    ).fit(None, SENTIMENTS)

    with pytest.warns(UserWarning, match="1 of 2 samples failed"):
        labels = classifier.predict(["Works great", "???"])
    with pytest.warns(UserWarning):
        probabilities = classifier.predict_proba(["Works great", "???"])

    assert labels.tolist() == ["positive", "neutral"]
    assert probabilities[1].tolist() == [0.0, 0.0, 0.0]


def test_probabilities_that_miss_a_class_fail_the_sample(jev: SystemOneFake) -> None:
    jev.answers_script = choice({"negative": 0.3, "positive": 0.7})
    classifier = ZeroShotClassifier(model=MODEL, on_error="raise", verbose=0).fit(
        None, SENTIMENTS
    )

    with pytest.raises(BackendError, match="probability per fitted class"):
        classifier.predict(["Works great"])


def test_a_few_shot_classifier_describes_each_class_by_its_examples(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = choice({"negative": 0.9, "positive": 0.1})
    classifier = FewShotClassifier(model=MODEL, verbose=0).fit(
        ["Awful", "Love it", "Broke in a day"], ["negative", "positive", "negative"]
    )

    classifier.predict(["Terrible"])

    assert jev.calls[0].questions == {
        "category": {
            "type": "choice",
            "instructions": CATEGORY_QUESTION,
            "criteria": {
                "negative": {"examples": ["Awful", "Broke in a day"]},
                "positive": {"examples": ["Love it"]},
            },
        }
    }


def test_a_dynamic_few_shot_classifier_demonstrates_with_the_nearest_examples(
    jev: SystemOneFake, fake_backend: FakeBackend
) -> None:
    jev.answers_script = choice({"negative": 0.9, "positive": 0.1})
    classifier = DynamicFewShotClassifier(
        model=MODEL,
        n_examples=1,
        vectorizer=Vectorizer(model="fake::embedder", verbose=0),
        verbose=0,
    ).fit(
        ["awful awful", "zzzz", "love it", "qqqq"],
        ["negative", "negative", "positive", "positive"],
    )

    classifier.predict(["awful"])

    criteria = jev.calls[0].questions["category"]["criteria"]
    assert criteria == {
        "negative": {"examples": ["awful awful"]},
        "positive": {"examples": ["love it"]},
    }


def test_more_classes_than_a_choice_holds_are_rejected_at_fit(
    jev: SystemOneFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SystemOneFake, "max_choice_options", 2)

    with pytest.raises(ValueError, match="at most 2 options"):
        ZeroShotClassifier(model=MODEL).fit(None, SENTIMENTS)


def test_sampled_voting_is_rejected(jev: SystemOneFake) -> None:
    with pytest.raises(ValueError, match="calibrated"):
        ZeroShotClassifier(model=MODEL, n_predictors=3).fit(None, SENTIMENTS)


def test_a_prompt_template_is_rejected(jev: SystemOneFake) -> None:
    with pytest.raises(ValueError, match="prompt_template has no effect"):
        ZeroShotClassifier(model=MODEL, prompt_template="{labels} {text}").fit(
            None, SENTIMENTS
        )


def test_model_params_reach_the_provider(jev: SystemOneFake) -> None:
    jev.answers_script = choice({"negative": 0.1, "positive": 0.9})
    classifier = ZeroShotClassifier(
        model=MODEL, model_params={"beam_width": 4}, verbose=0
    ).fit(None, ["negative", "positive"])

    classifier.predict(["Works great"])

    assert jev.calls[0].params == {"beam_width": 4}


def test_a_system_one_member_votes_in_a_consensus(
    jev: SystemOneFake, fake_backend: FakeBackend
) -> None:
    jev.answers_script = choice({"negative": 0.2, "positive": 0.8})
    fake_backend.structured_script = {"label": "positive"}
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model=MODEL, verbose=0),
            ZeroShotClassifier(model="fake::chat", verbose=0),
        ]
    ).fit(None, ["negative", "positive"])

    assert ensemble.predict(["Works great"]).tolist() == ["positive"]


# --- multi-label --------------------------------------------------------------


def test_a_multi_label_classifier_asks_one_yes_no_per_class_in_one_request(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = nouls({"delivery": 0.2, "price": 0.9, "quality": 0.6})
    classifier = MultiLabelZeroShotClassifier(model=MODEL, verbose=0).fit(None, TOPICS)

    assert classifier.predict(["Too expensive, but solid"]) == [["price", "quality"]]

    (call,) = jev.calls
    assert call.state == "Too expensive, but solid"
    assert call.questions == {
        label: {
            "type": "noul",
            "instructions": {"category": label, "question": MEMBERSHIP_QUESTION},
        }
        for label in ["delivery", "price", "quality"]
    }


def test_multi_label_keeps_only_labels_more_likely_than_not_most_likely_first(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = nouls({"delivery": 0.5, "price": 0.7, "quality": 0.95})
    classifier = MultiLabelZeroShotClassifier(model=MODEL, verbose=0).fit(None, TOPICS)

    assert classifier.predict(["text"]) == [["quality", "price"]]
    np.testing.assert_allclose(classifier.predict_proba(["text"]), [[0.5, 0.7, 0.95]])


def test_multi_label_answers_are_capped_at_max_labels(jev: SystemOneFake) -> None:
    jev.answers_script = nouls({"delivery": 0.8, "price": 0.7, "quality": 0.95})
    classifier = MultiLabelZeroShotClassifier(model=MODEL, max_labels=2, verbose=0).fit(
        None, TOPICS
    )

    assert classifier.predict(["text"]) == [["quality", "delivery"]]


def test_a_failed_multi_label_sample_is_filled_with_no_labels(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = nouls({"delivery": 0.8, "price": 0.7})
    classifier = MultiLabelZeroShotClassifier(model=MODEL, verbose=0).fit(None, TOPICS)

    with pytest.warns(UserWarning, match="1 of 1 samples failed"):
        assert classifier.predict(["text"]) == [[]]
    assert "'quality'" in classifier.errors_[0].error


def test_a_multi_label_few_shot_classifier_gives_each_class_its_examples(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = nouls({"delivery": 0.1, "price": 0.9, "quality": 0.2})
    classifier = MultiLabelFewShotClassifier(model=MODEL, verbose=0).fit(
        REVIEWS, TOPICS
    )

    classifier.predict(["Pricey"])

    questions = jev.calls[0].questions
    assert questions["price"]["criteria"] == {"true": {"examples": ["Too expensive"]}}
    assert questions["quality"]["criteria"] == {
        "true": {"examples": ["Late and broken", "Works great"]}
    }


# --- scoring ------------------------------------------------------------------


def test_a_rubric_scorer_rates_on_one_level_per_whole_number(
    jev: SystemOneFake,
) -> None:
    jev.answers_script = score(2.5)
    scorer = RubricScorer(model=MODEL, rubric="How helpful", scale=(1, 5), verbose=0)

    assert scorer.predict(["Very helpful"]).tolist() == [3.5]

    (call,) = jev.calls
    assert call.state == "Very helpful"
    assert call.questions == {
        "score": {
            "type": "score",
            "instructions": {"rubric": "How helpful", "question": RUBRIC_QUESTION},
            "criteria": ["1", "2", "3", "4", "5"],
        }
    }


def test_a_level_beyond_the_scale_is_clamped(jev: SystemOneFake) -> None:
    jev.answers_script = score(12.0)
    scorer = RubricScorer(model=MODEL, rubric="How helpful", scale=(1, 5), verbose=0)

    assert scorer.predict(["text"]).tolist() == [5.0]


def test_an_answer_without_a_level_is_a_nan(jev: SystemOneFake) -> None:
    jev.answers_script = {"score": {"type": "score"}}
    scorer = RubricScorer(model=MODEL, rubric="How helpful", scale=(1, 5), verbose=0)

    with pytest.warns(UserWarning):
        assert math.isnan(scorer.predict(["text"])[0])


@pytest.mark.parametrize("scale", [(0, 10), (0.5, 5), (1, 4.5)])
def test_a_scale_that_does_not_fit_the_levels_is_rejected(
    jev: SystemOneFake, scale: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="at most 10 levels"):
        RubricScorer(model=MODEL, rubric="How helpful", scale=scale).fit()


def test_the_widest_scale_uses_every_level(jev: SystemOneFake) -> None:
    jev.answers_script = score(0.0)
    RubricScorer(model=MODEL, rubric="How helpful", scale=(1, 10), verbose=0).predict(
        ["text"]
    )

    assert len(jev.calls[0].questions["score"]["criteria"]) == 10


# --- what a System One model cannot serve -------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [Summarizer(model=MODEL), Vectorizer(model=MODEL)],
    ids=["Summarizer", "Vectorizer"],
)
def test_estimators_that_need_text_refuse_a_system_one_model(
    jev: SystemOneFake, estimator: Any
) -> None:
    with pytest.raises(ValueError, match="needs a model that generates text"):
        estimator.fit(REVIEWS)

    assert jev.calls == []


# --- serving ------------------------------------------------------------------


def test_a_bundle_declares_the_typesafe_key_and_extra(
    jev: SystemOneFake, tmp_path: Any
) -> None:
    path = str(tmp_path / "model.fnnx")
    ZeroShotClassifier(model=MODEL, verbose=0).fit(None, SENTIMENTS).save(path)

    assert [var["name"] for var in manifest(path)["env_vars"]] == ["TYPESAFE_API_KEY"]
    assert dependencies(path)[0] == f"scikit-llm[typesafe]=={skllm.__version__}"
