from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

from skllm.backends import BackendError
from skllm.clustering import ClusterLabeler
from skllm.core import UsageRecord
from tests.fakes import FakeBackend, FakeCall

FAKE_MODEL = "fake::m"

TEXTS = [
    "the battery dies within three hours",
    "battery life is far too short",
    "delivery took three weeks",
    "unrelated to anything else here",
]
CLUSTERS = [0, 0, 1, -1]


def name_by_topic(call: FakeCall) -> str:
    return "battery life" if "battery" in call.prompt else "slow delivery"


@pytest.fixture
def labeler(fake_backend: FakeBackend) -> ClusterLabeler:
    fake_backend.chat_script = name_by_topic
    return ClusterLabeler(model=FAKE_MODEL, random_state=0, verbose=0)


# --- naming ------------------------------------------------------------------


def test_every_cluster_but_the_noise_id_is_named_once(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.fit(TEXTS, CLUSTERS)

    names = labeler.transform([0, 1, -1])

    assert len(fake_backend.calls) == 2
    assert names.tolist() == ["battery life", "slow delivery", None]
    assert labeler.names_ == {0: "battery life", 1: "slow delivery"}


def test_a_naming_prompt_carries_that_cluster_alone(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.set_params(max_concurrency=1).fit(TEXTS, CLUSTERS)

    battery, delivery = fake_backend.prompts
    assert TEXTS[0] in battery and TEXTS[1] in battery
    assert TEXTS[2] not in battery
    assert TEXTS[2] in delivery


def test_the_noise_id_reaches_no_model(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.fit(TEXTS, CLUSTERS)

    assert all(TEXTS[3] not in prompt for prompt in fake_backend.prompts)


def test_a_clustering_of_noise_alone_names_nothing(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.fit(TEXTS[:2], [-1, -1])

    assert fake_backend.calls == []
    assert labeler.names_ == {}
    assert labeler.transform([-1, -1]).tolist() == [None, None]


def test_cluster_ids_may_arrive_as_an_array(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    """What a clusterer hands back is a numpy array of its own integer type."""
    clusters = np.asarray(CLUSTERS)

    names = labeler.fit(np.asarray(TEXTS), clusters).transform(clusters)

    assert names.tolist() == ["battery life", "battery life", "slow delivery", None]


# --- exemplar sampling -------------------------------------------------------


def test_only_n_exemplars_of_a_cluster_are_shown(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    texts = [f"sample number {index}" for index in range(5)]

    labeler.set_params(n_exemplars=2).fit(texts, [0] * 5)

    prompt = fake_backend.prompts[0]
    assert sum(text in prompt for text in texts) == 2


def test_a_cluster_smaller_than_n_exemplars_shows_what_it_has(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.set_params(n_exemplars=10).fit(TEXTS[:1], [0])

    assert TEXTS[0] in fake_backend.prompts[0]


def test_the_random_state_fixes_which_exemplars_are_shown(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    texts = [f"sample number {index}" for index in range(6)]
    clusters = [0, 0, 0, 1, 1, 1]
    labeler.set_params(n_exemplars=2, max_concurrency=1)

    labeler.fit(texts, clusters)
    first = list(fake_backend.prompts)
    fake_backend.calls.clear()
    clone(labeler).fit(texts, clusters)

    assert fake_backend.prompts == first


def test_a_different_random_state_samples_different_exemplars(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    """Repeatability alone would also hold for a sampler that ignored the seed."""
    texts = [f"sample number {index}" for index in range(6)]
    clusters = [0, 0, 0, 1, 1, 1]
    labeler.set_params(n_exemplars=2, max_concurrency=1)

    def shown(random_state: int) -> list[str]:
        fake_backend.calls.clear()
        clone(labeler).set_params(random_state=random_state).fit(texts, clusters)
        return list(fake_backend.prompts)

    assert shown(0) != shown(1)


# --- transform ---------------------------------------------------------------


def test_an_id_unseen_at_fit_raises(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.fit(TEXTS, CLUSTERS)

    with pytest.raises(ValueError, match="Cluster 7 was not seen at fit"):
        labeler.transform([0, 7])


def test_fit_transform_names_the_clustering_just_fitted(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    names = labeler.fit_transform(TEXTS, CLUSTERS)

    assert len(fake_backend.calls) == 2  # one per cluster, not one per text
    assert names.tolist() == ["battery life", "battery life", "slow delivery", None]


@pytest.mark.parametrize("ids", [[0, 1, -1], []])
def test_transform_reports_the_nothing_it_spent(
    ids: list[int], fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    """The naming was paid for at fit; the lookups that follow cost nothing."""
    labeler.fit(TEXTS, CLUSTERS)
    named = labeler.usage_.total
    fake_backend.calls.clear()

    labeler.transform(ids)

    assert fake_backend.calls == []
    assert labeler.usage_.last == UsageRecord()
    assert labeler.usage_.total == named
    assert named.requests == 2


# --- failures ----------------------------------------------------------------


def test_a_naming_failure_aborts_the_fit_under_the_fill_policy(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    """A half-named mapping would poison every later transform, so fill cannot apply."""
    fake_backend.chat_script = lambda call: (
        "battery life"
        if "battery" in call.prompt
        else RuntimeError("the provider rejected the request")
    )

    assert labeler.on_error == "fill"
    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        labeler.fit(TEXTS, CLUSTERS)

    with pytest.raises(NotFittedError):
        labeler.transform([0])


def test_an_empty_name_is_a_failure(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    fake_backend.chat_script = "  \n "

    with pytest.raises(BackendError, match="empty name"):
        labeler.fit(TEXTS, CLUSTERS)


def test_surrounding_whitespace_is_trimmed_from_a_name(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    fake_backend.chat_script = "\n  battery life\n"

    assert labeler.fit(TEXTS[:1], [0]).names_ == {0: "battery life"}


# --- parameters --------------------------------------------------------------


def test_texts_and_cluster_ids_must_match_up(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    with pytest.raises(ValueError, match="4 samples but y has 2 cluster ids"):
        labeler.fit(TEXTS, [0, 1])


@pytest.mark.parametrize("clusters", [["0", "1"], [0.0, 1.0], [True, False]])
def test_cluster_ids_must_be_integers(
    clusters: Any, fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    with pytest.raises(TypeError, match="cluster ids"):
        labeler.fit(TEXTS[:2], clusters)


@pytest.mark.parametrize("clusters", [None, 0, "0"])
def test_a_scalar_is_not_a_clustering(
    clusters: Any, fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    with pytest.raises(TypeError, match="sequence of cluster ids"):
        labeler.fit(TEXTS[:1], clusters)


@pytest.mark.parametrize("value", [0, -1, 1.5, None])
def test_invalid_n_exemplars_are_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="n_exemplars"):
        ClusterLabeler(model=FAKE_MODEL, n_exemplars=value).fit(TEXTS, CLUSTERS)


# --- prompts and sklearn compliance ------------------------------------------


def test_the_default_template_matches_its_documented_placeholders() -> None:
    """A default template is exactly what an override is held to."""
    ClusterLabeler._check_prompt_template(ClusterLabeler.default_prompt_template)


def test_an_invalid_prompt_template_names_the_placeholders(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        ClusterLabeler(model=FAKE_MODEL, prompt_template="Name {samples}").fit(
            TEXTS, CLUSTERS
        )

    message = str(excinfo.value)
    assert "missing ['texts']" in message
    assert "unknown ['samples']" in message


def test_the_labeler_is_stateful(fake_backend: FakeBackend) -> None:
    labeler = ClusterLabeler(model=FAKE_MODEL, verbose=0)

    assert get_tags(labeler).requires_fit
    with pytest.raises(NotFittedError):
        labeler.transform([0])


def test_serving_a_labeler_needs_no_credentials(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    """Its only requests happen at fit, so a bundle must declare no key."""
    labeler.fit(TEXTS, CLUSTERS)

    assert labeler.llm_requirements() == ()


def test_params_survive_a_clone() -> None:
    labeler = ClusterLabeler(
        model=FAKE_MODEL, n_exemplars=4, random_state=7, api_key="sk-secret"
    )

    cloned = clone(labeler)

    assert cloned.get_params() == labeler.get_params()
    assert cloned.random_state == 7
    assert cloned.api_key == "sk-secret"
    assert vars(labeler) == labeler.get_params()


def test_an_empty_clustering_issues_no_requests(
    fake_backend: FakeBackend, labeler: ClusterLabeler
) -> None:
    labeler.fit([], [])

    assert fake_backend.calls == []
    assert labeler.usage_.last == UsageRecord()
    assert labeler.transform([]).tolist() == []
