from __future__ import annotations

import re
from typing import Any

import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from skllm.core import SampleError, UsageRecord
from skllm.tagging import NER, render_tagged_html
from tests.fakes import FakeBackend

FAKE_MODEL = "fake::m"

ENTITIES = {"PER": "a person", "ORG": "an organization"}

TEXT = "Tim Cook leads Apple."
TAGGED = "<PER>Tim Cook</PER> leads <ORG>Apple</ORG>."

#: Deliberately not the estimator's own parser: the invariant is about what any
#: reader of the output sees when it takes the entity tags back off.
_ENTITY_TAG = re.compile(rf"</?({'|'.join(ENTITIES)})>")


def strip_tags(tagged: str) -> str:
    return _ENTITY_TAG.sub("", tagged)


@pytest.fixture
def tagger(fake_backend: FakeBackend) -> NER:
    return NER(model=FAKE_MODEL, entities=ENTITIES, verbose=0)


# --- the tagging invariant ---------------------------------------------------


def test_entities_are_wrapped_in_tags_named_after_their_label(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = TAGGED

    tagged = tagger.transform([TEXT])

    assert len(fake_backend.calls) == 1
    assert list(tagged) == [TAGGED]
    assert strip_tags(tagged[0]) == TEXT


@pytest.mark.parametrize(
    "rewrite",
    [
        "<PER>Tim  Cook</PER>  leads\n<ORG>Apple</ORG> .",  # re-spaced
        "<PER> Tim Cook </PER> leads <ORG> Apple </ORG>.",  # spacing inside the tags
        "<per>Tim Cook</per> leads <org>Apple</org>.",  # lowercased tag names
    ],
)
def test_a_rewrite_is_realigned_onto_the_source_text(
    rewrite: str, fake_backend: FakeBackend, tagger: NER
) -> None:
    """Whitespace is the only disagreement a rewrite is allowed."""
    fake_backend.chat_script = rewrite

    tagged = tagger.transform([TEXT])

    assert list(tagged) == [TAGGED]
    assert strip_tags(tagged[0]) == TEXT


def test_the_spacing_of_the_source_survives_verbatim(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    """Only the source's own whitespace ends up in the output, wherever it sits."""
    source = "  Tim   Cook\n\tleads Apple.  "
    fake_backend.chat_script = TAGGED

    tagged = tagger.transform([source])

    assert list(tagged) == ["  <PER>Tim   Cook</PER>\n\tleads <ORG>Apple</ORG>.  "]
    assert strip_tags(tagged[0]) == source


def test_nested_tags_cannot_be_reconciled(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    """Spans are flat, so the inner tags stay literal text and fail the sample."""
    fake_backend.chat_script = "<PER>Tim <ORG>Cook</ORG></PER> leads Apple."

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        tagged = tagger.transform([TEXT])

    assert list(tagged) == [TEXT]


def test_a_text_without_entities_comes_back_unchanged(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = "Nothing to see here."

    assert list(tagger.transform(["Nothing to see here."])) == ["Nothing to see here."]


def test_a_tag_left_open_runs_to_the_end_of_the_text(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = "<PER>Tim Cook leads Apple."

    assert list(tagger.transform([TEXT])) == ["<PER>Tim Cook leads Apple.</PER>"]


def test_a_span_holding_no_text_is_dropped(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = "<PER> </PER>Tim Cook leads Apple."

    assert list(tagger.transform([TEXT])) == [TEXT]


def test_angle_brackets_of_the_source_are_not_read_as_tags(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    """Only the fitted entity labels name a tag, so other markup survives."""
    source = "Wrap it in <b> to embolden it."
    fake_backend.chat_script = source

    assert list(tagger.transform([source])) == [source]


# --- reconciliation failures -------------------------------------------------


def test_a_rewrite_that_changed_the_text_fills_with_the_untagged_original(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = "<PER>Timothy Cook</PER> runs <ORG>Apple Inc</ORG>."

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        tagged = tagger.transform([TEXT])

    assert list(tagged) == [TEXT]
    assert [error.index for error in tagger.errors_] == [0]
    assert "does not reproduce the text sample" in tagger.errors_[0].error


def test_a_tag_outside_the_entity_schema_fails_the_sample(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    """An unasked-for tag is left in the text rather than quietly deleted."""
    fake_backend.chat_script = "<MISC>Tim Cook</MISC> leads Apple."

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        tagged = tagger.transform([TEXT])

    assert list(tagged) == [TEXT]
    assert len(tagger.errors_) == 1


def test_only_the_failing_sample_is_filled_with_its_own_text(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    other = "Satya Nadella runs Microsoft."
    fake_backend.chat_script = lambda call: (
        TAGGED if TEXT in call.prompt else "Satya Nadella runs Google."
    )

    with pytest.warns(UserWarning, match="1 of 2 samples"):
        tagged = tagger.transform([TEXT, other])

    assert list(tagged) == [TAGGED, other]
    assert [error.index for error in tagger.errors_] == [1]


def test_a_request_failure_fills_with_the_untagged_text(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = RuntimeError("the provider rejected the request")

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        tagged = tagger.transform([TEXT])

    assert list(tagged) == [TEXT]
    assert tagger.errors_ == (
        SampleError(index=0, error="RuntimeError: the provider rejected the request"),
    )
    assert tagger.usage_.last.failures == 1


def test_the_raise_policy_aborts_the_batch(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = "<PER>Somebody else</PER> entirely."
    tagger.set_params(on_error="raise")

    with pytest.raises(Exception, match="does not reproduce the text sample"):
        tagger.transform([TEXT])


# --- prompts and params ------------------------------------------------------


def test_the_prompt_lists_every_entity_with_its_description(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = TAGGED

    tagger.transform([TEXT])

    prompt = fake_backend.prompts[0]
    assert "<PER>...</PER>: a person" in prompt
    assert "<ORG>...</ORG>: an organization" in prompt
    assert TEXT in prompt


def test_the_default_template_matches_its_documented_placeholders() -> None:
    NER._check_prompt_template(NER.default_prompt_template)


def test_an_invalid_prompt_template_names_the_placeholders(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        NER(
            model=FAKE_MODEL,
            entities=ENTITIES,
            prompt_template="Tag {text} with {tags}",
        ).transform([TEXT])

    message = str(excinfo.value)
    assert "missing ['entities']" in message
    assert "unknown ['tags']" in message


@pytest.mark.parametrize("value", [None, {}, [("PER", "a person")]])
def test_entities_must_be_a_non_empty_mapping(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="entities must be a non-empty mapping"):
        NER(model=FAKE_MODEL, entities=value).transform([TEXT])


@pytest.mark.parametrize("label", ["a person", "1PER", "", "PER:X"])
def test_a_label_that_cannot_name_a_tag_is_rejected(
    label: str, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="cannot name an XML tag"):
        NER(model=FAKE_MODEL, entities={label: "a person"}).transform([TEXT])


@pytest.mark.parametrize("description", ["", "  ", None])
def test_an_entity_without_a_description_is_rejected(
    description: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="needs a description"):
        NER(model=FAKE_MODEL, entities={"PER": description}).transform([TEXT])


def test_labels_differing_only_in_case_are_rejected(fake_backend: FakeBackend) -> None:
    """Tag names are matched case-insensitively, so such a pair is ambiguous."""
    with pytest.raises(ValueError, match="differ only in case"):
        NER(
            model=FAKE_MODEL, entities={"PER": "a person", "per": "a persona"}
        ).transform([TEXT])


# --- sklearn compliance ------------------------------------------------------


def test_the_tagger_is_stateless(fake_backend: FakeBackend, tagger: NER) -> None:
    assert not get_tags(tagger).requires_fit
    assert tagger.fit([TEXT]) is tagger
    assert fake_backend.calls == []  # fit only validates params

    fake_backend.chat_script = TAGGED
    assert list(NER(model=FAKE_MODEL, entities=ENTITIES).transform([TEXT])) == [TAGGED]


def test_fit_validates_the_entity_schema(fake_backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="entities must be a non-empty mapping"):
        NER(model=FAKE_MODEL).fit([TEXT])


def test_params_survive_a_clone() -> None:
    tagger = NER(
        model=FAKE_MODEL, entities=ENTITIES, api_key="sk-secret", alias="team_a"
    )

    cloned = clone(tagger)

    assert cloned.get_params() == tagger.get_params()
    assert cloned.entities == ENTITIES
    assert vars(tagger) == tagger.get_params()


def test_the_tagger_has_no_predict_alias() -> None:
    """v1 aliased predict to transform on taggers; v2 does not."""
    assert not hasattr(NER, "predict")


def test_the_tagger_runs_as_a_pipeline_step(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = TAGGED
    pipeline = Pipeline([("tag", NER(model=FAKE_MODEL, entities=ENTITIES, verbose=0))])

    tagged = pipeline.fit_transform([TEXT])

    assert list(tagged) == [TAGGED]
    assert len(fake_backend.calls) == 1  # fitting the step issued none


def test_an_empty_batch_issues_no_requests(
    fake_backend: FakeBackend, tagger: NER
) -> None:
    fake_backend.chat_script = TAGGED
    tagger.transform([TEXT])
    before = tagger.usage_.total
    fake_backend.calls.clear()

    tagged = tagger.transform([])

    assert tagged.shape == (0,)
    assert fake_backend.calls == []
    assert tagger.usage_.last == UsageRecord()
    assert tagger.usage_.total == before


# --- rendering ---------------------------------------------------------------


def test_rendering_highlights_each_entity_and_explains_the_labels() -> None:
    rendered = render_tagged_html([TAGGED], ENTITIES)

    assert ">Tim Cook<" in rendered
    assert ">Apple<" in rendered
    assert 'title="PER: a person"' in rendered
    assert 'title="ORG: an organization"' in rendered
    assert (
        rendered.count("</span>") == 6
    )  # two legend badges, two spans and their labels
    assert " leads " in rendered


def test_rendering_escapes_the_text_it_highlights() -> None:
    rendered = render_tagged_html(
        ["<PER>A & B</PER> wrote <script>x</script>"], ENTITIES
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;x&lt;/script&gt;" in rendered
    assert "A &amp; B" in rendered


def test_rendering_is_deterministic() -> None:
    """v1 picked colors at random, so no two renders of a text ever matched."""
    assert render_tagged_html([TAGGED], ENTITIES) == render_tagged_html(
        [TAGGED], ENTITIES
    )


def test_rendered_html_displays_itself_in_a_notebook() -> None:
    rendered = render_tagged_html([TAGGED], ENTITIES)

    assert rendered._repr_html_() == str(rendered)  # type: ignore[attr-defined]


def test_rendering_separates_the_texts_it_was_given() -> None:
    rendered = render_tagged_html([TAGGED, "<ORG>Apple</ORG> hires."], ENTITIES)

    assert rendered.count("<hr>") == 1
