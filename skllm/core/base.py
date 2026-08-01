"""Base estimators: the sklearn contract, the request path and the error policy.

Estimators hold declarative configuration and fitted state only. Backends,
credentials and settings are resolved into call-locals on every request, which
is what makes a fitted instance safe to share across threads and safe to
serialize without leaking connection config.
"""

from __future__ import annotations

import inspect
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from string import Formatter
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypeVar

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skllm.backends import (
    Backend,
    Connection,
    Message,
    ModelRef,
    Response,
    get_backend,
    parse_model,
)
from skllm.config import MissingCredentialError, resolve_model_params, validate_alias
from skllm.core.engine import map_concurrently
from skllm.core.errors import ON_ERROR_POLICIES, SampleError
from skllm.core.telemetry import llm_span, set_usage
from skllm.core.usage import UsageRecord, UsageReport, collect_usage, record_request

if TYPE_CHECKING:
    from typing import Self

    from sklearn.utils import Tags

__all__ = [
    "CONNECTION_PARAMS",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MAX_CONCURRENCY",
    "BaseLLMEstimator",
    "BasePromptEstimator",
    "Requirement",
    "StatelessMixin",
    "as_texts",
    "check_count",
]

DEFAULT_CHAT_MODEL = "openai::gpt-5.6-sol"
DEFAULT_EMBEDDING_MODEL = "openai::text-embedding-3-small"
DEFAULT_MAX_CONCURRENCY = 8

#: Params saying where and as whom the model runs; excluded from every
#: serialized form and re-supplied through the config layers at call time.
CONNECTION_PARAMS = ("api_key", "base_url")

_T = TypeVar("_T")
_R = TypeVar("_R")

#: Guards the read-modify-write of every estimator's bookkeeping. Estimators keep
#: no lock of their own: it would have to be restored on unpickling, and the
#: critical section is a handful of integer additions.
_BOOKKEEPING_LOCK = threading.Lock()

#: Modules whose frames sit between an estimator's public method and the code
#: that called it: skllm's own, plus the sklearn shim wrapping every
#: ``transform`` for the set_output API.
_PASS_THROUGH_MODULES = ("skllm.", "sklearn.utils._set_output")

#: Failures the ``"fill"`` policy must not absorb: they are configuration errors
#: that every sample of the batch hits identically, so filling them would report
#: a missing key as N indistinguishable prediction failures.
_UNFILLABLE = (MissingCredentialError,)


class Requirement(NamedTuple):
    """One credential an estimator needs: a provider, optionally routed by alias."""

    provider: str
    alias: str | None = None


class BaseLLMEstimator(BaseEstimator):
    """Shared behavior for every estimator that answers through an LLM provider.

    Subclasses declare their own ``__init__`` (sklearn reads its signature for
    ``get_params``), implement ``fit`` and their task method, and issue requests
    through :meth:`_chat`, :meth:`_structured` or :meth:`_embed` from inside a
    :meth:`_run_batch` call.
    """

    #: Whether one failed sample aborts the batch whatever ``on_error`` says.
    #: Set where the answers are not per-sample predictions but material other
    #: models consume — embeddings, cluster names — so that a filled sample
    #: would silently poison everything downstream instead of showing up.
    _all_or_nothing: ClassVar[bool] = False

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        self.model = model
        self.model_params = model_params
        self.api_key = api_key
        self.base_url = base_url
        self.alias = alias
        self.on_error = on_error
        self.max_concurrency = max_concurrency
        self.verbose = verbose

    @property
    def usage_(self) -> UsageReport:
        """What the most recent operation cost, and everything since construction."""
        return getattr(self, "_usage", UsageReport())

    @property
    def errors_(self) -> tuple[SampleError, ...]:
        """The samples the most recent operation could not answer."""
        return getattr(self, "_errors", ())

    def llm_requirements(self) -> tuple[Requirement, ...]:
        """Every provider/alias pair this estimator needs a credential for to serve.

        Covers skllm-native nesting — components supplied as params and those
        built during ``fit`` alike — so callers walking foreign structures (a
        pipeline, a wrapper) never need to know skllm's internals.
        """
        found = list(self._own_requirements())
        for component in self._nested_estimators():
            found.extend(component.llm_requirements())
        return tuple(dict.fromkeys(found))

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.non_deterministic = True
        return tags

    def __getstate__(self) -> dict[str, Any]:
        """Drop connection config, so not even a raw pickle carries credentials."""
        state = dict(super().__getstate__())
        return {**state, **{name: None for name in CONNECTION_PARAMS if name in state}}

    def _own_requirements(self) -> tuple[Requirement, ...]:
        """This estimator's own need; empty when it calls no LLM outside ``fit``."""
        return (Requirement(parse_model(self.model).provider, self.alias),)

    def _nested_estimators(self) -> Iterator[BaseLLMEstimator]:
        """skllm estimators held as params or as fitted state, containers included."""
        for value in list(vars(self).values()):
            yield from _nested(value)

    def _check_params(self) -> None:
        """Validate constructor params and resolve the backend, at fit/first use."""
        self._check_batch_params()
        if self.alias is not None:
            validate_alias(self.alias)
        if self.model_params is not None and not isinstance(self.model_params, Mapping):
            raise TypeError(
                "model_params must be a mapping of provider request params, got "
                f"{type(self.model_params).__name__}."
            )
        self._backend()

    def _check_batch_params(self) -> None:
        """Validate how a batch is run, which even a model-less ensemble does."""
        if self.on_error not in ON_ERROR_POLICIES:
            raise ValueError(
                f"Invalid on_error={self.on_error!r}: expected one of "
                f"{list(ON_ERROR_POLICIES)}."
            )
        check_count("max_concurrency", self.max_concurrency, required=True)

    def _check_ready(self) -> None:
        """Guard predict/transform: fitted where required, valid params either way."""
        check_is_fitted(self)
        self._check_params()

    def _backend(self) -> tuple[Backend, ModelRef]:
        ref = parse_model(self.model)
        return get_backend(ref.provider), ref

    def _connection(self) -> Connection:
        return Connection(
            api_key=self.api_key, base_url=self.base_url, alias=self.alias
        )

    def _request_params(self) -> dict[str, Any]:
        """Stored ``model_params`` with any alias-targeted override applied on top."""
        return resolve_model_params(self.alias, self.model_params)

    def _chat(self, messages: Sequence[Message], *, deterministic: bool = True) -> str:
        backend, ref = self._backend()
        return self._issue(
            "chat",
            ref,
            lambda: backend.chat(
                ref.model_id,
                messages,
                params=self._request_params(),
                connection=self._connection(),
                deterministic=deterministic,
            ),
        )

    def _structured(
        self,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> Any:
        backend, ref = self._backend()
        return self._issue(
            "chat",
            ref,
            lambda: backend.structured(
                ref.model_id,
                messages,
                schema,
                params=self._request_params(),
                connection=self._connection(),
                deterministic=deterministic,
            ),
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        backend, ref = self._backend()
        return self._issue(
            "embeddings",
            ref,
            lambda: backend.embed(
                ref.model_id,
                texts,
                params=self._request_params(),
                connection=self._connection(),
            ),
        )

    def _issue(
        self, operation: str, ref: ModelRef, call: Callable[[], Response[_T]]
    ) -> _T:
        """Issue one request, accounting for what it cost."""
        with llm_span(operation, ref.provider, ref.model_id) as span:
            response = call()
            set_usage(span, response.usage, record_request(response.usage, self.model))
        return response.content

    def _run_batch(
        self,
        items: Sequence[_T],
        call: Callable[[_T], _R],
        *,
        fill: Callable[[], _R],
    ) -> list[_R]:
        """Answer every item once, concurrently, under this estimator's error policy.

        ``fill`` is a factory rather than a value so that each failed sample gets
        its own object: a shared mutable fill would alias across samples.
        """
        answers = self._run_repeated(items, call, repeats=1)
        return [answer[0] if answer else fill() for answer in answers]

    def _run_repeated(
        self, items: Sequence[_T], call: Callable[[_T], _R], *, repeats: int
    ) -> list[list[_R]]:
        """Answer every item ``repeats`` times, returning the answers that arrived.

        All attempts of all items share one bounded pool, so ``max_concurrency``
        stays the cap on in-flight requests however many attempts each item takes.
        An item whose every attempt failed answers with an empty list and is
        reported as a failed sample; partial failures are left to the caller to
        aggregate over.
        """
        attempts = [item for item in items for _ in range(repeats)]
        answers: list[list[_R]] = []
        errors: list[SampleError] = []
        raising = self._all_or_nothing or self.on_error == "raise"
        with collect_usage() as usage:
            try:
                outcomes = map_concurrently(
                    call,
                    attempts,
                    max_concurrency=self.max_concurrency,
                    verbose=self.verbose,
                    description=type(self).__name__,
                    # One attempt per item makes any failure a sample failure,
                    # so the batch can abort the moment one arrives. Repeated
                    # attempts tolerate individual failures, and whether an item
                    # lost every one of them is only known once they are all in.
                    fail_fast=raising and repeats == 1,
                )
                for index, group in enumerate(_grouped(outcomes, repeats)):
                    for outcome in group:
                        if isinstance(outcome, _UNFILLABLE):
                            raise outcome
                    answers.append(
                        [item for item in group if not isinstance(item, Exception)]
                    )
                    if not answers[-1]:
                        if raising:
                            raise _first_failure(group)
                        errors.append(SampleError(index, _reason(group, repeats)))
            except BaseException:
                # An aborted batch still paid for the requests it did complete,
                # and a report left describing the previous operation would be
                # read as describing this one.
                self._record_usage(usage.record, ())
                raise

        usage.add(UsageRecord(failures=len(errors)))
        self._record_usage(usage.record, tuple(errors))
        if errors:
            self._warn_filled(len(errors), len(answers))
        return answers

    def _warn_filled(self, filled: int, total: int) -> None:
        """Say that a batch holds fill values, at the caller's own call site."""
        warnings.warn(
            f"{type(self).__name__}: {filled} of {total} samples failed and were "
            "filled; see the errors_ attribute for the reasons.",
            stacklevel=_caller_stacklevel(),
        )

    def _record_usage(
        self, record: UsageRecord, errors: tuple[SampleError, ...]
    ) -> None:
        with _BOOKKEEPING_LOCK:
            self._usage = UsageReport(last=record, total=self.usage_.total + record)
            self._errors = errors


class StatelessMixin(BaseLLMEstimator):
    """Mixed in ahead of the task base by estimators that need no fitted state.

    Their ``fit`` only validates params and their task method works without one,
    which sklearn expresses as the ``requires_fit`` tag.
    """

    def fit(self, X: Any = None, y: Any = None) -> Self:
        self._check_params()
        return self

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class BasePromptEstimator(BaseLLMEstimator):
    """A task expressed as a prompt template the user may override."""

    #: The placeholders a template must use — the task's documented set.
    prompt_placeholders: ClassVar[frozenset[str]] = frozenset()
    #: The template used when none is supplied.
    default_prompt_template: ClassVar[str] = ""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.prompt_template = prompt_template

    def _check_params(self) -> None:
        super()._check_params()
        if self.prompt_template is not None:
            self._check_prompt_template(self.prompt_template)

    def _prompt(self, **values: Any) -> str:
        return (self.prompt_template or self.default_prompt_template).format(**values)

    @classmethod
    def _check_prompt_template(cls, template: str) -> None:
        if not isinstance(template, str):
            raise TypeError(
                f"prompt_template must be a string, got {type(template).__name__}."
            )
        try:
            used = _placeholders(template)
        except ValueError as exc:
            raise ValueError(
                f"Invalid prompt_template for {cls.__name__}: {exc}"
            ) from exc

        problems = []
        if missing := sorted(cls.prompt_placeholders - used):
            problems.append(f"missing {missing}")
        if unknown := sorted(used - cls.prompt_placeholders):
            problems.append(f"unknown {unknown}")
        if problems:
            raise ValueError(
                f"Invalid prompt_template for {cls.__name__}: "
                f"{' and '.join(problems)} placeholders. The template must use "
                f"exactly {sorted(cls.prompt_placeholders)}."
            )


def as_texts(X: Any) -> list[str]:
    """Normalize a batch of texts (list, numpy array or pandas Series) to a list."""
    if X is None or isinstance(X, str):
        raise TypeError(
            f"X must be a sequence of strings, got {type(X).__name__}; pass a list, "
            "a numpy array or a pandas Series of texts."
        )
    if hasattr(X, "columns"):  # a DataFrame: iterating it would yield column names
        raise TypeError(
            "X must be a one-dimensional sequence of strings; select the text "
            "column first, e.g. df['text']."
        )
    dimensions = getattr(X, "ndim", 1)
    if dimensions != 1:
        raise TypeError(
            f"X must be one-dimensional, got a {dimensions}-dimensional array."
        )

    texts = X.tolist() if hasattr(X, "tolist") else list(X)
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"X[{index}] is a {type(text).__name__}, but every sample must be "
                "a string."
            )
    return texts


def check_count(name: str, value: Any, *, required: bool = False) -> None:
    """Validate a count param: a positive integer, or ``None`` when optional."""
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = "" if required else " or None"
        raise ValueError(f"{name} must be a positive integer{suffix}, got {value!r}.")


def _caller_stacklevel() -> int:
    """How many frames up the code that ran the batch sits, for a warning.

    A fixed level would point at the wrong file: sklearn wraps every
    ``transform`` for its set_output API, so a transformer's batch runs one
    frame deeper than a predictor's.
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None  # the warning's own site
    level = 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        _PASS_THROUGH_MODULES
    ):
        frame = frame.f_back
        level += 1
    return level


def _grouped(outcomes: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    """Re-split the flattened attempts into one group per item."""
    for start in range(0, len(outcomes), size):
        yield outcomes[start : start + size]


def _first_failure(group: Sequence[Any]) -> Exception:
    """The failure an item reports; only called when every attempt of it failed."""
    return next(item for item in group if isinstance(item, Exception))


def _reason(group: Sequence[Any], repeats: int) -> str:
    """Why an item has no answer, from the first of its failed attempts."""
    failure = _first_failure(group)
    reason = f"{type(failure).__name__}: {failure}"
    return reason if repeats == 1 else f"all {repeats} attempts failed; first: {reason}"


def _nested(value: Any) -> Iterator[BaseLLMEstimator]:
    if isinstance(value, BaseLLMEstimator):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _nested(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _nested(item)


def _placeholders(template: str) -> set[str]:
    """The field names a template refers to, nested format specs included.

    Index and attribute access is ignored. A positional field (``{}``, ``{0}``)
    has no name to check against the task's set and would only fail inside
    ``format`` — once per sample, as a filled failure — so it is rejected here.
    """
    names: set[str] = set()
    for _, field, spec, _ in Formatter().parse(template):
        nested = (name for _, name, _, _ in Formatter().parse(spec or ""))
        for name in (field, *nested):
            if name is None:
                continue
            root = name.partition("[")[0].partition(".")[0]
            if not root or root.isdigit():
                raise ValueError(f"placeholder {{{name}}} is positional, not named")
            names.add(root)
    return names
