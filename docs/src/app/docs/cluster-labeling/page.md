---
title: Cluster labeling
---

`ClusterLabeler` names the clusters another model produced. It clusters nothing itself: `fit(X, y)` takes the texts and the cluster ids a clusterer already assigned to them, samples a few texts out of each cluster, and asks the model what topic holds that group together. Naming costs one request per cluster, not one per text.

```python
from sklearn.cluster import KMeans

from skllm import ClusterLabeler, Vectorizer
from skllm.datasets import get_classification_dataset

X, _ = get_classification_dataset()  # 30 movie reviews

embeddings = Vectorizer().fit_transform(X)
clusters = KMeans(n_clusters=3, random_state=0).fit_predict(embeddings)

labeler = ClusterLabeler(n_exemplars=8, random_state=0)
names = labeler.fit_transform(X, clusters)

labeler.names_  # {0: 'enthusiastic praise', 1: 'mixed verdicts', 2: 'harsh criticism'}
names[:2]       # array(['enthusiastic praise', 'enthusiastic praise'], dtype=object)
```

Any clusterer works, since the labeler only ever sees ids. The embeddings above come from [`Vectorizer`](/docs/vectorization), but TF-IDF or any other representation ends in the same integer labels, and those labels alongside the texts are all `fit` needs.

`X` is a sequence of strings and `y` a one-dimensional sequence of integer cluster ids — a list, a numpy array or a pandas `Series`, as a clusterer's `fit_predict` hands them back. The two must be the same length, and floats, strings and booleans are rejected with `TypeError` rather than coerced. Which model does the naming, and where its credentials come from, follows the rules in [Models and providers](/docs/models-and-providers) and [Credentials and configuration](/docs/credentials-and-configuration).

## How a cluster is named

Each cluster contributes up to `n_exemplars` of its texts, ten by default, drawn at random and shown to the model in the order they appeared in `X`. A cluster holding fewer texts than that shows all of them. `random_state` seeds the draw with the usual scikit-learn semantics, and clusters are drawn from in id order, so one seed selects the same exemplars however the clusters happened to be encountered.

The exemplars go into a prompt asking for the topic that holds them together, in a few words, specific enough to tell this cluster apart from other groups in the same corpus. The answer is used verbatim once surrounding whitespace is trimmed. The template takes a single placeholder, `{texts}`. See [Customizing prompts](/docs/customizing-prompts) for replacing it, which is where a domain vocabulary or a naming convention belongs.

Raising `n_exemplars` gives the model more of the cluster to generalize from, in exchange for a longer prompt. Since exemplars are sampled rather than picked for centrality, a cluster that is not internally coherent will produce a name that fits only part of it.

{% callout title="The noise id" %}
Density-based clusterers such as DBSCAN and HDBSCAN assign `-1` to the points they grouped with nothing. The id stands for no cluster, so `ClusterLabeler` never names it: those texts reach no model, and `-1` maps to `None` at every later lookup. A clustering of nothing but noise names nothing and issues no request.
{% /callout %}

## Names and lookups

`fit` leaves the mapping in `names_`, a plain `dict[int, str]` from cluster id to name. Because it is an ordinary dictionary, a name that reads wrong can be replaced by hand before anything is transformed.

`transform(y)` takes cluster ids, not texts, and returns an object array of the same length holding each id's name, with `None` wherever the noise id appeared. It issues no request at all: the names were decided at `fit`, and the lookup is a dictionary read. It needs no credentials, and it reports an empty usage record without touching the cumulative total left by the naming run. An id that was not present at `fit` has no name and raises `ValueError` listing the ids that do.

```python
labeler.transform([0, 1, -1])
# array(['enthusiastic praise', 'mixed verdicts', None], dtype=object)
```

`fit_transform(X, y)` departs from the inherited transformer contract: it returns `self.fit(X, y).transform(y)`, transforming the cluster ids rather than the texts, since the texts are what the clusters were named *from* and the clustering is what was named. For the same reason the labeler does not slot into a `Pipeline` behind a clusterer the way an ordinary transformer would — the step before it would have to yield cluster ids as its output.

## When a cluster cannot be named

Naming is all-or-nothing. A cluster whose request failed, or that the model answered for with an empty name, aborts the whole `fit` and raises, whatever `on_error` says. The estimator is left unfitted, so a later `transform` raises `NotFittedError` rather than reading a half-built mapping. The `on_error="fill"` policy, which elsewhere lets a batch finish with placeholder values, cannot apply here: every later `transform` reads this mapping, and a filled-in name would be indistinguishable from one the model gave. [Batch execution and error handling](/docs/batch-execution-and-error-handling) describes the policies and which estimators override them.

## Serving a fitted labeler

Because every request happens at `fit`, a fitted labeler needs no credential to answer. Its `llm_requirements()` is empty, and a bundle saved from it declares no key to supply at serving time: it carries the name mapping and answers lookups. See [Saving and loading](/docs/saving-and-loading).
