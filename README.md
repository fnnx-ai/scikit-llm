<div align="center">
  <img alt="logo" src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/41063fcd620d9091662fc6473f9331a7651b4465/scikit-llm.svg" height = "250">
</div>

# Scikit-LLM: Scikit-Learn Meets Large Language Models

Language models as scikit-learn estimators: classification, extraction, scoring,
tagging and embeddings behind `fit`/`predict`/`transform`, across OpenAI,
Anthropic, Google, Ollama and anything LiteLLM reaches.

> **v2 is a rewrite, and no v1 estimator survives it under its old name.** Coming
> from v1? Read **[MIGRATION.md](MIGRATION.md)**, or pin `scikit-llm<2`.

## Installation 💾

```bash
pip install scikit-llm
```

The core install covers OpenAI, Ollama and any OpenAI-compatible endpoint. Other
providers ship as extras: `scikit-llm[anthropic]`, `scikit-llm[google]`,
`scikit-llm[litellm]`, or `scikit-llm[all]`. Python ≥ 3.10.

## Quick start & Documentation 📚

```python
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()  # 30 movie reviews; positive, negative, neutral

clf = ZeroShotClassifier()  # openai::gpt-5.6-sol, key read from OPENAI_API_KEY
clf.fit(None, y)            # zero-shot needs the label set, not the texts
labels = clf.predict(X)
```

Every estimator is importable from the top-level package and takes a
`provider::model_id` string, so `model="anthropic::claude-sonnet-5"` switches
providers. Credentials, the full task catalog, error handling, and saving and
serving `.fnnx` bundles are covered in the
**[documentation](https://skllm.fnnx.ai)**.

## Support us 🤝

You can support the project in the following ways:

- ⭐ Star Scikit-LLM on GitHub (click the star button in the top right corner)
- 💡 Provide your feedback or propose ideas in the [issues](https://github.com/iryna-kondr/scikit-llm/issues) section or [Discord](https://discord.gg/YDAbwuWK7V)
- 📰 Post about Scikit-LLM on LinkedIn or other platforms

## Citation

You can cite Scikit-LLM using the following BibTeX:

```
@software{ScikitLLM,
  author = {Iryna Kondrashchenko and Oleh Kostromin},
  year = {2023},
  publisher = {fnnx.ai},
  address = {Linz, Austria},
  title = {Scikit-LLM: Scikit-Learn Meets Large Language Models},
  url = {https://github.com/iryna-kondr/scikit-llm }
}
```
