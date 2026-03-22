<div align="center">
  <img alt="logo" src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/41063fcd620d9091662fc6473f9331a7651b4465/scikit-llm.svg" height = "250">
</div>

# Scikit-LLM: Scikit-Learn Meets Large Language Models

Seamlessly integrate powerful language models like ChatGPT, Claude, and MiniMax into scikit-learn for enhanced text analysis tasks.

## Installation 💾

```bash
pip install scikit-llm
```

## Support us 🤝

You can support the project in the following ways:

- ⭐ Star Scikit-LLM on GitHub (click the star button in the top right corner)
- 💡 Provide your feedback or propose ideas in the [issues](https://github.com/iryna-kondr/scikit-llm/issues) section or [Discord](https://discord.gg/YDAbwuWK7V)
- 📰 Post about Scikit-LLM on LinkedIn or other platforms
- 🔗 Check out our other projects: <a href="https://github.com/beastbyteai/agent_dingo">Dingo</a>, <a href="https://github.com/beastbyteai/agent_dingo">Falcon</a>

<br>
<a href="https://github.com/OKUA1/agent_dingo">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/ding_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" height = "65">
</picture>
</a> <br><br>
<a href="https://github.com/OKUA1/falcon">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" height = "65">
</picture>
</a>

## Quick Start & Documentation 📚

Quick start example of zero-shot text classification using GPT:

```python
# Import the necessary modules
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# Configure the credentials
SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")

# Load a demo dataset
X, y = get_classification_dataset() # labels: positive, negative, neutral

# Initialize the model and make the predictions
clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(X,y)
clf.predict(X)
```

### Using MiniMax

```python
from skllm.config import SKLLMConfig
from skllm.models.minimax.classification.zero_shot import ZeroShotMiniMaxClassifier

# Configure the credentials
SKLLMConfig.set_minimax_key("<YOUR_MINIMAX_API_KEY>")

# Initialize the model and make predictions
clf = ZeroShotMiniMaxClassifier(model="MiniMax-M2.7")
clf.fit(X, y)
clf.predict(X)
```

Available MiniMax models: `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`.

For more information please refer to the **[documentation](https://skllm.beastbyte.ai)**.

## Citation

You can cite Scikit-LLM using the following BibTeX:

```
@software{ScikitLLM,
  author = {Iryna Kondrashchenko and Oleh Kostromin},
  year = {2023},
  publisher = {beastbyte.ai},
  address = {Linz, Austria},
  title = {Scikit-LLM: Scikit-Learn Meets Large Language Models},
  url = {https://github.com/iryna-kondr/scikit-llm }
}
```
