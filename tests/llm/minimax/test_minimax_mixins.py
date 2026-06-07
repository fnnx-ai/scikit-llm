import unittest
from unittest.mock import patch, MagicMock
import json
from skllm.llm.minimax.mixin import (
    MiniMaxMixin,
    MiniMaxTextCompletionMixin,
    MiniMaxClassifierMixin,
)


class TestMiniMaxMixin(unittest.TestCase):
    def test_set_and_get_key(self):
        mixin = MiniMaxMixin()
        mixin._set_keys("test_minimax_key")
        self.assertEqual(mixin._get_minimax_key(), "test_minimax_key")

    def test_get_key_from_config(self):
        mixin = MiniMaxMixin()
        mixin._set_keys(None)
        with patch("skllm.llm.minimax.mixin._Config") as mock_config:
            mock_config.get_minimax_key.return_value = "config_key"
            self.assertEqual(mixin._get_minimax_key(), "config_key")

    def test_get_key_raises_when_not_found(self):
        mixin = MiniMaxMixin()
        mixin._set_keys(None)
        with patch("skllm.llm.minimax.mixin._Config") as mock_config:
            mock_config.get_minimax_key.return_value = None
            with self.assertRaises(RuntimeError):
                mixin._get_minimax_key()


class TestMiniMaxTextCompletionMixin(unittest.TestCase):
    @patch("skllm.llm.minimax.mixin.get_chat_completion")
    def test_chat_completion_with_string_message(self, mock_get_chat_completion):
        mixin = MiniMaxTextCompletionMixin()
        mixin._set_keys("test_key")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "test response"
        mock_get_chat_completion.return_value = mock_completion

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.7",
            messages="Hello",
            system_message="Test system",
        )

        self.assertEqual(
            mixin._convert_completion_to_str(completion),
            "test response",
        )
        mock_get_chat_completion.assert_called_once()

    @patch("skllm.llm.minimax.mixin.get_chat_completion")
    def test_chat_completion_with_list_messages(self, mock_get_chat_completion):
        mixin = MiniMaxTextCompletionMixin()
        mixin._set_keys("test_key")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "response"
        mock_get_chat_completion.return_value = mock_completion

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(
            mixin._convert_completion_to_str(completion),
            "response",
        )

    @patch("skllm.llm.minimax.mixin.get_chat_completion")
    def test_convert_completion_dict_format(self, mock_get_chat_completion):
        mixin = MiniMaxTextCompletionMixin()
        mixin._set_keys("test_key")

        completion = {
            "choices": [{"message": {"content": "dict response"}}]
        }
        self.assertEqual(
            mixin._convert_completion_to_str(completion),
            "dict response",
        )


class TestMiniMaxClassifierMixin(unittest.TestCase):
    @patch("skllm.llm.minimax.mixin.get_chat_completion")
    def test_extract_out_label_with_valid_json(self, mock_get_chat_completion):
        mixin = MiniMaxClassifierMixin()
        mixin._set_keys("test_key")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"label":"positive"}'
        mock_get_chat_completion.return_value = mock_completion

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.7",
            messages="Classify this text",
            system_message="You are a classifier",
        )
        self.assertEqual(mixin._extract_out_label(completion), "positive")

    @patch("skllm.llm.minimax.mixin.get_chat_completion")
    def test_extract_out_label_with_invalid_json(self, mock_get_chat_completion):
        mixin = MiniMaxClassifierMixin()
        mixin._set_keys("test_key")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "not json"
        mock_get_chat_completion.return_value = mock_completion

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.7",
            messages="test",
        )
        self.assertEqual(mixin._extract_out_label(completion), "")

    def test_prefer_json_output_is_true(self):
        mixin = MiniMaxClassifierMixin()
        self.assertTrue(mixin._prefer_json_output)


class TestMiniMaxCompletion(unittest.TestCase):
    @patch("skllm.llm.minimax.completion.set_credentials")
    def test_get_chat_completion_validates_messages(self, mock_creds):
        from skllm.llm.minimax.completion import get_chat_completion

        # @retry wraps exceptions in RuntimeError after max_retries
        with self.assertRaises(RuntimeError):
            get_chat_completion(messages=[], key="test")

        with self.assertRaises(RuntimeError):
            get_chat_completion(messages="not a list", key="test")

    @patch("skllm.llm.minimax.completion.set_credentials")
    def test_get_chat_completion_clamps_temperature(self, mock_creds):
        mock_client = MagicMock()
        mock_creds.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        from skllm.llm.minimax.completion import get_chat_completion

        get_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            key="test",
            temperature=2.0,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertLessEqual(call_kwargs["temperature"], 1.0)

    @patch("skllm.llm.minimax.completion.set_credentials")
    def test_get_chat_completion_with_system_message(self, mock_creds):
        mock_client = MagicMock()
        mock_creds.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        from skllm.llm.minimax.completion import get_chat_completion

        get_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            key="test",
            system="You are helpful.",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")

    @patch("skllm.llm.minimax.completion.set_credentials")
    def test_get_chat_completion_json_response(self, mock_creds):
        mock_client = MagicMock()
        mock_creds.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        from skllm.llm.minimax.completion import get_chat_completion

        get_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            key="test",
            json_response=True,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(
            call_kwargs["response_format"], {"type": "json_object"}
        )


class TestMiniMaxStripThinkTags(unittest.TestCase):
    def test_strip_think_tags(self):
        from skllm.llm.minimax.completion import _strip_think_tags

        text = '<think>\nSome reasoning here\n</think>\n{"label": "positive"}'
        result = _strip_think_tags(text)
        self.assertEqual(result, '{"label": "positive"}')

    def test_strip_think_tags_no_tags(self):
        from skllm.llm.minimax.completion import _strip_think_tags

        text = '{"label": "positive"}'
        result = _strip_think_tags(text)
        self.assertEqual(result, '{"label": "positive"}')

    def test_strip_think_tags_empty(self):
        from skllm.llm.minimax.completion import _strip_think_tags

        text = "<think></think>"
        result = _strip_think_tags(text)
        self.assertEqual(result, "")


class TestMiniMaxCredentials(unittest.TestCase):
    @patch("skllm.llm.minimax.credentials.OpenAI")
    def test_set_credentials_returns_client(self, mock_openai):
        from skllm.llm.minimax.credentials import set_credentials, MINIMAX_BASE_URL

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        client = set_credentials("test_key")
        mock_openai.assert_called_once_with(
            api_key="test_key", base_url=MINIMAX_BASE_URL
        )
        self.assertEqual(client, mock_client)


class TestMiniMaxConfig(unittest.TestCase):
    def test_set_and_get_minimax_key(self):
        from skllm.config import SKLLMConfig
        import os

        SKLLMConfig.set_minimax_key("test_minimax_key_123")
        self.assertEqual(
            SKLLMConfig.get_minimax_key(), "test_minimax_key_123"
        )
        # Clean up
        os.environ.pop("SKLLM_CONFIG_MINIMAX_KEY", None)

    def test_get_minimax_key_returns_none_when_not_set(self):
        from skllm.config import SKLLMConfig
        import os

        os.environ.pop("SKLLM_CONFIG_MINIMAX_KEY", None)
        self.assertIsNone(SKLLMConfig.get_minimax_key())


class TestMiniMaxModelConstants(unittest.TestCase):
    def test_minimax_model_constant(self):
        from skllm.model_constants import MINIMAX_MODEL

        self.assertEqual(MINIMAX_MODEL, "MiniMax-M3")


class TestMiniMaxModels(unittest.TestCase):
    def test_zero_shot_classifier_init(self):
        from skllm.models.minimax.classification.zero_shot import (
            ZeroShotMiniMaxClassifier,
        )

        clf = ZeroShotMiniMaxClassifier(key="test_key")
        self.assertEqual(clf.model, "MiniMax-M3")
        self.assertEqual(clf.key, "test_key")

    def test_cot_classifier_init(self):
        from skllm.models.minimax.classification.zero_shot import (
            CoTMiniMaxClassifier,
        )

        clf = CoTMiniMaxClassifier(key="test_key", model="MiniMax-M2.7")
        self.assertEqual(clf.model, "MiniMax-M2.7")

    def test_multi_label_classifier_init(self):
        from skllm.models.minimax.classification.zero_shot import (
            MultiLabelZeroShotMiniMaxClassifier,
        )

        clf = MultiLabelZeroShotMiniMaxClassifier(key="test_key", max_labels=3)
        self.assertEqual(clf.max_labels, 3)

    def test_few_shot_classifier_init(self):
        from skllm.models.minimax.classification.few_shot import (
            FewShotMiniMaxClassifier,
        )

        clf = FewShotMiniMaxClassifier(key="test_key")
        self.assertEqual(clf.model, "MiniMax-M3")

    def test_summarizer_init(self):
        from skllm.models.minimax.text2text.summarization import MiniMaxSummarizer

        s = MiniMaxSummarizer(key="test_key", max_words=20)
        self.assertEqual(s.model, "MiniMax-M3")
        self.assertEqual(s.max_words, 20)

    def test_translator_init(self):
        from skllm.models.minimax.text2text.translation import MiniMaxTranslator

        t = MiniMaxTranslator(key="test_key", output_language="French")
        self.assertEqual(t.output_language, "French")

    def test_ner_init(self):
        from skllm.models.minimax.tagging.ner import MiniMaxExplainableNER

        entities = {"PERSON": "A person's name", "ORG": "An organization"}
        ner = MiniMaxExplainableNER(entities=entities, key="test_key")
        self.assertEqual(ner.entities, entities)
        self.assertEqual(ner.model, "MiniMax-M3")


if __name__ == "__main__":
    unittest.main()
