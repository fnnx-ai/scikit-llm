"""Integration tests for MiniMax provider.

These tests require a valid MINIMAX_API_KEY environment variable.
They are skipped when the key is not available.
"""
import os
import unittest

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_KEY, SKIP_REASON)
class TestMiniMaxCompletionIntegration(unittest.TestCase):
    def test_basic_completion(self):
        from skllm.llm.minimax.completion import get_chat_completion

        response = get_chat_completion(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            key=MINIMAX_KEY,
            model="MiniMax-M2.5-highspeed",
            max_tokens=500,
        )
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertTrue(len(content) > 0)

    def test_completion_with_system_message(self):
        from skllm.llm.minimax.completion import get_chat_completion

        response = get_chat_completion(
            messages=[{"role": "user", "content": "What are you?"}],
            key=MINIMAX_KEY,
            model="MiniMax-M2.5-highspeed",
            system="You are a friendly bot. Respond in one sentence.",
            max_tokens=500,
        )
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertTrue(len(content) > 0)

    def test_completion_json_response(self):
        import json
        from skllm.llm.minimax.completion import get_chat_completion
        from skllm.utils import find_json_in_string

        response = get_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": 'Classify the sentiment of "I love this product" as positive, negative, or neutral. Return JSON with a "label" key.',
                }
            ],
            key=MINIMAX_KEY,
            model="MiniMax-M2.5-highspeed",
            json_response=True,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        # The model may wrap JSON in markdown code fences
        json_str = find_json_in_string(content)
        data = json.loads(json_str)
        self.assertIn("label", data)


@unittest.skipUnless(MINIMAX_KEY, SKIP_REASON)
class TestMiniMaxMixinIntegration(unittest.TestCase):
    def test_text_completion_mixin(self):
        from skllm.llm.minimax.mixin import MiniMaxTextCompletionMixin

        mixin = MiniMaxTextCompletionMixin()
        mixin._set_keys(MINIMAX_KEY)

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.5-highspeed",
            messages="Say 'test' and nothing else.",
        )
        result = mixin._convert_completion_to_str(completion)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_classifier_mixin(self):
        from skllm.llm.minimax.mixin import MiniMaxClassifierMixin

        mixin = MiniMaxClassifierMixin()
        mixin._set_keys(MINIMAX_KEY)

        completion = mixin._get_chat_completion(
            model="MiniMax-M2.5-highspeed",
            messages='Classify "I love this!" as positive, negative, or neutral.',
            system_message="You are a text classifier. Return a JSON object with a single key 'label'.",
        )
        label = mixin._extract_out_label(completion)
        self.assertIsInstance(label, str)


@unittest.skipUnless(MINIMAX_KEY, SKIP_REASON)
class TestMiniMaxConfigIntegration(unittest.TestCase):
    def test_config_set_get_key(self):
        from skllm.config import SKLLMConfig

        SKLLMConfig.set_minimax_key(MINIMAX_KEY)
        self.assertEqual(SKLLMConfig.get_minimax_key(), MINIMAX_KEY)


if __name__ == "__main__":
    unittest.main()
