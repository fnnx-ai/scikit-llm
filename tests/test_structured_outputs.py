
import unittest
from pydantic import BaseModel
from skllm.llm.gpt.clients.openai.completion import get_parsed_completion
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import skllm.llm.gpt.clients.openai.completion as completion_mod

class DummyCompletions:
    def __init__(self, model_cls):
        self._model_cls = model_cls

    def parse(self, *, model, messages, response_format, temperature):
        # response_format is the Pydantic model class (TestEvent)
        fake = self._model_cls(
            event_name="science fair",
            date="Friday",
            attendees=["Alice", "Bob"],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=fake))]
        )

    def create(self, *, temperature, messages, **kwargs):
        # if you ever test get_chat_completion
        return {"id": "dummy", "choices": []}

class DummyClient:
    def __init__(self, model_cls):
        self.chat = SimpleNamespace(completions=DummyCompletions(model_cls))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions(model_cls)))

class OpenAITestCase(unittest.TestCase):
    def setUp(self):
        self.patcher1 = patch.object(
            completion_mod,
            "set_credentials",
            lambda key, org: DummyClient(TestEvent)
        )
        self.patcher2 = patch.object(
            completion_mod,
            "set_azure_credentials",
            lambda key, org: DummyClient(TestEvent)
        )
        self.patcher1.start()
        self.patcher2.start()

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()


class TestEvent(BaseModel):
    event_name: str
    date: str
    attendees: list[str]

class TestOpenAIStructuredOutput(OpenAITestCase):
    def test_openai_structured_output(self):
        """Test that structured outputs are properly parsed into Pydantic models."""
        messages = [
            {"role": "system", "content": "Extract event information in JSON format"},
            {"role": "user", "content": "Alice and Bob are attending the science fair on Friday"}
        ]
        
        # Test successful parsing
        result = get_parsed_completion(
            messages=messages,
            output_model=TestEvent,
            key="dummy_value",  # Replace with actual key
            org="dummy_value",   # Replace with actual org
            model="gpt-4o-mini"
        )
        
        # Validate the result structure
        self.assertIsInstance(result, TestEvent)
        self.assertIsInstance(result.event_name, str)
        self.assertGreater(len(result.event_name), 0)
        self.assertIsInstance(result.date, str)
        self.assertGreater(len(result.date), 0)
        self.assertIsInstance(result.attendees, list)
        self.assertGreaterEqual(len(result.attendees), 2)  # Should have at least Alice and Bob
        self.assertTrue(all(isinstance(name, str) for name in result.attendees))

if __name__ == '__main__':
    unittest.main()
