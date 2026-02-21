import unittest
from unittest.mock import patch
from skllm.models.vertex.classification.zero_shot import (
    ZeroShotVertexClassifier, 
    MultiLabelZeroShotVertexClassifier
)
from skllm.model_constants import VERTEX_GEMINI_MODEL

class TestZeroShotVertexClassifier(unittest.TestCase):
    
    def test_initialization_default(self):
        """Test if the classifier initializes with the new default Gemini model."""
        clf = ZeroShotVertexClassifier()
        self.assertEqual(clf.model, VERTEX_GEMINI_MODEL)

    @patch("skllm.llm.vertex.mixin.get_completion_chat_gemini")
    def test_single_label_predict(self, mock_gemini):
        """Test single-label fit and predict with mocked Gemini completion."""
        mock_gemini.return_value = '{"label": "positive"}'
        
        X = ["I love this!", "This is bad."]
        y = ["positive", "negative"]
        
        clf = ZeroShotVertexClassifier()
        clf.fit(X, y)
        
        predictions = clf.predict(["I am happy"])
        
        self.assertEqual(predictions[0], "positive")
        mock_gemini.assert_called()
        
        # Verify it uses the correct model
        args, _ = mock_gemini.call_args
        self.assertEqual(args[0], VERTEX_GEMINI_MODEL)

    @patch("skllm.llm.vertex.mixin.get_completion_chat_gemini")
    def test_multi_label_predict(self, mock_gemini):
        """Test multi-label fit and predict with mocked Gemini completion."""
        mock_gemini.return_value = '{"label": ["tech", "science"]}'
        
        X = ["New discovery in AI.", "Space exploration."]
        y = [["tech", "science"], ["science"]]
        
        clf = MultiLabelZeroShotVertexClassifier()
        clf.fit(X, y)
        
        predictions = clf.predict(["AI in space"])
        
        # The mixin pads the result to max_labels (default 5) with empty strings
        self.assertIn("tech", predictions[0])
        self.assertIn("science", predictions[0])
        self.assertEqual(len(predictions[0]), 5) 
        mock_gemini.assert_called()

if __name__ == "__main__":
    unittest.main()
