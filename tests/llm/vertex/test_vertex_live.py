import unittest
import os
from skllm.models.vertex.classification.zero_shot import (
    ZeroShotVertexClassifier,
    MultiLabelZeroShotVertexClassifier
)
from skllm.config import SKLLMConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Run this with:
# SKLLM_RUN_LIVE_TESTS=True GOOGLE_CLOUD_PROJECT=your-project uv run pytest tests/llm/vertex/test_vertex_live.py
# Or use a .env file in project root

@unittest.skipIf(os.environ.get("SKLLM_RUN_LIVE_TESTS") != "True", "Skipping live API test")
class TestVertexLive(unittest.TestCase):
    """
    Live tests for Vertex AI Gemini.
    """

    def setUp(self):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            SKLLMConfig.set_google_project(project)
    
    def test_zero_shot_predict_live(self):
        """Test single-label zero-shot classification with real Gemini API."""
        X = ["This is a fantastic product!"]
        y = ["positive", "negative"]
        
        clf = ZeroShotVertexClassifier() # Uses default Gemini 2.5 flash
        clf.fit(None, y) 
        
        labels = clf.predict(X)
        self.assertEqual(len(labels), 1)
        self.assertIn(labels[0], y)
        print(f"\n[Live Test] Single-label prediction: {labels[0]}")

    def test_multi_label_predict_live(self):
        """Test multi-label zero-shot classification with real Gemini API."""
        X = ["The new smartphone has a great camera and long battery life."]
        y = ["camera", "battery", "display", "price"]
        
        # We expect at least 'camera' and 'battery'
        clf = MultiLabelZeroShotVertexClassifier(max_labels=2)
        clf.fit(None, y)
        
        labels = clf.predict(X)
        self.assertEqual(len(labels), 1)
        # The mixin returns a list padded to max_labels
        self.assertIn("camera", labels[0])
        self.assertIn("battery", labels[0])
        print(f"\n[Live Test] Multi-label prediction: {labels[0]}")

if __name__ == "__main__":
    unittest.main()
