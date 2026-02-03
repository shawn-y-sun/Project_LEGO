import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import sys

# Ensure the repository root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Architect.model_evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        # Create a mock CM object
        self.mock_cm = MagicMock()
        self.mock_cm.model_id = 'TestModel_001'
        self.mock_cm.target = 'Sales'
        self.mock_cm.formula = 'Sales ~ InterestRate + GDP'

        # Create a mock ModelBase for in-sample model
        self.mock_model_in = MagicMock()

        # Mock performance measures
        # Note: Using keys as seen in the code (R² unicode)
        self.mock_model_in.in_perf_measures = pd.Series({
            'R²': 0.85,
            'Adj R²': 0.83,
            'RMSE': 120.5,
            'MAPE': 0.05
        })

        self.mock_model_in.out_perf_measures = pd.Series({
            'RMSE': 130.0,
            'MAPE': 0.06
        })

        self.mock_cm.model_in = self.mock_model_in

    @patch('Architect.model_evaluator.genai')
    def test_evaluate_success(self, mock_genai):
        # Setup mock LLM
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a strong candidate model."
        mock_model_instance.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model_instance

        # Instantiate evaluator with a dummy key
        evaluator = ModelEvaluator(api_key='dummy_key')

        # Run evaluation
        summary = evaluator.evaluate(self.mock_cm)

        # Assertions
        self.assertEqual(summary, "This is a strong candidate model.")

        # Check if generate_content was called
        mock_model_instance.generate_content.assert_called_once()

        # Check prompt content
        args, _ = mock_model_instance.generate_content.call_args
        prompt = args[0]
        self.assertIn("TestModel_001", prompt)
        self.assertIn("Sales", prompt)
        self.assertIn("0.8500", prompt) # R2
        self.assertIn("0.8300", prompt) # Adj R2
        self.assertIn("120.5000", prompt) # RMSE IS
        self.assertIn("130.0000", prompt) # RMSE OOS

    def test_missing_api_key(self):
        # Unset env var if present
        with patch.dict(os.environ, {}, clear=True):
            evaluator = ModelEvaluator(api_key=None)
            result = evaluator.evaluate(self.mock_cm)
            self.assertTrue(result.startswith("Error: Gemini API key not found"))

    @patch('Architect.model_evaluator.genai')
    def test_metrics_extraction_missing_oos_error(self, mock_genai):
        # Setup mock with empty OOS
        self.mock_model_in.out_perf_measures = pd.Series(dtype=float)

        mock_model_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model_instance

        evaluator = ModelEvaluator(api_key='dummy')
        result = evaluator.evaluate(self.mock_cm)

        # Assert that the result contains the error message about OOS metrics
        self.assertIn("Error extracting metrics", result)
        self.assertIn("Model out-of-sample performance measures are empty", result)

        # Ensure LLM was NOT called
        mock_model_instance.generate_content.assert_not_called()

    @patch('Architect.model_evaluator.genai')
    def test_metrics_extraction_missing_is_error(self, mock_genai):
        # Setup mock with empty IS
        self.mock_model_in.in_perf_measures = pd.Series(dtype=float)

        mock_model_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model_instance

        evaluator = ModelEvaluator(api_key='dummy')
        result = evaluator.evaluate(self.mock_cm)

        # Assert that the result contains the error message about IS metrics
        self.assertIn("Error extracting metrics", result)
        self.assertIn("Model in-sample performance measures are empty", result)

        # Ensure LLM was NOT called
        mock_model_instance.generate_content.assert_not_called()

if __name__ == '__main__':
    unittest.main()
