import os
import pandas as pd
from typing import Optional, Dict, Any, Union
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelEvaluator:
    """
    Agent C: Model Evaluator.

    This agent evaluates a Candidate Model (CM) by extracting performance metrics
    and generating a natural-language summary using an LLM (Gemini).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-3-flash-preview'):
        """
        Initialize the ModelEvaluator.

        Parameters
        ----------
        api_key : str, optional
            Google Gemini API key. If None, tries to load from 'GENAI_API_KEY' environment variable.
        model_name : str
            Name of the Gemini model to use. Defaults to 'gemini-pro'.
        """
        self.api_key = api_key or os.getenv('GENAI_API_KEY')
        self.model_name = model_name
        self.model = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                # Log error or print warning, but don't crash init
                # (allows instantiation without key for limited usage)
                print(f"Warning: Failed to configure Gemini API: {e}")

    def _extract_metrics(self, cm) -> Dict[str, Any]:
        """
        Extract key performance metrics from the Candidate Model.

        Parameters
        ----------
        cm : Technic.cm.CM
            The Candidate Model object.

        Returns
        -------
        dict
            Dictionary containing extracted metrics.

        Raises
        ------
        ValueError
            If in-sample or out-of-sample performance metrics are empty.
        """
        metrics = {
            'model_id': getattr(cm, 'model_id', 'Unknown'),
            'target': getattr(cm, 'target', 'Unknown'),
            'formula': getattr(cm, 'formula', 'Unknown'),
            'in_sample': {},
            'out_sample': {}
        }

        # Access in-sample model metrics
        if hasattr(cm, 'model_in') and cm.model_in is not None:
            # We use the public property in_perf_measures which aggregates Fit and Error measures
            in_perf = cm.model_in.in_perf_measures
            if in_perf.empty:
                raise ValueError("Model in-sample performance measures are empty.")

            # Keys are usually 'R²', 'Adj R²', 'RMSE', 'MAE', 'ME'
            # We map them to standard keys for the prompt
            metrics['in_sample'] = {
                'R2': in_perf.get('R²', in_perf.get('R2', None)),
                'Adj_R2': in_perf.get('Adj R²', in_perf.get('Adj R2', None)),
                'MAPE': in_perf.get('MAPE', None),
                'RMSE': in_perf.get('RMSE', None)
            }

            # Access out-of-sample metrics
            # Note: out_perf_measures is a property of the model instance (model_in)
            # if the testset was built with OOS measures.
            out_perf = cm.model_in.out_perf_measures
            if out_perf.empty:
                raise ValueError("Model out-of-sample performance measures are empty.")

            metrics['out_sample'] = {
                'MAPE': out_perf.get('MAPE', None),
                'RMSE': out_perf.get('RMSE', None)
            }
        else:
            # Fallback if model_in is missing entirely
             raise ValueError("Model in-sample performance measures are empty (model_in is None).")

        return metrics

    def _construct_prompt(self, metrics: Dict[str, Any]) -> str:
        """
        Construct the prompt for the LLM.
        """
        prompt = f"""
                You are an expert econometrician and model evaluator.
                Review the following Candidate Model (CM) performance metrics and provide a concise, professional summary of its fitness.

                **Model Identification:**
                - ID: {metrics['model_id']}
                - Target: {metrics['target']}
                - Formula: {metrics['formula']}

                **In-Sample Performance:**
                """
        in_sample = metrics.get('in_sample', {})
        for k, v in in_sample.items():
            if v is not None:
                prompt += f"- {k}: {v:.4f}\n"
            else:
                prompt += f"- {k}: N/A\n"

        prompt += "\n**Out-of-Sample Performance:**\n"
        out_sample = metrics.get('out_sample', {})
        # OOS is now guaranteed by _extract_metrics
        for k, v in out_sample.items():
            if v is not None:
                prompt += f"- {k}: {v:.4f}\n"
            else:
                prompt += f"- {k}: N/A\n"

        prompt += """
                **Instructions:**
                1. Evaluate the In-Sample fit quality (R2, Adj R2). High is generally > 0.8, but depends on context.
                2. Compare In-Sample vs Out-of-Sample errors (MAPE/RMSE) to check for overfitting. If OOS error is significantly higher than IS error, flag it.
                3. Provide a final verdict: "Strong Candidate", "Potential Candidate" (with caveats), or "Weak Candidate".
                4. Keep the response under 150 words.
                """
        return prompt

    def evaluate(self, cm) -> str:
        """
        Evaluate the candidate model and return a natural language summary.

        Parameters
        ----------
        cm : Technic.cm.CM
            The Candidate Model to evaluate.

        Returns
        -------
        str
            The evaluation summary from the LLM.
        """
        if not self.api_key:
            return "Error: Gemini API key not found. Please set GENAI_API_KEY environment variable."

        if self.model is None:
             # Try re-initializing if it failed in __init__ but key is present now?
             try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
             except Exception as e:
                 return f"Error configuring Gemini API: {str(e)}"

        try:
            metrics = self._extract_metrics(cm)
        except ValueError as ve:
             return f"Error extracting metrics: {str(ve)}"

        prompt = self._construct_prompt(metrics)

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                return "Error: LLM returned empty response."
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
