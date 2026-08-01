import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from solver.sandbox import safe_execute_solve

class LLMSurgicalLifeline:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load LLM {self.model_id}: {e}. Falling back to pure Beam Search.")
            self._model = None
            self._tokenizer = None
            return False

    def _grid_to_str(self, grid: np.ndarray) -> str:
        return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)

    def _get_diff_crop(self, pred: np.ndarray, target: np.ndarray) -> str:
        return f"Predicted:\n{self._grid_to_str(pred)}\nTarget:\n{self._grid_to_str(target)}"

    def synthesize_correction(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        partial_sequence: List[Tuple[str, Dict]],
        dsl_context: Dict[str, Any]
    ) -> Optional[List[Tuple[str, Dict]]]:
        if not self._load_model():
            return None

        inp, out = train_pairs[0]
        current = inp.copy()
        try:
            for name, params in partial_sequence:
                func = dsl_context[name]
                current = func(current, **params)
        except Exception:
            return None

        diff_str = self._get_diff_crop(current, out)

        prompt = f"""You are an elite ARC-AGI expert programmer. A partial DSL program produced an incorrect grid.
Train Pair 0 Mismatch:
{diff_str}

Write a Python function named `solve()` that takes `input_grid` (a 2D numpy array) and returns the correct output 2D numpy array. You can use numpy operations and helper functions.
Return ONLY executable Python code inside a markdown code block.
"""

        messages = [
            {"role": "system", "content": "You are an expert ARC-AGI Python programmer."},
            {"role": "user", "content": prompt}
        ]

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True
            )
            response_text = self._tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            code_str = self._extract_code(response_text)
            if not code_str:
                return None

            success = True
            for inp_t, out_t in train_pairs:
                pred, err = safe_execute_solve(code_str, inp_t, dsl_context, timeout_secs=2)
                if err or not np.array_equal(pred, out_t):
                    success = False
                    break

            if success:
                return [('llm_custom_patch', {'code_str': code_str})]
        except Exception as e:
            print(f"LLM generation error: {e}")
            return None

        return None

    def _extract_code(self, response: str) -> Optional[str]:
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        return None
