import os
import time
import random
import torch
import gc
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from solver.sandbox import safe_execute_solve, IPyBoxSandbox_run
from core.grid import get_object_metadata

class LLMSurgicalLifeline:
    def __init__(self, model_id: str = "/kaggle/input/qwen2.5-coder-7b-instruct"):
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
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
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

    def generate_mutated_test_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        mutated_pairs = []
        for inp, out in train_pairs:
            # 1. Bijective color mapping (preserve bg=0)
            colors = np.unique(np.concatenate([inp, out]))
            fg_colors = [c for c in colors if c != 0]
            if fg_colors:
                shuffled_fg = fg_colors.copy()
                random.shuffle(shuffled_fg)
                color_map = {c: s for c, s in zip(fg_colors, shuffled_fg)}
                color_map[0] = 0

                # Apply map to input
                new_inp = np.zeros_like(inp)
                for c in np.unique(inp):
                    if c in color_map:
                        new_inp[inp == c] = color_map[c]

                # Apply map to output
                new_out = np.zeros_like(out)
                for c in np.unique(out):
                    if c in color_map:
                        new_out[out == c] = color_map[c]
            else:
                new_inp, new_out = inp.copy(), out.copy()

            # 2. Random Dihedral S4 geometric transform
            k_rot = random.choice([0, 1, 2, 3])
            do_flip = random.choice([True, False])

            m_inp = np.rot90(new_inp, k=k_rot)
            m_out = np.rot90(new_out, k=k_rot)

            if do_flip:
                m_inp = np.fliplr(m_inp)
                m_out = np.fliplr(m_out)

            mutated_pairs.append((np.ascontiguousarray(m_inp), np.ascontiguousarray(m_out)))

        return mutated_pairs

    def synthesize_correction(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        partial_sequence: List[Tuple[str, Dict]],
        dsl_context: Dict[str, Any],
        max_retries: int = 10,
        max_repl_iterations: int = 3
    ) -> Optional[List[Tuple[str, Dict]]]:
        if not self._load_model():
            return None

        inp, out = train_pairs[0]
        current = inp.copy()
        try:
            for name, params in partial_sequence:
                func = dsl_context.get(name)
                if func is not None:
                    current = func(current.copy(), **params)
        except Exception:
            return None

        diff_str = self._get_diff_crop(current, out)
        obj_metadata = get_object_metadata(current)
        obj_meta_str = json.dumps(obj_metadata, indent=2)

        prompt = f"""You are an elite ARC-AGI expert programmer. A partial DSL program produced an incorrect grid.
Train Pair 0 Mismatch:
{diff_str}

Train Pair 0 Predicted Object Metadata:
{obj_meta_str}

Write a Python function named `solve()` that takes `input_grid` (a 2D numpy array) and returns the correct output 2D numpy array. You can use numpy operations and helper functions.
Return ONLY executable Python code inside a markdown code block.
"""

        messages = [
            {"role": "system", "content": "You are an expert ARC-AGI Python programmer."},
            {"role": "user", "content": prompt}
        ]

        locked_messages_count = len(messages)

        wait_time = 5.0
        candidate_scripts = []

        for attempt in range(max_retries):
            try:
                for repl_iter in range(max_repl_iterations):
                    text = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.2 + (0.05 * attempt),
                        do_sample=True
                    )
                    response_text = self._tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

                    code_str = self._extract_code(response_text)
                    messages.append({"role": "assistant", "content": response_text})

                    if not code_str:
                        messages.append({"role": "user", "content": "No code block found. Return ONLY executable Python code inside a markdown code block."})
                        continue

                    # Evaluate via stateful REPL sandbox on original pairs
                    feedback = IPyBoxSandbox_run(code_str, train_pairs, dsl_context, timeout_secs=2.0)

                    if feedback["success"]:
                        # Anti-Hardcoding Validation
                        mutated_pairs = self.generate_mutated_test_pairs(train_pairs)
                        mutated_feedback = IPyBoxSandbox_run(code_str, mutated_pairs, dsl_context, timeout_secs=2.0)

                        if mutated_feedback["success"]:
                            return [('llm_custom_patch', {'code_str': code_str})]
                        else:
                            # Program overfitted to static coords/colors
                            feedback_prompt = "CRITICAL ERROR: Overfitting/Hardcoding detected! Your program successfully solved the static training examples but failed to generalize when those same examples underwent geometric/color mutations. Rewrite your algorithm using general programmatic logic rather than hardcoded coordinate indexes or static grid values."
                            messages.append({"role": "user", "content": feedback_prompt})
                            if len(messages) > locked_messages_count + 4:
                                print(f"[LADDER Eviction] Sliding-window context eviction engaged: retained {locked_messages_count} locked rules, pruned intermediate traces down to 4.")
                                messages = messages[:locked_messages_count] + messages[-4:]
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
                    else:
                        error_msg = feedback["error"]
                        mismatches = feedback["mismatches"]

                        if error_msg is None and mismatches:
                            # It ran without throwing an error but had mismatches
                            candidate_scripts.append(code_str)

                        feedback_prompt = "The code did not pass all train pairs.\n"
                        if error_msg:
                            feedback_prompt += f"Execution Error:\n{error_msg}\n"
                        if mismatches:
                            first_mismatch = mismatches[0]
                            feedback_prompt += f"Mismatch on a train pair:\n"
                            feedback_prompt += f"Predicted Shape: {first_mismatch['pred_shape']}, Target Shape: {first_mismatch['target_shape']}\n"
                            feedback_prompt += f"Predicted Sample (top 3 rows): {first_mismatch['pred_sample']}\n"
                            feedback_prompt += f"Target Sample (top 3 rows): {first_mismatch['target_sample']}\n"

                        feedback_prompt += "Please correct the code."
                        messages.append({"role": "user", "content": feedback_prompt})

                        # Apply Sliding-Window token eviction to preserve memory
                        if len(messages) > locked_messages_count + 4:
                            print(f"[LADDER Eviction] Sliding-window context eviction engaged: retained {locked_messages_count} locked rules, pruned intermediate traces down to 4.")
                            messages = messages[:locked_messages_count] + messages[-4:]

                        # GC after evaluating feedback to keep memory down in iterative loop
                        gc.collect()
                        torch.cuda.empty_cache()

            except Exception as e:
                error_msg = str(e).lower()
                # Check for rate limit / quota exhaustion / 429 / resource exhausted
                if any(err_keyword in error_msg for err_keyword in ['429', 'resource_exhausted', 'rate_limit', 'quota', 'gpu out of memory', 'oom']):
                    jitter = random.uniform(0.1, 1.5)
                    sleep_duration = wait_time + jitter
                    print(f"[WARNING] Rate limit / quota / OOM encountered on attempt {attempt + 1}: {e}. Retrying in {sleep_duration:.2f} seconds...")
                    time.sleep(sleep_duration)
                    wait_time = min(wait_time * 2, 120.0)  # Exponential backoff capped at 120s
                    continue
                else:
                    print(f"LLM generation attempt {attempt + 1} error: {e}")
                    time.sleep(2.0)
                    continue

        # If we exhausted retries and have collected scripts, try holistic judge
        if candidate_scripts:
            # deduplicate
            candidate_scripts = list(set(candidate_scripts))
            return self.holistic_judge(candidate_scripts, train_pairs, dsl_context)

        return None

    def holistic_judge(self, candidate_scripts: List[str], train_pairs: List[Tuple[np.ndarray, np.ndarray]], dsl_context: dict) -> Optional[List[Tuple[str, Dict]]]:
        if not candidate_scripts:
            return None

        print(f"[LLM] Triggering Holistic Judge with {len(candidate_scripts)} candidate scripts...")

        inp, out = train_pairs[0]
        inp_str = self._grid_to_str(inp)
        out_str = self._grid_to_str(out)

        candidates_block = ""
        for i, script in enumerate(candidate_scripts[:5]): # cap at 5 scripts to prevent context overflow
            candidates_block += f"=== Candidate Script {i+1} ===\n```python\n{script}\n```\n\n"

        prompt = f"""You are an elite AI Architect acting as a Holistic Judge.
You are given an ARC-AGI input and output grid (Train Pair 0), and a list of candidate Python scripts.
All of these scripts executed without crashing, but failed to produce the exact target grid.

Input:
{inp_str}

Target:
{out_str}

{candidates_block}

Acting as a consensus judge, rigorously evaluate and compare the logical consistency and execution trace of each candidate script. Synthesize their correct components into a single, flawless Python function named `solve()` that takes `input_grid` and returns the correct 2D numpy array.
Return ONLY executable Python code inside a markdown code block.
"""

        messages = [
            {"role": "system", "content": "You are a holistic judge synthesising the best approach from multiple flawed Python scripts."},
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
                temperature=0.1,
                do_sample=True
            )
            response_text = self._tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

            code_str = self._extract_code(response_text)
            if code_str:
                feedback = IPyBoxSandbox_run(code_str, train_pairs, dsl_context, timeout_secs=2.0)
                if feedback["success"]:
                    return [('llm_custom_patch', {'code_str': code_str})]

        except Exception as e:
            print(f"[LLM] Holistic Judge error: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
