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
from core.grid import get_object_metadata, canonicalize_grid

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

        _, color_map = canonicalize_grid(current)
        canonical_color_map_info = json.dumps(color_map)

        traceback_info = f"{diff_str}\n\nTrain Pair 0 Predicted Object Metadata:\n{obj_meta_str}"

        system_prompt = """### [SYSTEM INSTRUCTIONS]
You are the Active Coding Lifeline agent in AuroraGate v3.6.
Your goal is to write a Python 3 function `def solve(grid: List[List[int]]) -> List[List[int]]:`
that solves the given ARC puzzle.

[CONSTRAINTS]
- You must ONLY use the approved primitives defined below.
- Your output must be a valid python function wrapped in ```python ... ```.

[APPROVED PRIMITIVES]
def rotate_90(grid: np.ndarray) -> np.ndarray
def flip_h(grid: np.ndarray) -> np.ndarray
def flip_v(grid: np.ndarray) -> np.ndarray
def transpose(grid: np.ndarray) -> np.ndarray
def crop_bbox(grid: np.ndarray, bg: int) -> np.ndarray
def scale(grid: np.ndarray, factor: int) -> np.ndarray
def replace_color(grid: np.ndarray, old: int, new: int) -> np.ndarray
def keep_only_color(grid: np.ndarray, color: int, bg: int) -> np.ndarray
def remove_color(grid: np.ndarray, color: int, bg: int) -> np.ndarray
def extract_largest(grid: np.ndarray, bg: int) -> np.ndarray
def extract_smallest(grid: np.ndarray, bg: int) -> np.ndarray
def gravity_down(grid: np.ndarray, bg: int) -> np.ndarray
def fill_holes(grid: np.ndarray, bg: int) -> np.ndarray
def tile_to_size(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray
def pad_to_size(grid: np.ndarray, target_h: int, target_w: int, bg: int) -> np.ndarray
"""

        user_prompt = f"""### [WORKSPACE STATE]
The last compilation attempt failed. Here is the feedback from our stateful REPL executor:

[EXECUTION RESULTS & TRACEBACK]
{traceback_info}

[DIAGNOSTIC GUIDELINES]
1. If the error is 'DimensionMismatch', verify if you should apply 'crop_bbox' or 'pad_to_size' to align with target dimensions.
2. If the error is 'ColorMismatch', check your color mappings. Non-zero colors are canonicalized as: {canonical_color_map_info}.
3. Adjust your logic to satisfy the remaining Train Pairs. Do not repeat the failed code structure!
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        locked_messages_count = len(messages)

        wait_time = 5.0
        candidate_scripts = []

        def check_and_evict(messages_list):
            # Hybrid token eviction schema
            token_count = 0
            if self._tokenizer:
                try:
                    text = self._tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=False)
                    token_count = len(self._tokenizer.encode(text))
                except Exception:
                    token_count = sum(len(m['content']) for m in messages_list) // 4
            else:
                token_count = sum(len(m['content']) for m in messages_list) // 4

            if token_count > 6000:
                print(f"[LADDER Eviction] Context exceeds 6000 tokens ({token_count}). Evicting Turn N-2 and older.")
                # Pinned Message rule: strictly lock and preserve System Prompt and initial task description
                # which are in the first locked_messages_count messages.
                # Only keep the most recent user error prompt and assistant response.
                recent_messages = messages_list[locked_messages_count:]
                if len(recent_messages) > 2:
                    recent_messages = recent_messages[-2:]
                return messages_list[:locked_messages_count] + recent_messages
            return messages_list

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
                            messages = check_and_evict(messages)
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
                    else:
                        error_msg = feedback["error"]
                        mismatches = feedback["mismatches"]

                        if error_msg is None and mismatches:
                            # It ran without throwing an error but had mismatches
                            candidate_scripts.append(code_str)

                        # Capture full traceback and mismatches to populate {traceback_info}
                        tb_info = "The code did not pass all train pairs.\n"
                        if error_msg:
                            tb_info += f"Execution Error & Traceback:\n{error_msg}\n"
                        if mismatches:
                            first_mismatch = mismatches[0]
                            tb_info += f"Mismatch on a train pair:\n"
                            tb_info += f"Predicted Shape: {first_mismatch['pred_shape']}, Target Shape: {first_mismatch['target_shape']}\n"
                            tb_info += f"Predicted Sample (top 3 rows): {first_mismatch['pred_sample']}\n"
                            tb_info += f"Target Sample (top 3 rows): {first_mismatch['target_sample']}\n"

                        traceback_info = tb_info
                        feedback_prompt = f"""### [WORKSPACE STATE]
The last compilation attempt failed. Here is the feedback from our stateful REPL executor:

[EXECUTION RESULTS & TRACEBACK]
{traceback_info}

[DIAGNOSTIC GUIDELINES]
1. If the error is 'DimensionMismatch', verify if you should apply 'crop_bbox' or 'pad_to_size' to align with target dimensions.
2. If the error is 'ColorMismatch', check your color mappings. Non-zero colors are canonicalized as: {canonical_color_map_info}.
3. Adjust your logic to satisfy the remaining Train Pairs. Do not repeat the failed code structure!
"""

                        feedback_prompt += "Please correct the code."
                        messages.append({"role": "user", "content": feedback_prompt})

                        # Apply Sliding-Window token eviction to preserve memory
                        messages = check_and_evict(messages)

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
