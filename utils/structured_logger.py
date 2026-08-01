import os
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, log_filepath: str = "logs/runs.jsonl"):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)

    def classify_failure(self, train_pairs, best_score, sequence):
        if not sequence:
            return "NO_PROGRAM_FOUND"
        if best_score < 1.0:
            return "PARTIAL_MATCH_FAILURE"
        return "TRAIN_OVERFIT_OR_TEST_MISMATCH"

    def log_task_result(self, task_id: str, success: bool, sequence: list, beam_scores: list, train_pairs: list, test_pairs: list, duration_secs: float, nodes_explored: int = 0, depth_reached: int = 0):
        best_score = max(beam_scores) if beam_scores else 0.0
        failure_type = None if success else self.classify_failure(train_pairs, best_score, sequence)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "success": success,
            "best_score": best_score,
            "beam_scores": beam_scores,
            "program_sequence": sequence,
            "failure_type": failure_type,
            "duration_secs": duration_secs,
            "nodes_explored": nodes_explored,
            "depth_reached": depth_reached
        }

        with open(self.log_filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
