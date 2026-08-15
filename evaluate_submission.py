import json
import os

def evaluate():
    submission_path = "submission.json"
    tasks_dir = "tasks_eval/"

    with open(submission_path, 'r') as f:
        submission = json.load(f)

    tasks = [f for f in os.listdir(tasks_dir) if f.endswith(".json")]

    total_tasks = len(tasks)
    solved = 0
    failed = 0

    dim_mismatches = 0
    pixel_mismatches = 0
    timeouts = 0

    for task_file in tasks:
        task_id = task_file.split(".")[0]
        if task_id not in submission:
            continue

        with open(os.path.join(tasks_dir, task_file), 'r') as f:
            task_data = json.load(f)

        test_pairs = task_data.get('test', [])
        attempts = submission[task_id]

        task_solved = True

        for i, test_pair in enumerate(test_pairs):
            expected_output = test_pair['output']

            attempt_1 = attempts[i].get('attempt_1', [])
            attempt_2 = attempts[i].get('attempt_2', [])

            match_1 = attempt_1 == expected_output
            match_2 = attempt_2 == expected_output

            if not match_1 and not match_2:
                task_solved = False

                # Check failure modes based on attempt_1
                if not attempt_1 or len(attempt_1) != len(expected_output) or len(attempt_1[0]) != len(expected_output[0]):
                    if attempt_1 == test_pair['input']:
                        timeouts += 1
                    else:
                        dim_mismatches += 1
                else:
                    pixel_mismatches += 1
                break

        if task_solved:
            solved += 1
        else:
            failed += 1

    accuracy = (solved / total_tasks) * 100 if total_tasks > 0 else 0

    print("=" * 60)
    print("Local Scoring Evaluation")
    print("=" * 60)
    print(f"Total Tasks Evaluated: {total_tasks}")
    print(f"Exact-Match Accuracy: {accuracy:.2f}%")
    print(f"Tasks Solved: {solved}")
    print(f"Tasks Failed: {failed}")
    print("-" * 60)
    print("Failure Breakdown:")
    print(f"  Dimension Mismatch: {dim_mismatches}")
    print(f"  Pixel Mismatch (Correct Dimensions): {pixel_mismatches}")
    print(f"  Defaulted to Input (Timeout/Error/No Program): {timeouts}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
