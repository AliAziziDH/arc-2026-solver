import json

tasks_eval = ["00576224", "007bbfb7", "009d5c81", "00d62c1b", "00dbd492"]
log_file = "logs/runs.jsonl"
submission_file = "submission.json"
tasks_dir = "tasks_eval/"

results = []

with open(log_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['task_id'] in tasks_eval:
            results.append(data)

with open(submission_file, 'r') as f:
    submission = json.load(f)

for res in results:
    task_id = res['task_id']
    program = res['program_sequence']

    with open(f"{tasks_dir}{task_id}.json", 'r') as f:
        task_data = json.load(f)

    test_pair = task_data['test'][0]
    expected_output = test_pair['output']

    attempts = submission.get(task_id, [{}])
    attempt_1 = attempts[0].get('attempt_1', [])

    expected_dim = f"{len(expected_output)}x{len(expected_output[0])}"

    print(f"\nTask ID: {task_id}")
    print(f"Generated DSL Sequence: {program}")

    if attempt_1:
        attempt_dim = f"{len(attempt_1)}x{len(attempt_1[0])}"
        print(f"Target Dimensions: {expected_dim} vs Predicted Dimensions: {attempt_dim}")

        if expected_dim != attempt_dim:
            print("Failure Mode: Dimension Mismatch")
        else:
            print("Failure Mode: Pixel Mismatch")
            # Calculate pixel mismatches
            mismatches = []
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if expected_output[r][c] != attempt_1[r][c]:
                        mismatches.append(f"Coord ({r},{c}): Expected {expected_output[r][c]} vs Predicted {attempt_1[r][c]}")
            print(f"Total Mismatched Pixels: {len(mismatches)}")
            print("First 5 mismatches:")
            for m in mismatches[:5]:
                print(m)
    else:
        print("No prediction generated (Empty Attempt).")
