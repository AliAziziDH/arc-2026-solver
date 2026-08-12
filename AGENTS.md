# Agent Guide: arc-2026-solver
This repository contains a hybrid neuro-symbolic solver for the Abstraction and Reasoning Corpus (ARC-AGI-2).

## System Requirements
- Python 3.12
- Active virtual environment (source ~/.venv/bin/activate)
- Key dependencies: `google-genai`, `numpy`, `scipy`

## Execution Commands
To execute the pipeline locally on tasks:
```bash
python generate_submission.py --input tasks/ --output submission.json --timeout 60
```

## Testing Protocol
Run unit tests to verify pipeline consistency:
```bash
pytest
```

## Development & Agent Instructions
- Ensure all LLM operations are routed via `google-genai` using the `GEMINI_API_KEY` environment variable.
- Enforce symbol and color permutation equivariance in StateMemo by hashing canonical grid color-mappings.
- All code executions must be strictly isolated inside the multiprocessing IPyBoxSandbox in `solver/sandbox.py`.
- Always wrap task execution loops in high-level try/except blocks to provide raw input grid fallbacks.
