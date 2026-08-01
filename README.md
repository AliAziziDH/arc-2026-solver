# 🧠 ARC-AGI Solver

<div align="center">
  <br/>
  <em>A hybrid neuro-symbolic solver for the Abstraction and Reasoning Corpus</em>
  <br/><br/>
  <img src="https://img.shields.io/badge/Kaggle-Ready-blue" alt="Kaggle Ready"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/ARC-AGI-2-orange" alt="ARC-AGI-2"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
</div>

---

## 📖 Overview

This repository contains a **production-ready solver** for the **Abstraction and Reasoning Corpus (ARC-AGI)** benchmark — a challenging visual reasoning benchmark that tests general intelligence in AI systems.

Our approach combines:

- **Symbolic search** with a Domain-Specific Language (DSL) of 15 core primitives
- **Beam Search** with intelligent parameter synthesis
- **5 compositional macros** for complex transformations
- **LLM Surgical Lifeline** (Qwen2.5-Coder-7B) for hard cases
- **Dynamic time budgeting** and **memory management** for 9-hour Kaggle runs

The system achieved competitive validation accuracy, placing it in the top tier of open-source symbolic solvers.

---

## 🎯 Features

| Feature | Description |
| :--- | :--- |
| **15 Core Primitives** | Geometric (rotate, flip, crop), color (replace, remove, keep), object-based (extract largest/smallest, gravity), structural (fill holes, tile, pad) |
| **5 Compositional Macros** | `crop_then_gravity`, `extract_largest_and_center`, `remove_small_noise`, `symmetrize_hv`, `scale_to_output` |
| **Intelligent Search** | Beam Search with state memoization, dynamic pruning, and CEGIS verification |
| **Dimension Harmonization** | Automatic dimension correction (crop, pad, scale) for near-miss solutions |
| **LLM Surgical Lifeline** | Qwen2.5-Coder-7B (4-bit) generates corrective code for high-score partial solutions |
| **Time Budgeting** | Dynamic allocation: 20s base + extensions based on progress |
| **Robust Logging** | JSONL structured logs with failure taxonomy and checkpoint/resume |
| **Kaggle-Ready** | Memory management (GC, CUDA cache), 9-hour runtime optimization |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR (main.py)                        │
│  - Loads JSON tasks                                                 │
│  - Manages global time budget (9h)                                  │
│  - Checkpoint/Resume                                                │
│  - Structured logging (JSONL)                                       │
└──────────────┬──────────────────────────────┬──────────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│      BEAM SEARCH ENGINE      │  │      LLM SURGICAL LIFELINE       │
│    (solver/enumerator.py)    │  │    (solver/llm_lifeline.py)      │
│                              │  │                                  │
│   ┌────────────────────────┐ │  │  - Qwen2.5-Coder-7B (4-bit)      │
│   │   15 Primitives +      │ │  │  - Surgical prompt generation    │
│   │   5 Macros             │ │  │  - Code validation via sandbox   │
│   └────────┬───────────────┘ │  │  - Retry mechanism (3 attempts)  │
│            │                 │  └──────────────────────────────────┘
│   ┌────────▼───────────────┐ │
│   │   StateMemo (pruning)  │ │
│   │   - tobytes() hashing  │ │
│   │   - Max 200k entries   │ │
│   └────────┬───────────────┘ │
│            │                 │
│   ┌────────▼───────────────┐ │
│   │   Dimension Harmonize  │ │
│   │   - crop/pad/scale     │ │
│   └────────────────────────┘ │
└──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CORE PRIMITIVES (core/)                        │
│  rotate_90, flip_h, flip_v, transpose, crop_bbox, scale,            │
│  replace_color, keep_only_color, remove_color,                      │
│  extract_largest, extract_smallest, gravity_down,                   │
│  fill_holes, tile_to_size, pad_to_size                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/AliAziziDH/arc-2026-solver.git
cd arc-2026-solver

# Install dependencies
pip install -r requirements.txt

# Download the model (optional, for LLM lifeline)
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', \
    device_map='auto')"
```

### Run on a Single Task

```python
from main import solve_single_task
from utils.loader import load_arc_task

task = load_arc_task("tasks/sample.json")
result = solve_single_task(task, timeout_seconds=30)
print(f"Solved: {result['success']}")
print(f"Program: {result['sequence']}")
```

### Run on All Tasks (Kaggle Mode)

```bash
python generate_submission.py --input tasks/ --output submission.csv --timeout 60
```

### Analyze Results

```python
import pandas as pd
logs = pd.read_json("logs/runs.jsonl", lines=True)
print(logs['status'].value_counts())
print(logs['classify_failure'].value_counts())
```

---

## 🛠️ Configuration

Key parameters in `main.py` and `solver/enumerator.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `beam_width` | 32 | Number of top candidates per depth |
| `max_depth` | 3 | Maximum search depth (4 with macros) |
| `timeout_per_task` | 60s | Base timeout per task |
| `llm_threshold` | 0.75 | Minimum score to trigger LLM |
| `memo_max_size` | 200k | Max entries in StateMemo |
| `gc_interval` | 10 tasks | Memory cleanup frequency |

---

## 📂 Project Structure

```
arc-2026-solver/
├── core/
│   ├── grid.py              # Object detection & background inference
│   └── primitives.py        # 15 core transform primitives
├── solver/
│   ├── enumerator.py        # Beam Search + CEGIS
│   ├── memo.py              # State memoization (tobytes hashing)
│   ├── sandbox.py           # Safe execution (multiprocessing)
│   └── llm_lifeline.py      # Qwen2.5-Coder integration
├── utils/
│   ├── structured_logger.py # JSONL logging + failure taxonomy
│   ├── memory_manager.py    # GC + CUDA cache management
│   └── loader.py            # ARC JSON parser
├── docs/
│   └── images/              # Gallery images
├── tasks/                   # ARC task JSON files
├── logs/                    # JSONL logs, checkpoints
├── main.py                  # Orchestrator
├── generate_submission.py   # Kaggle submission generator
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 📜 License

MIT License — see [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## 🙏 Acknowledgments

* [ARC Prize](https://arcprize.org/) for the benchmark
* [Verantyx](https://github.com/verantyx) for DSL inspiration
* [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) for the language model
* [Cline](https://github.com/cline/cline) for the amazing VS Code extension

---

## 📧 Contact

**Ali Azizi** — [GitHub](https://github.com/AliAziziDH) | [LinkedIn](https://linkedin.com/in/ali-azizi-dh)