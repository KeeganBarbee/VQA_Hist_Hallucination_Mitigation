# VQA-Hist-Hallucination-Mitigation
This is a repository documenting my work for my independent study research on hallucination mitigation for histograms. It is important to note that not all of this code is uniquely mine. Some of it is uniquely mine, some was not used when running the actual models, and some of it had significant changes for my specific situation. Here, I will cite the paper where I found the code. In terms of what I used for this project, I heavily utilized this paepr Chang, Yue, et al. "A unified hallucination mitigation framework for large vision-language models." arXiv preprint arXiv:2409.16494 (2024). I also used the corresponding repository [Dentist](https://github.com/CYandYue/Dentist/blob/main/demo.py). They inspired most of my verifier.py, what I changed were the prompts being fed to the LVLMs and also the verify_perception and verify_reasoning functions. The demo.py file was significantly different and included other functions as a result of me having to use the same model for answering the questions and verifying. The separate verifier files for the two models I used were created by me, inspired by the paper.

## Baseline
We first select 2 LVLMs as our baseline models, including [LLaVa](https://ollama.com/library/llava:7b) and
[Qwen2.5VL](https://ollama.com/library/qwen2.5vl)

The sizes of baseline parameters are both 7 billion parameters.
Histogram VQA with Dentist Hallucination Mitigation

## Overview
 
This project adapts the **Dentist** hallucination mitigation framework — published in *Transactions on Machine Learning Research* (October 2024) — for the domain of histogram-based VQA. The central research question is whether Dentist's divide-and-conquer verification strategy generalizes to specialized scientific visualizations like histograms when using smaller open-source vision-language models.
 
---
 
## Background
 
### The Dentist Framework
 
Dentist mitigates hallucinations in VQA without requiring a separate, more powerful verifier model. It classifies questions into two types and handles each differently:
 
```
Question
   │
   ├── Perception (directly readable from image)
   │       └── Generate sub-questions → Answer with image → Aggregate → Corrected answer
   │
   └── Reasoning (inferred or interpreted from image)
           └── Chain-of-thought generation → Extract corrected answer
```
 
A **conflict-detection loop** then compares the revised answer to a second independent verification pass. If the answers conflict, the process iterates up to `limited_cnt` times.
 
### Why Histograms?
 
Histograms present a unique challenge for VLMs:
- They require distinguishing x-axis (data values) from y-axis (frequency)
- Bar counting requires precise visual enumeration
- Statistical properties like mean and median must be inferred, not read directly
- Models trained on general web data have strong priors that override what they actually see
---
 
## Dataset
 
**[`ReadingTimeMachine/visual_qa_histograms`](https://huggingface.co/datasets/ReadingTimeMachine/visual_qa_histograms)** — synthetic histogram images paired with structured QA pairs across six question types:
 
| Question Type | Description | Scoring |
|---|---|---|
| `nbars` | How many bars are in the histogram? | Exact match |
| `ngaussians` | How many Gaussians generated the data? | Exact match |
| `minimum` | Minimum value on the x-axis? | Relative error |
| `maximum` | Maximum value on the x-axis? | Relative error |
| `median` | Median value of the data? | Relative error |
| `mean` | Mean value of the data? | Relative error |

---
 
## Project Structure
 
```
Dentist/
├── demo.py                        # Main experiment runner
├── evaluation.ipynb               # Scoring and analysis notebook
├── results.json                   # Raw model outputs
├── results_scored.csv             # Per-result scores
├── results_summary.csv            # Aggregated accuracy by model and question type
└── Dentist/
    ├── __init__.py
    └── model/
        ├── __init__.py
        ├── verifier.py            # Base Dentist class with histogram-specific prompts
        ├── my_llava/
        │   ├── __init__.py
        │   └── my_llava_verifier.py
        └── Qwen2VL/
            ├── __init__.py
            └── Qwen2VL_Verifier.py
```
 
---
 
## Setup
```
Run ollama.ipynb
```
### Download the Dataset
 
```python
from datasets import load_dataset
dataset = load_dataset("ReadingTimeMachine/visual_qa_histograms")
```
 
---
 
## Usage
 
### Run the Full Experiment
 
```bash
cd /path/to/Dentist
python demo.py
```
 
Results are saved incrementally to `results.json` after each sample. To test on a subset first:
 
```python
# In demo.py
max_samples = 5    # set to None for full dataset
```
 
## Prompt Engineering
 
A core contribution of this project is redesigning Dentist's generic prompts for the histogram domain. Four failure modes were identified and addressed:
 
### 1. Axis Confusion
Models read y-axis bar heights instead of x-axis data values when asked for min/max/mean/median.
 
```python
# Added to every reasoning CoT prompt
"CRITICAL:\n"
"- The X-AXIS shows DATA VALUES (the range of the actual data).\n"
"- The Y-AXIS shows FREQUENCY (how many data points fall in each bin).\n"
"- min/max/mean/median refer to X-AXIS values, NOT bar heights.\n"
```
 
### 2. Verbose Output
Qwen produced paragraph-length answers when single numbers were needed.
 
```python
# Appended to every image-facing call
"Reply with ONLY the answer — a number or short value. No explanation."
```
 
### 3. Hallucinated Bar Counts
Both models defaulted to "20 bars" regardless of actual histogram content.
 
```python
# Sub-question generation restricted to visual properties only
"Focus only on: bar count, x-axis range, y-axis range, bar heights.\n"
"Do NOT ask about statistics, distributions, or interpretations.\n"
```
 
### 4. Classification Calibration
The question classifier was updated with histogram-specific examples so that perception vs. reasoning routing worked correctly for scientific chart questions.
 
---
 
## Evaluation
 
Run `evaluation.ipynb` after `demo.py` completes. 
 
## Key Findings
 
### 🔴 Axis Misreading
Both models systematically confused x-axis and y-axis values. When asked for minimum data value, models returned bar heights — producing answers like `1500` when the correct answer was `-0.37`. This was consistent across all numerical question types.
 
### 🔴 Bar Count Hallucination
Qwen consistently reported **"20 bars"** regardless of actual histogram content, defaulting to a memorized prior from training data. LLaVA frequently refused to answer, citing insufficient context.
 
### 🔴 Self-Reinforcing Errors — Core Finding
The most significant result: **Dentist's verification loop does not improve accuracy for histogram VQA with 7B models.**
 
```
Baseline: "The figure shows a histogram with 20 bars."
Revised:  "20"
#          ↑ same wrong answer — the model confirmed its own hallucination
```
 
When the same model performs both the initial answer and the verification sub-questions, it consistently agrees with its own incorrect answer. This self-reinforcement is the fundamental limitation that the original Dentist paper avoided by using GPT-4 as a stronger external verifier.
 
> **Takeaway:** Single-model self-correction is insufficient for specialized visual domains where the model's baseline visual understanding is weak. A stronger, separate verifier is necessary to break the self-reinforcement cycle.
 
---
 
## Limitations
 
- **Zero-shot only** — models were not fine-tuned on histogram data
- **Model scale** — 7B parameter models may lack the visual precision needed for scientific chart reading
- **Single-model verification** — Dentist's full potential requires a stronger verifier than the model being tested
- **Synthetic data** — performance on real scientific figures from papers may differ
- **Hardware dependency** — GPU with sufficient VRAM is required; CPU inference is impractically slow for full-dataset runs
---
 
## References
 
- **Dentist:** Chang, Yue, et al. "A unified hallucination mitigation framework for large vision-language models." arXiv preprint arXiv:2409.16494 (2024).
- **Dataset:** [ReadingTimeMachine/visual_qa_histograms](https://huggingface.co/datasets/ReadingTimeMachine/visual_qa_histograms)
- **Ollama:** [ollama.com](https://ollama.com)
## Models
Two open-source vision-language models were evaluated:

LLaVA 7B (llava:7b) — served locally via Ollama
Qwen2.5-VL 7B (qwen2.5vl:7b) — served locally via Ollama
