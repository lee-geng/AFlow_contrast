# AFlow: Automating Agentic Workflow Generation

[![Arxiv](https://img.shields.io/badge/arXiv-AFlow-b31b1b)](https://arxiv.org/abs/2410.10762)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/FoundationAgents/AFlow/pulls)

> If you encounter any difficulties in using or reproducing the code, please contact me directly (Email: didi4goooogle@gmail.com, Wechat: 18831933368). Some Operators may have bugs during the migration from MetaGPT to this repository.


AFlow is a framework for automatically generating and optimizing Agentic Workflows. It uses Monte Carlo tree search in a code-represented workflow space to find effective workflows, replacing manual development with machine effort. Our approach shows potential to outperform handcrafted workflows on various tasks. 

We're building it to support more benchmarks and open-ended tasks! If you have any questions, please open an issue or email us!

<p align="center">
<a href=""><img src="assets/AFLOW-performance.jpg" alt="Performance Of AFlow" title="Performance of AFlow<sub>1</sub>" width="80%"></a>
</p>

## Framework Components

- **Node**: Basic unit of LLM invocation. See `metagpt_core/action_nodes/action_node.py` for a flexible interface to control LLM, temperature, format, and prompt.
- **Operator**: Predefined combinations of Nodes to enhance search efficiency. Encapsulates common operations like Generate, Format, Review, Revise, Ensemble, Test, and Programmer. See `operator.py` for details. You can customize your own Operator by referencing the implementations in this code.
- **Workflow**: A sequence of LLM-invoking nodes connected by edges. Can be represented as graphs, neural networks, or code to express various execution structures. See `workflow.py` for our implementation.
- **Optimizer**: Uses LLMs within a Monte Carlo Tree Search variant to explore and refine workflows. Iteratively selects, expands, evaluates, and updates workflows based on performance. See `optimizer.py` for details.
- **Evaluator**: Assesses workflow performance on given tasks. Provides feedback to guide the optimization process towards more effective workflows. See `evaluator.py` for details.

<p align="center">
<a href=""><img src="assets/AFLOW-method.jpg" alt="Framework of AFlow" title="Framework of AFlow <sub>1</sub>" width="80%"></a>
</p>

## Datasets

### Experimental Datasets
We conducted experiments on six datasets (HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP) and provide their evaluation code. The data can be found in this [datasets](https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e) link, or you can download them using `metagpt/ext/aflow/data/download_data.py`

<p align="center">
<a href=""><img src="assets/AFLOW-experiment.jpg" alt="Performance Of AFlow" title="Performance Of AFlow <sub>1</sub>" width="80%"></a>
</p>

### Custom Datasets
For custom tasks, you can reference the code in the `benchmark` folder. Inherit the `BaseBenchmark` class and implement `evaluate_problem`, `calculate_score`, and `get_result_columns` to add your custom dataset benchmark. Then, add your benchmark name in `evaluator.py` and `optimizer.py` to find effective workflows for your custom dataset.

## Quick Start

1. Set up the Python environment:
   ```bash
   # Create and activate a Python 3.9 virtual environment
   conda create -n <your_env_name> python=3.9

   # Install dependencies
   pip install -r requirements.txt
   ```

2. Configure optimization parameters:
   - Use command line arguments or modify default parameters in `run.py`:
     ```python
     --dataset              # (Required) Dataset type (HumanEval/MBPP/GSM8K/MATH/HotpotQA/DROP)
     --sample 4             # Sample count - number of workflows to be resampled
     --optimized_path PATH  # Optimized result save path
     --initial_round 1      # Initial round
     --max_rounds 20        # Max iteration rounds for AFLOW
     --check_convergence    # Whether to enable early stop
     --validation_rounds 5  # Validation rounds for AFLOW
     --if_force_download    # Force dataset download if set to True
     ```

3. Configure LLM parameters in `config/config2.yaml` (see `config/config2.example.yaml` for reference)

4. Set up operators in `run.py` and in `operator.py`, `optimized_path/template/operator.json`. You can reference our implementation to add operators for specific datasets

5. For first-time use, download datasets and initial rounds by setting `download(["datasets"])` in `run.py`

6. (Optional) Add your custom dataset and corresponding evaluation function following the [Custom Datasets](#custom-datasets) section

7. (Optional) If you want to use a portion of the validation data, you can set `va_list` in `evaluator.py`

8. Run the optimization:
   ```bash
   # Using default parameters
   python run.py --dataset MATH
   
   # Or with custom parameters
   python run.py --dataset MATH --sample n --optimized_path xxx ...
   ```

## Reproduce the Results in the Paper
1. We provide the raw data obtained from our experiments in this [link](https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv), including the workflows and prompts generated in each iteration, as well as their trajectories on the validation dataset. We also provide the optimal workflow for each dataset and the corresponding data on the test dataset. You can download these data using `data/download_data.py`. 
2. You can directly reproduce our experimental results by use different `ExperimentConfig` of `run.py`.

## Build Contrastive Dataset From AFlow Tree

This repo now includes a minimal pipeline to convert AFlow search-tree data into pairwise contrastive samples and train an edit ranker.

### 1) Build dataset

```bash
python data/build_contrastive_dataset.py \
  --input_path workspace/MATH \
  --output_path data/contrastive_math.jsonl \
  --task_family MATH
```

`--input_path` supports two formats:

- A single JSON file representing a tree/list of nodes.
- A workspace/log directory (`workspace/<dataset>` or `<...>/workflows`).

Adapter behavior for directory input:

- Uses `workflows/round_*/experience.json` as the primary parent-child edge source when available.
- Optionally reads `workflows/round_*/log.json` for extra edge/edit/score fields.
- Uses `workflows/results.json` for `score_mean` fallback and `workflows/round_*/graph.py` for workflow text.

Sampling policy:

- Positive child: `delta_score > 0.03`
- Negative child: `delta_score <= 0`
- Prefer sibling pairs under the same parent.
- If no sibling pair is available, fallback to parent-child binary pairs (`[NO_EDIT]` on one side).

### 2) Train edit ranker

```bash
python train/train_edit_ranker.py \
  --train_path data/contrastive_math.jsonl \
  --val_path data/contrastive_math.jsonl \
  --output_dir workspace/edit_ranker_math \
  --margin 0.2 \
  --batch_size 16 \
  --epochs 1
```

The baseline scorer is in `models/edit_scorer.py`:

- Input tuple: `(task_summary, parent_workflow_summary, candidate_edit)`
- Text is concatenated and encoded via a lightweight hashed bag-of-words encoder.
- Output is a scalar score.
- Training objective: `max(0, margin - s(pos) + s(neg))`

### 3) Output sample (`jsonl`)

```json
{
  "task_family": "MATH",
  "task_summary": "math reasoning",
  "parent_id": "p1",
  "parent_workflow": "....",
  "parent_workflow_summary": "....",
  "positive_edit": "add self-check",
  "positive_child_workflow": "....",
  "positive_delta_score": 0.06,
  "negative_edit": "remove tool",
  "negative_child_workflow": "....",
  "negative_delta_score": -0.02,
  "sample_type": "sibling_pair"
}
```

### 4) Minimal test

```bash
python -m unittest tests/test_contrastive_pipeline.py
```

## Amortizing AFlow Search with a Blame-Aware Workflow Revision Model

This prototype adds a complete pipeline:

1. Build a parent-conditioned, blame-aware revision dataset from AFlow tree/logs.
2. Attribute local failure modes (JudgeFlow-lite) to produce structured blame signals.
3. Train a workflow revision model that proposes the next local edit.
4. Insert the model back into AFlow as an amortized edit prior.

### What problem this solves

Vanilla AFlow still explores many edits blindly. This extension learns reusable edit priors from previous search trajectories and execution feedback so AFlow can prioritize higher-yield local modifications under a fixed budget.

### Embedding-based task similarity

The experience-guided path now supports embedding-based task matching instead of keyword-only task patterns. This makes memory retrieval and adaptive validation more portable across datasets such as MATH, HotpotQA, MBPP, and HumanEval.

Example with a local OpenAI-compatible embedding server:

```bash
python run.py \
  --dataset MATH \
  --experience_memory_enabled \
  --adaptive_validation_enabled \
  --task_embedding_enabled true \
  --task_embedding_model_name Qwen/Qwen3-Embedding-0.6B \
  --task_embedding_base_url http://localhost:8090/v1 \
  --task_embedding_api_key EMPTY
```

If the embedding endpoint is unavailable, the code falls back to a deterministic local hash embedding so tests and offline runs still work.

### A) Build revision dataset

```bash
python data/build_revision_dataset.py \
  --input_path workspace/MATH \
  --output_dir data/revision_math \
  --task_family MATH \
  --val_ratio 0.1
```

Outputs:

- `data/revision_math/train_revision.jsonl`
- `data/revision_math/val_revision.jsonl`
- `data/revision_math/revision_stats.json`

Supported input formats (`--input_path`):

- Unified tree JSON (`nodes` + optional `edges`)
- AFlow workspace/log directory (`workspace/<dataset>` or direct `.../workflows`)

Adapter design is implemented in `data/aflow_adapters.py` to avoid format-specific branching in the main builder.

### B) Blame attribution (JudgeFlow-lite)

`analysis/blame_attributor.py` provides a minimal blame-aware signal:

- `planning_error`
- `retrieval_error`
- `tool_usage_error`
- `verification_missing`
- `patch_incomplete`
- `unknown`

It uses workflow block heuristics (`await self.xxx(...)`) + execution feedback/trace keywords and returns:

```json
{
  "target_block": "b2",
  "blame_type": "verification_missing",
  "blame_confidence": 0.78,
  "blame_rationale": "..."
}
```

### C) Train revision model

```bash
python train/train_revision_model.py \
  --train_path data/revision_math/train_revision.jsonl \
  --val_path data/revision_math/val_revision.jsonl \
  --output_dir workspace/revision_model_math \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 0.3 \
  --feature_dim 2048 \
  --lora_rank 8
```

Model (`models/workflow_revision_model.py`) is a lightweight conditional generator baseline:

- context encoder: hashed text features
- LoRA-like low-rank adapter heads for `edit_type` and `target_block`
- nearest-neighbor memory for `edit_text` generation
- SFT-ready now, DPO export supported via a separate script

### D) Export DPO preference data

```bash
python data/export_dpo_dataset.py \
  --input_path data/revision_math/train_revision.jsonl \
  --output_path data/revision_math/train_revision_dpo.jsonl
```

Each row:

- `prompt`
- `chosen` (structured chosen edit JSON string)
- `rejected` (structured rejected edit JSON string)
- `meta` (task/blame/parent metadata)

### E) Integrate revision prior into AFlow

Run AFlow with prior:

```bash
python run.py \
  --dataset MATH \
  --use_revision_prior \
  --revision_prior_path workspace/revision_model_math \
  --revision_prior_mode generate_then_search
```

or

```bash
python run.py \
  --dataset MATH \
  --use_revision_prior \
  --revision_prior_path workspace/revision_model_math \
  --revision_prior_mode rank_then_search \
  --revision_rank_candidates 3
```

Modes:

- `generate_then_search`: prior proposes a local edit and injects it into optimization prompt.
- `rank_then_search`: sample multiple optimization candidates and rerank by prior score.

Per-round prior usage logs are written to:

- `workspace/<dataset>/workflows/round_*/revision_prior_log.json`

### F) Evaluation scripts

Revision model quality:

```bash
python eval/eval_revision_model.py \
  --model_dir workspace/revision_model_math \
  --data_path data/revision_math/val_revision.jsonl
```

Search-with-prior summary:

```bash
python eval/eval_search_with_prior.py \
  --workflows_dir workspace/MATH/workflows
```

## Experience-Guided Efficient Workflow Search

This repo now includes an optional experience-guided extension for AFlow-style search. It adds:

- node-level success/failure attribution from workflow execution traces
- compression of attribution events into workflow memories
- adaptive validation with targeted failure sets, success guard sets, and shared anchor sets

### Enable the extension

```bash
python run.py \
  --dataset MATH \
  --experience_memory_enabled \
  --adaptive_validation_enabled \
  --attribution_enabled \
  --experience_store_path workspace/MATH/workflows/experience_store
```

Useful knobs:

- `--experience_min_support`
- `--experience_min_confidence`
- `--experience_compression_interval`
- `--targeted_failure_size`
- `--success_guard_size`
- `--anchor_size`
- `--full_validation_top_k`

To keep artifacts from different execution/optimization models separate, you can enable model-scoped output directories:

```bash
python run.py \
  --dataset MATH \
  --optimized_path workspace \
  --separate_model_artifacts \
  --artifact_tag smoke \
  --opt_model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --exec_model_name meta-llama/Meta-Llama-3-8B-Instruct
```

This stores outputs under a path like:

- `workspace/exec__meta-llama_Meta-Llama-3-8B-Instruct__opt__meta-llama_Meta-Llama-3-8B-Instruct__smoke/MATH/...`

and writes `run_metadata.json` into the dataset root for traceability.

Artifacts are written under `workspace/<dataset>/workflows/experience_store/`:

- `execution_traces.jsonl`
- `attribution_events.jsonl`
- `workflow_memories.jsonl`
- `experience_metrics.jsonl`

### Report script

```bash
python eval/report_experience_guided.py \
  --store_path workspace/MATH/workflows/experience_store
```

### G) Minimal end-to-end checks

```bash
python -m unittest tests/test_revision_pipeline.py
```

This includes:

- chosen/rejected pair construction from mock tree
- structured blame attribution
- revision training sample -> model generation
- DPO export run
- revision prior invocation and prior log recording

## Roadmap

- Support multiple search algorithms
- Support multi model search in workflow
- Support LeaderBoard
- Support more benchmarks
- Support multimodality tasks

## Citation

If you use AFlow in your research, please cite our paper:

```
@inproceedings{
   zhang2025aflow,
   title={{AF}low: Automating Agentic Workflow Generation},
   author={Jiayi Zhang and Jinyu Xiang and Zhaoyang Yu and Fengwei Teng and Xiong-Hui Chen and Jiaqi Chen and Mingchen Zhuge and Xin Cheng and Sirui Hong and Jinlin Wang and Bingnan Zheng and Bang Liu and Yuyu Luo and Chenglin Wu},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=z5uVAKwmjf}
}
```
