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
   - The repository now includes these two ready-to-use local OpenAI-compatible model entries:
     - `meta-llama/Meta-Llama-3-8B-Instruct`
     - `qwen2.5-coder:7b`

4. Set up operators in `run.py` and in `operator.py`, `optimized_path/template/operator.json`. You can reference our implementation to add operators for specific datasets

5. For first-time use, download datasets and initial rounds by setting `download(["datasets"])` in `run.py`

6. (Optional) Add your custom dataset and corresponding evaluation function following the [Custom Datasets](#custom-datasets) section

7. (Optional) If you want to use a portion of the validation data, you can set `va_list` in `evaluator.py`

8. Run the optimization:
   ```bash
   # Using default parameters
   python run.py --dataset MATH
   
   # Or with custom parameters
   python run.py --dataset MATH --sample n --optimized_path workspace ...
   ```

9. Results are now isolated by model pair under:
   ```text
   workspace/model_runs/<DATASET>/opt_<OPT_MODEL>__exec_<EXEC_MODEL>/workflows
   ```
   This keeps workflows, logs, and scores from different optimization/execution model combinations separate.

## Contrastive Experience-Guided Diagnosis

This repository includes an optional v1 prototype for contrastive experience-guided workflow optimization. It does not change the default AFlow search path unless trace logging is explicitly enabled.

### Collect operator traces

Enable trace logging during a normal optimization run:

```bash
python run.py --dataset MATH --trace --auto_diagnose --trace_dir traces --trace_preview_chars 512
```

Optional flags:

```text
--trace_full_io                 Save full observable operator inputs/outputs instead of previews
--success_threshold FLOAT       Override dataset-aware success/failure grouping
```

Trace files are written as JSONL:

```text
traces/<DATASET>/<RUN_ID>/workflow_round_<ROUND>.jsonl
```

Each sample record stores the workflow round, sample id, final score/result, prediction, expected answer, and per-operator observable I/O metadata. It does not log private hidden reasoning.

When `--auto_diagnose` is enabled, AFlow automatically diagnoses every trace file generated in the current run after optimization finishes and writes the diagnosis artifacts next to the trace file.

To collect traces for an existing workflow round without running a full optimization loop:

```bash
python scripts/collect_traces.py --dataset MATH --round 1 --optimized_path workspace
```

### Diagnose saved traces separately

The separate diagnosis command is still available for reruns, debugging, or diagnosing old trace files:

```bash
python scripts/diagnose_traces.py traces/MATH/<RUN_ID>/workflow_round_1.jsonl
```

The script writes these files next to the trace unless `--output_dir` is provided:

```text
divergence_report.json
prototypes.json
attribution_prompt.txt
```

The report ranks candidate bottleneck operators using success/failure divergence over schema type, output style, parse status, and amplification from input to output features. The localization result intentionally uses the phrase "candidate bottleneck" rather than "root cause"; causality still requires edit verification.

### Update structured experience

After verifying a candidate child workflow, update the structured experience library:

```bash
python scripts/update_experience.py \
  --dataset MATH \
  --parent W7 \
  --child W9 \
  --edit_description "{\"edit_level\":\"operator\",\"edit_type\":\"replace_custom_with_answer_generate\"}" \
  --verification_results "{\"target_repair_rate\":0.42,\"success_regression_rate\":0.03,\"accepted\":true}"
```

Experience is saved under:

```text
experience_library/<DATASET>/experience.json
```

### Current prototype scope

Implemented: optional trace collection, rule-based trace abstraction, success/failure grouping, operator divergence scoring, bottleneck localization, prototype selection, attribution prompt generation, structured experience library updates, and utility-level edit verification metrics.

Not yet implemented: automatic contrastive guidance injection into AFlow expansion prompts, full workflow re-evaluation inside `verification.py`, and experience-guided parent selection.

## Patch-as-Hypothesis Workflow Repair

The repository also includes a Patch-as-Hypothesis workflow repair prototype. It takes an existing initial workflow, runs validation samples, traces failures, synthesizes local guarded patches, replays the failed sample plus matched previous successes, and keeps only patches that pass the online check. AFlow can provide the initial workflow, but the repair loop has its own entrypoint.

Repair an existing workflow directly:

```bash
python -m patch_evolution.repair_workflow ^
  --dataset DROP ^
  --workflow_dir workspace_patch_qwen3_fp8\model_runs\DROP\opt_qwen3_32b_fp8__exec_qwen3_32b_fp8\workflows\round_3 ^
  --model_name qwen3-32b-fp8 ^
  --trace_dir traces_repair_qwen3_fp8 ^
  --patch_registry_dir results_repair_qwen3_fp8 ^
  --output_dir repair_runs_qwen3_fp8 ^
  --run_id drop_round3_repair_qwen3_fp8 ^
  --repair_rounds 3 ^
  --stop_when_no_new_patches
```

The repair runner uses a hybrid strategy by default: deterministic rule patches handle simple answer-style and numeric normalization, while LLM repair patches handle missing `answer` fields and verbose answers. Add `--no_llm_repair` to run only deterministic rule patches.

The original AFlow path remains unchanged. If you only want AFlow to use already accepted guarded patches, enable the runtime wrapper with:

```bash
python run.py --dataset DROP --enable_patch_as_hypothesis --patch_registry_dir results/patch_evolution
```

Generate a candidate patch from a failed trace:

```bash
python -m patch_evolution.generate_patch --trace traces/DROP/<RUN_ID>/workflow_round_1.jsonl
```

Run a no-LLM toy example:

```bash
python -m patch_evolution.toy_example --output_dir logs/patch_evolution_toy
```

Validate and consolidate patches:

```bash
python -m patch_evolution.validate_patch --patch results/patch_evolution/patches/<PATCH_ID>.json
python -m patch_evolution.consolidate --registry results/patch_evolution
```

See `docs/patch_as_hypothesis.md` for the repair loop, TraceIR, PatchSpec, guarded runtime, telemetry, and consolidation design.

## Reproduce the Results in the Paper
1. We provide the raw data obtained from our experiments in this [link](https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv), including the workflows and prompts generated in each iteration, as well as their trajectories on the validation dataset. We also provide the optimal workflow for each dataset and the corresponding data on the test dataset. You can download these data using `data/download_data.py`. 
2. You can directly reproduce our experimental results by use different `ExperimentConfig` of `run.py`.

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
