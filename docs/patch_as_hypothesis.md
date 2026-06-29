# Patch-as-Hypothesis Workflow Repair

Patch-as-Hypothesis is a workflow repair method for an existing initial workflow. AFlow can provide that initial workflow, but the repair loop is independent of AFlow search: run the workflow, trace failures, attach local guarded patches, replay target and matched success samples, then consolidate useful patches offline.

## Motivation

AFlow modifies code-represented workflows after batch evaluation. A single failed trace can contain useful evidence, but immediately rewriting the workflow from that trace is risky. Patch-as-Hypothesis makes the repair local, reversible, trigger-conditioned, and measurable.

## Key Concepts

- `TraceIR`: a serializable execution trace with workflow metadata, final output, and operator-level records.
- `ProbeBeforeEdit`: a rule-extensible diagnosis pass over a failed trace.
- `PatchSpec`: a structured local repair hypothesis with trigger, fixer, contract, rollback, telemetry, and status.
- `TriggerDetector`: an executable predicate over observable runtime signals.
- `FixerExecutor`: executes a local repair such as contract repair, existing operator call, prompt patch, or mini workflow.
- `ContractChecker`: validates patched output before it can continue downstream.
- `PatchRegistry`: stores patch specs and telemetry under `results/patch_evolution`.
- `OfflineConsolidator`: uses patch telemetry to prune, keep, merge, or promote patches.

## Repair Flow

```text
initial workflow
-> run validation samples sequentially
-> failed sample creates operator trace
-> probe-before-edit diagnosis
-> synthesize local PatchSpec
-> temporarily activate patch
-> replay target failure sample
-> replay matched previous success samples
-> accept or disable patch
-> continue with accepted guarded patches
-> offline consolidation
```

## Stage A: Online Guarded Patching

Stage A is immediate and reversible. During operator execution, the original operator output is produced first. Active patches targeting that operator are then checked:

```text
original operator output
-> trigger detector
-> fixer
-> contract checker
-> accept patched output or rollback to original output
```

When `--enable_patch_as_hypothesis` is not set in AFlow, this wrapper is inactive and AFlow behavior is unchanged. For the independent repair method, use `patch_evolution.repair_workflow` instead of `run.py`.

## Stage B: Offline Patch Consolidation

Stage B uses accumulated telemetry:

- prune patches with low activation, low gain, high regression, high cost, or high contract failures;
- keep low-risk patches as guarded runtime safeguards;
- merge patches with the same target operator, failure symptom, and fixer type;
- mark high-frequency, high-gain, low-regression patches for promotion.

Promotion should create a new workflow version and validate it before adoption.

## Relationship to AFlow

AFlow is optional context, not the core method. AFlow can be used to generate or select an initial `round_N` workflow, and its benchmark/evaluation code is reused. The repair method then treats that workflow as fixed code and learns guarded local repairs from its failures.

## Commands

Run a no-LLM toy example:

```bash
python -m patch_evolution.toy_example --output_dir logs/patch_evolution_toy
```

Generate a patch from the first failed trace in a contrastive JSONL file:

```bash
python -m patch_evolution.generate_patch --trace traces/DROP/run/workflow_round_1.jsonl
```

Validate a patch:

```bash
python -m patch_evolution.validate_patch --patch results/patch_evolution/patches/patch_id.json
```

Inspect active guarded patches for an operator:

```bash
python -m patch_evolution.run_guarded --target_operator Solver
```

Consolidate a registry:

```bash
python -m patch_evolution.consolidate --registry results/patch_evolution
```

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
  --stop_when_no_new_patches ^
  --max_samples 200
```

By default, this uses hybrid repair candidates:

- deterministic rule patches for simple answer-style and numeric normalization;
- LLM repair patches for missing `answer` fields and verbose answers that need local extraction.

Use `--no_llm_repair` to disable LLM repair and keep only deterministic rule patches.

LLM repair still follows the guarded workflow: it can only use the task prompt and observable operator output. Gold answers are used only by replay verification to accept or reject the patch.

Run AFlow with active patches:

```bash
python run.py --dataset DROP --enable_patch_as_hypothesis --patch_registry_dir results/patch_evolution
```

## Current Limitations

- The online v1 focuses on observable answer-style/output-contract repairs. It is not meant to fix illegal workflow code such as missing `prompt.py` symbols.
- LLM-based PatchSpec synthesis is supported by interface, but tests use deterministic fallback synthesis.
- Full workflow promotion is represented as a consolidation decision, not an automatic graph rewrite.
- Static workflow validation catches missing `prompt_custom.*` symbols before spending a full validation run.
