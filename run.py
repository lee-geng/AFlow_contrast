# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from pathlib import Path
from typing import Dict, List

from data.download_data import download
from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "LiveCodeBench": ExperimentConfig(
        dataset="LiveCodeBench",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="workspace",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")
    parser.add_argument("--check_convergence", type=bool, default=True, help="Whether to enable early stop")
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds")
    parser.add_argument(
        "--if_force_download",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether enforce dataset download.",
    )
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="qwen2.5-coder:7b",
        help="Specifies the name of the model used for execution tasks.",
    )
    parser.add_argument("--trace", action="store_true", help="Enable optional operator-level trace logging")
    parser.add_argument("--trace_dir", type=str, default="traces", help="Directory for contrastive trace JSONL files")
    parser.add_argument("--trace_full_io", action="store_true", help="Log full observable operator inputs/outputs")
    parser.add_argument(
        "--trace_preview_chars",
        type=int,
        default=512,
        help="Maximum characters for trace input/output previews",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=None,
        help="Override dataset-aware success threshold for trace success/failure grouping",
    )
    parser.add_argument(
        "--auto_diagnose",
        action="store_true",
        help="After a traced run, automatically diagnose generated trace files",
    )
    parser.add_argument(
        "--enable_patch_as_hypothesis",
        action="store_true",
        help="Enable guarded runtime Patch-as-Hypothesis patches",
    )
    parser.add_argument(
        "--patch_registry_dir",
        type=str,
        default="results/patch_evolution",
        help="Patch-as-Hypothesis registry directory",
    )
    return parser.parse_args()


def run_trace_diagnosis(args, optimizer) -> None:
    if not args.trace:
        print("--auto_diagnose was set, but --trace is disabled; no trace files were generated.")
        return

    from contrastive_experience.report import diagnose_trace_file, write_diagnosis_outputs

    trace_root = Path(args.trace_dir) / args.dataset / optimizer.trace_run_id
    trace_files = sorted(trace_root.glob("workflow_*.jsonl"))
    if not trace_files:
        print(f"No trace files found under {trace_root}")
        return

    print(f"Running contrastive diagnosis for {len(trace_files)} trace file(s) under {trace_root}")
    for trace_file in trace_files:
        report = diagnose_trace_file(trace_file, success_threshold=args.success_threshold)
        output_dir = trace_file.parent / trace_file.stem
        write_diagnosis_outputs(trace_file, report, output_dir=output_dir)
        candidate = report.get("localization", {}).get("candidate_operator")
        confidence = report.get("localization", {}).get("confidence")
        confidence_text = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "n/a"
        print(
            f"Diagnosed {trace_file}: candidate_bottleneck={candidate}, "
            f"confidence={confidence_text}, output_dir={output_dir}"
        )


if __name__ == "__main__":
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    download(["datasets"], force_download=args.if_force_download) # remove download initial_rounds in new version.

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
        trace_enabled=args.trace,
        trace_dir=args.trace_dir,
        trace_full_io=args.trace_full_io,
        trace_preview_chars=args.trace_preview_chars,
        success_threshold=args.success_threshold,
        enable_patch_as_hypothesis=args.enable_patch_as_hypothesis,
        patch_registry_dir=args.patch_registry_dir,
    )

    # Optimize workflow via setting the optimizer's mode to 'Graph'
    optimizer.optimize("Graph")

    if args.auto_diagnose:
        run_trace_diagnosis(args, optimizer)

    # Test workflow via setting the optimizer's mode to 'Test'
    # optimizer.optimize("Test")
