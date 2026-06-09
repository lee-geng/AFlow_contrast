import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.async_llm import LLMsConfig
from scripts.optimizer import Optimizer


OPTIMIZED_PATH = (
    "workspace_smoke/"
    "exec__qwen3_32b_fp8__opt__qwen3_32b_fp8__"
    "qwen332bfp8_drop_sample4_run20_val3_exp_fullonly_noemb_conc5"
)


def main() -> None:
    models_config = LLMsConfig.default()
    llm_config = models_config.get("qwen3-32b-fp8")
    optimizer = Optimizer(
        dataset="DROP",
        question_type="qa",
        opt_llm_config=llm_config,
        exec_llm_config=llm_config,
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
        sample=4,
        optimized_path=OPTIMIZED_PATH,
        initial_round=1,
        max_rounds=1,
        validation_rounds=1,
        eval_max_concurrent_tasks=5,
        check_convergence=False,
        attribution_enabled=False,
        task_embedding_enabled=False,
    )
    optimizer.optimize("Test")


if __name__ == "__main__":
    main()
