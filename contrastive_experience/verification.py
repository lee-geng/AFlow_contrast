from typing import Any, Dict


def compute_net_utility(
    target_repair_rate: float,
    success_regression_rate: float,
    non_target_failure_increase: float = 0.0,
    cost_increase: float = 0.0,
    lambda_success: float = 1.0,
    lambda_spillover: float = 0.5,
    lambda_cost: float = 0.1,
) -> float:
    return (
        target_repair_rate
        - lambda_success * success_regression_rate
        - lambda_spillover * non_target_failure_increase
        - lambda_cost * cost_increase
    )


def summarize_verification(
    target_repair_rate: float,
    success_regression_rate: float,
    non_target_failure_increase: float = 0.0,
    cost_increase: float = 0.0,
    regression_threshold: float = 0.15,
    overall_score_collapsed: bool = False,
) -> Dict[str, Any]:
    net_utility = compute_net_utility(
        target_repair_rate,
        success_regression_rate,
        non_target_failure_increase,
        cost_increase,
    )
    accepted = net_utility > 0 and success_regression_rate < regression_threshold and not overall_score_collapsed
    return {
        "target_repair_rate": target_repair_rate,
        "success_regression_rate": success_regression_rate,
        "non_target_failure_increase": non_target_failure_increase,
        "cost_increase": cost_increase,
        "net_utility": net_utility,
        "accepted": accepted,
    }

