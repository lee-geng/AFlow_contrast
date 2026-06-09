import unittest
import uuid
from pathlib import Path
from typing import Any, Callable, List, Tuple

from benchmarks.benchmark import BaseBenchmark


class _DummyBenchmark(BaseBenchmark):
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        return ("q", "p", "e", 1, 0.0)

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        return 1.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]


class BenchmarkCsvEncodingTests(unittest.TestCase):
    def test_save_results_csv_handles_unicode(self):
        tmp = Path(".tmp_tests") / f"csv_encoding_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        bench = _DummyBenchmark("dummy", file_path="", log_path=str(tmp))
        results = [("Q", "value ↔ value", "gold", 1, 0.1)]
        bench.save_results_to_csv(results, bench.get_result_columns())

        csv_files = list(tmp.glob("*.csv"))
        self.assertEqual(len(csv_files), 1)
        content = csv_files[0].read_text(encoding="utf-8-sig")
        self.assertIn("↔", content)


if __name__ == "__main__":
    unittest.main()
