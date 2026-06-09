import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class RevisionModelConfig:
    feature_dim: int = 2048
    lora_rank: int = 8
    temperature: float = 1.0


class WorkflowRevisionModel:
    """A minimal conditional generation baseline with LoRA-like low-rank adapters.

    - Context encoder: hashed bag-of-words.
    - Learns low-rank adapters for edit_type and target_block classification.
    - Generates edit text with nearest-neighbor retrieval from SFT memory bank.
    """

    def __init__(self, config: RevisionModelConfig):
        self.config = config
        self.edit_types: List[str] = []
        self.target_blocks: List[str] = []

        self.base_type_w: np.ndarray = np.zeros((1, config.feature_dim), dtype=np.float32)
        self.base_block_w: np.ndarray = np.zeros((1, config.feature_dim), dtype=np.float32)

        self.type_A: np.ndarray = np.zeros((1, config.lora_rank), dtype=np.float32)
        self.type_B: np.ndarray = np.zeros((config.lora_rank, config.feature_dim), dtype=np.float32)

        self.block_A: np.ndarray = np.zeros((1, config.lora_rank), dtype=np.float32)
        self.block_B: np.ndarray = np.zeros((config.lora_rank, config.feature_dim), dtype=np.float32)

        self.memory_embeddings: np.ndarray = np.zeros((0, config.feature_dim), dtype=np.float32)
        self.memory_outputs: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall((text or "").lower())

    def encode_context(self, task_summary: str, parent_workflow: str, execution_feedback: Dict[str, Any], blame: Dict[str, Any]) -> np.ndarray:
        feedback_text = json.dumps(execution_feedback or {}, ensure_ascii=False)
        blame_text = json.dumps(blame or {}, ensure_ascii=False)
        merged = (
            f"Task:\n{task_summary}\n\n"
            f"Current workflow:\n{parent_workflow}\n\n"
            f"Execution feedback:\n{feedback_text}\n\n"
            f"Blame signal:\n{blame_text}\n"
        )
        tokens = self._tokenize(merged)
        x = np.zeros(self.config.feature_dim, dtype=np.float32)
        if not tokens:
            return x
        for token in tokens:
            x[hash(token) % self.config.feature_dim] += 1.0
        x /= float(len(tokens))
        return x

    def _effective_type_w(self) -> np.ndarray:
        return self.base_type_w + np.matmul(self.type_A, self.type_B)

    def _effective_block_w(self) -> np.ndarray:
        return self.base_block_w + np.matmul(self.block_A, self.block_B)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z / max(self.config.temperature, 1e-6))
        return e / max(np.sum(e), 1e-8)

    def _prepare_label_space(self, rows: List[Dict[str, Any]]) -> None:
        self.edit_types = sorted({str(r["chosen_edit"].get("edit_type", "unknown")) for r in rows}) or ["unknown"]
        self.target_blocks = sorted({str(r["chosen_edit"].get("target_block", "b0")) for r in rows}) or ["b0"]

        n_type = len(self.edit_types)
        n_block = len(self.target_blocks)
        self.base_type_w = np.zeros((n_type, self.config.feature_dim), dtype=np.float32)
        self.base_block_w = np.zeros((n_block, self.config.feature_dim), dtype=np.float32)

        rng = np.random.default_rng(42)
        self.type_A = (rng.normal(0.0, 0.01, size=(n_type, self.config.lora_rank))).astype(np.float32)
        self.type_B = (rng.normal(0.0, 0.01, size=(self.config.lora_rank, self.config.feature_dim))).astype(np.float32)
        self.block_A = (rng.normal(0.0, 0.01, size=(n_block, self.config.lora_rank))).astype(np.float32)
        self.block_B = (rng.normal(0.0, 0.01, size=(self.config.lora_rank, self.config.feature_dim))).astype(np.float32)

    def train_sft(self, rows: List[Dict[str, Any]], epochs: int = 3, learning_rate: float = 0.3, batch_size: int = 8, seed: int = 42) -> List[Dict[str, float]]:
        if not rows:
            raise ValueError("SFT rows are empty")

        self._prepare_label_space(rows)
        rng = np.random.default_rng(seed)

        type_to_idx = {v: i for i, v in enumerate(self.edit_types)}
        block_to_idx = {v: i for i, v in enumerate(self.target_blocks)}

        features: List[np.ndarray] = []
        type_labels: List[int] = []
        block_labels: List[int] = []

        memory_rows: List[Dict[str, Any]] = []
        for row in rows:
            x = self.encode_context(
                task_summary=str(row.get("task_summary") or row.get("task_family") or ""),
                parent_workflow=str(row.get("parent_workflow") or ""),
                execution_feedback=row.get("execution_feedback") or {},
                blame=row.get("blame") or {},
            )
            features.append(x)
            chosen = row.get("chosen_edit") or {}
            type_labels.append(type_to_idx.get(str(chosen.get("edit_type", "unknown")), 0))
            block_labels.append(block_to_idx.get(str(chosen.get("target_block", "b0")), 0))
            memory_rows.append(
                {
                    "context": row,
                    "edit_type": str(chosen.get("edit_type", "unknown")),
                    "target_block": str(chosen.get("target_block", "b0")),
                    "edit_text": str(chosen.get("edit_text", "")),
                    "expected_benefit": f"Estimated delta_score={chosen.get('delta_score', 'N/A')}",
                }
            )

        X = np.stack(features, axis=0)
        y_type = np.array(type_labels, dtype=np.int64)
        y_block = np.array(block_labels, dtype=np.int64)

        self.memory_embeddings = X.copy()
        self.memory_outputs = memory_rows

        n = X.shape[0]
        history: List[Dict[str, float]] = []

        for ep in range(1, epochs + 1):
            indices = rng.permutation(n)
            total_loss = 0.0
            correct_type = 0
            correct_block = 0

            for start in range(0, n, batch_size):
                bidx = indices[start : start + batch_size]
                xb = X[bidx]
                tb = y_type[bidx]
                bb = y_block[bidx]

                for i in range(len(bidx)):
                    x = xb[i]
                    t = int(tb[i])
                    b = int(bb[i])

                    # Type head update.
                    Wt = self._effective_type_w()
                    logits_t = np.matmul(Wt, x)
                    pt = self._softmax(logits_t)
                    loss_t = -np.log(max(pt[t], 1e-8))
                    grad_logits_t = pt
                    grad_logits_t[t] -= 1.0

                    grad_Wt = np.outer(grad_logits_t, x)
                    grad_A_t = np.matmul(grad_Wt, self.type_B.T)
                    grad_B_t = np.matmul(self.type_A.T, grad_Wt)
                    self.type_A -= learning_rate * grad_A_t.astype(np.float32)
                    self.type_B -= learning_rate * grad_B_t.astype(np.float32)

                    # Block head update.
                    Wb = self._effective_block_w()
                    logits_b = np.matmul(Wb, x)
                    pb = self._softmax(logits_b)
                    loss_b = -np.log(max(pb[b], 1e-8))
                    grad_logits_b = pb
                    grad_logits_b[b] -= 1.0

                    grad_Wb = np.outer(grad_logits_b, x)
                    grad_A_b = np.matmul(grad_Wb, self.block_B.T)
                    grad_B_b = np.matmul(self.block_A.T, grad_Wb)
                    self.block_A -= learning_rate * grad_A_b.astype(np.float32)
                    self.block_B -= learning_rate * grad_B_b.astype(np.float32)

                    total_loss += float(loss_t + loss_b)
                    correct_type += int(int(np.argmax(logits_t)) == t)
                    correct_block += int(int(np.argmax(logits_b)) == b)

            history.append(
                {
                    "epoch": float(ep),
                    "loss": total_loss / max(n, 1),
                    "type_acc": correct_type / max(n, 1),
                    "block_acc": correct_block / max(n, 1),
                }
            )

        return history

    def _predict_type_block(self, x: np.ndarray) -> Tuple[str, str]:
        logits_t = np.matmul(self._effective_type_w(), x)
        logits_b = np.matmul(self._effective_block_w(), x)
        type_idx = int(np.argmax(logits_t)) if len(logits_t) else 0
        block_idx = int(np.argmax(logits_b)) if len(logits_b) else 0
        edit_type = self.edit_types[type_idx] if self.edit_types else "unknown"
        target_block = self.target_blocks[block_idx] if self.target_blocks else "b0"
        return edit_type, target_block

    def score_candidate(self, context_row: Dict[str, Any], edit_payload: Dict[str, Any]) -> float:
        x = self.encode_context(
            task_summary=str(context_row.get("task_summary") or context_row.get("task_family") or ""),
            parent_workflow=str(context_row.get("parent_workflow") or ""),
            execution_feedback=context_row.get("execution_feedback") or {},
            blame=context_row.get("blame") or {},
        )
        edit_type = str(edit_payload.get("edit_type", "unknown"))
        target_block = str(edit_payload.get("target_block", "b0"))

        et_idx = self.edit_types.index(edit_type) if edit_type in self.edit_types else 0
        tb_idx = self.target_blocks.index(target_block) if target_block in self.target_blocks else 0

        logits_t = np.matmul(self._effective_type_w(), x)
        logits_b = np.matmul(self._effective_block_w(), x)
        pt = self._softmax(logits_t)
        pb = self._softmax(logits_b)
        return float(pt[et_idx] * pb[tb_idx])

    def _nearest_memory(self, x: np.ndarray, desired_type: str = "", desired_block: str = "") -> Dict[str, Any]:
        if self.memory_embeddings.shape[0] == 0:
            return {
                "edit_type": desired_type or "insert_block",
                "target_block": desired_block or "b1",
                "edit_text": "Insert a verification step before final answer synthesis.",
                "expected_benefit": "Improve robustness by adding explicit validation.",
            }

        sims = np.matmul(self.memory_embeddings, x)
        order = np.argsort(-sims)
        for idx in order:
            item = self.memory_outputs[int(idx)]
            if desired_type and item.get("edit_type") != desired_type:
                continue
            if desired_block and item.get("target_block") != desired_block:
                continue
            return item
        return self.memory_outputs[int(order[0])]

    def generate_proposal(self, context_row: Dict[str, Any]) -> Dict[str, Any]:
        x = self.encode_context(
            task_summary=str(context_row.get("task_summary") or context_row.get("task_family") or ""),
            parent_workflow=str(context_row.get("parent_workflow") or ""),
            execution_feedback=context_row.get("execution_feedback") or {},
            blame=context_row.get("blame") or {},
        )
        edit_type, target_block = self._predict_type_block(x)
        nearest = self._nearest_memory(x, desired_type=edit_type, desired_block=target_block)

        edit_text = str(nearest.get("edit_text") or "")
        if not edit_text:
            edit_text = f"Apply a {edit_type} modification on {target_block} to address the attributed failure."

        return {
            "edit_type": edit_type,
            "target_block": target_block,
            "edit_text": edit_text,
            "expected_benefit": str(nearest.get("expected_benefit") or "Improve the next-round score under current failure mode."),
        }

    def valid_proposal_format(self, proposal: Dict[str, Any]) -> bool:
        required = {"edit_type", "target_block", "edit_text", "expected_benefit"}
        if not required.issubset(set(proposal.keys())):
            return False
        return all(str(proposal[k]).strip() for k in required)

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.savez(
            out / "revision_model.npz",
            base_type_w=self.base_type_w,
            base_block_w=self.base_block_w,
            type_A=self.type_A,
            type_B=self.type_B,
            block_A=self.block_A,
            block_B=self.block_B,
            memory_embeddings=self.memory_embeddings,
        )

        with open(out / "revision_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(self.config),
                    "edit_types": self.edit_types,
                    "target_blocks": self.target_blocks,
                    "memory_outputs": self.memory_outputs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, model_dir: str) -> "WorkflowRevisionModel":
        model_path = Path(model_dir)
        with open(model_path / "revision_model_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        config = RevisionModelConfig(**meta.get("config", {}))
        model = cls(config)
        model.edit_types = list(meta.get("edit_types", []))
        model.target_blocks = list(meta.get("target_blocks", []))
        model.memory_outputs = list(meta.get("memory_outputs", []))

        arr = np.load(model_path / "revision_model.npz")
        model.base_type_w = arr["base_type_w"].astype(np.float32)
        model.base_block_w = arr["base_block_w"].astype(np.float32)
        model.type_A = arr["type_A"].astype(np.float32)
        model.type_B = arr["type_B"].astype(np.float32)
        model.block_A = arr["block_A"].astype(np.float32)
        model.block_B = arr["block_B"].astype(np.float32)
        model.memory_embeddings = arr["memory_embeddings"].astype(np.float32)
        return model


def build_revision_prompt(row: Dict[str, Any]) -> str:
    return (
        "Task:\n"
        f"{row.get('task_summary') or row.get('task_family') or ''}\n\n"
        "Current workflow:\n"
        f"{row.get('parent_workflow') or ''}\n\n"
        "Execution feedback:\n"
        f"{json.dumps(row.get('execution_feedback') or {}, ensure_ascii=False)}\n\n"
        "Blame signal:\n"
        f"- target block: {(row.get('blame') or {}).get('target_block', '')}\n"
        f"- blame type: {(row.get('blame') or {}).get('blame_type', '')}\n"
        f"- rationale: {(row.get('blame') or {}).get('blame_rationale', '')}\n\n"
        "Instruction:\n"
        "Propose the next local workflow edit that is most likely to improve performance.\n"
        "Return a concise structured edit proposal."
    )
