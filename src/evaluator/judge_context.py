"""Runtime judge-context construction for AgriEnv rollouts."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from env import AgriTask
from .llm_judge import LLMJudge


class JudgeContextBuilder:
    """Build the exact compact context consumed by LLMJudge while rollout runs.

    The builder intentionally stores only judge-relevant state:
    task metadata, compact per-step tool outcomes, agent decision metadata,
    evidence ledger deltas, final answer metadata, and facts cited by the final
    answer. Raw env messages and full retrieval payloads stay out of this
    context.
    """

    VERSION = "judge_context_v1"

    def __init__(self, task: AgriTask, max_steps: int):
        self.task = task
        self.max_steps = max_steps
        self.compact_trajectory: List[Dict[str, Any]] = []
        self.agent_decision_metadata: List[Dict[str, Any]] = []
        self.visual_artifacts: List[Dict[str, Any]] = []
        self.final_cited_evidence_facts: List[Dict[str, Any]] = []
        self.final: Dict[str, Any] = {}
        self._accepted_seen: set[str] = set()
        self._rejected_seen: set[str] = set()
        self._rows_by_faq_id: Dict[str, Dict[str, Any]] = {}
        self._final_evidence_ids: set[str] = set()

    def record_step(
        self,
        *,
        step: int,
        action: str,
        params: Dict[str, Any],
        observation: Dict[str, Any],
        decision_metadata: Dict[str, Any],
        tool_success: Optional[bool],
        tool_error: Optional[str],
        visual_artifact: Dict[str, Any],
        agent_visual_artifacts: List[Any],
        agent_accepted_evidence: List[Dict[str, Any]],
        agent_rejected_evidence: List[Dict[str, Any]],
        reward: float,
        done: bool,
        truncated: bool,
    ) -> None:
        env_message = observation.get("env_message", "")
        self._index_evidence_rows(env_message)

        outcome = LLMJudge._extract_tool_outcome(str(env_message))
        compact_artifact = LLMJudge._compact_visual_artifact(visual_artifact)
        if compact_artifact:
            compact_artifact["step"] = step
            self.visual_artifacts.append(compact_artifact)

        self.compact_trajectory.append(
            {
                "step": step,
                "action": action,
                "params": LLMJudge._compact_params(action, params),
                "observation_type": observation.get("type"),
                "env_state": LLMJudge._compact_env_state(observation.get("env_state", {})),
                "evidence": outcome,
            }
        )

        accepted_delta = LLMJudge._compact_evidence_ledger_delta(
            agent_accepted_evidence,
            self._accepted_seen,
        )
        rejected_delta = LLMJudge._compact_evidence_ledger_delta(
            agent_rejected_evidence,
            self._rejected_seen,
        )
        self.agent_decision_metadata.append(
            {
                "step": step,
                "action": action,
                "decision_metadata": LLMJudge._compact_decision_metadata(decision_metadata),
                "tool_success": tool_success,
                "tool_error": tool_error,
                "visual_artifact": compact_artifact,
                "agent_visual_artifacts": [
                    LLMJudge._compact_visual_artifact(item)
                    for item in (agent_visual_artifacts or [])
                    if isinstance(item, dict)
                ],
                "tool_evidence_ledger": {
                    "accepted_delta": accepted_delta,
                    "rejected_delta": rejected_delta,
                    "accepted_total": len(LLMJudge._compact_evidence_ledger(agent_accepted_evidence)),
                    "rejected_total": len(LLMJudge._compact_evidence_ledger(agent_rejected_evidence)),
                },
                "reward": reward,
                "done": done,
                "truncated": truncated,
            }
        )

        if action == "submit_answer":
            self.record_final_submission(params)

    def record_final_submission(self, params: Dict[str, Any]) -> None:
        evidence_used = params.get("evidence_used") if isinstance(params, dict) else []
        self._final_evidence_ids = self._extract_final_evidence_ids(evidence_used)
        self.final = {
            "pred_answer": str(params.get("answer", "")) if isinstance(params, dict) else "",
            "confidence": params.get("confidence") if isinstance(params, dict) else None,
            "remaining_uncertainty": params.get("remaining_uncertainty") if isinstance(params, dict) else None,
            "evidence_used": LLMJudge._drop_heavy_fields(evidence_used or []),
        }
        self.final_cited_evidence_facts = self._build_final_cited_evidence_facts()

    def to_dict(self, final_answer: str = "") -> Dict[str, Any]:
        if final_answer and not self.final.get("pred_answer"):
            self.final["pred_answer"] = final_answer
        if not self.final_cited_evidence_facts and self._final_evidence_ids:
            self.final_cited_evidence_facts = self._build_final_cited_evidence_facts()

        return {
            "version": self.VERSION,
            "task": {
                "task_id": self.task.task_id,
                "query": self.task.query,
                "gt_answer": self.task.gt_answer,
                "image_path": self.task.image_path,
                "image_paths": list(self.task.image_paths or []),
                "metadata": LLMJudge._task_metadata_for_judge(self.task.metadata or {}),
                "max_steps": self.max_steps,
            },
            "compact_trajectory": self.compact_trajectory,
            "agent_decision_metadata": self.agent_decision_metadata,
            "final": self.final,
            "final_cited_evidence_facts": self.final_cited_evidence_facts,
            "visual_artifacts": self.visual_artifacts,
            "compression_policy": {
                "query_gt_pred": "complete in result/task/final, never string-truncated by judge context",
                "raw_env_message": "omitted; replaced by outcome-level evidence summary",
                "raw_tool_json": "omitted; only top evidence headers and final-cited facts retained",
                "evidence_ledger": "stored as cumulative totals plus per-step accepted/rejected deltas",
            },
        }

    def _index_evidence_rows(self, env_message: Any) -> None:
        payload = self._parse_payload(env_message)
        if not payload:
            return
        for rows in self._row_sources(payload):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                faq_id = str(row.get("faq_id") or "").strip()
                if faq_id and faq_id not in self._rows_by_faq_id:
                    self._rows_by_faq_id[faq_id] = row

    def _build_final_cited_evidence_facts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for evidence_id in sorted(self._final_evidence_ids):
            row = self._rows_by_faq_id.get(evidence_id)
            if not row:
                continue
            compact = LLMJudge._compact_final_cited_row(row)
            identity = LLMJudge._evidence_identity(compact)
            if identity in seen:
                continue
            seen.add(identity)
            out.append(compact)
        return out

    @staticmethod
    def _extract_final_evidence_ids(evidence_used: Any) -> set[str]:
        ids: set[str] = set()
        if not isinstance(evidence_used, list):
            return ids
        for item in evidence_used:
            if not isinstance(item, dict):
                continue
            value = item.get("source_id") or item.get("faq_id") or item.get("case_id")
            text = str(value or "").strip()
            if text:
                ids.add(text)
        return ids

    @staticmethod
    def _parse_payload(env_message: Any) -> Dict[str, Any]:
        if isinstance(env_message, dict):
            return env_message
        try:
            payload = json.loads(str(env_message))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _row_sources(payload: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        sources: List[List[Dict[str, Any]]] = []
        for key in (
            "candidates",
            "results",
            "answer_evidence",
            "recall_candidates",
            "case_results",
            "advices",
            "possible_diseases",
        ):
            rows = payload.get(key)
            if isinstance(rows, list):
                sources.append(rows)
        return sources
