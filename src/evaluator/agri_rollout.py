"""
Rollout runner for AgriEnv + LLMAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from agent import LLMAgent
from env import AgriEnv
from .judge_context import JudgeContextBuilder


@dataclass
class RolloutResult:
    task_id: str
    query: str
    gt_answer: str
    final_answer: str
    reward: float
    success: bool
    exact_match: Optional[bool]
    token_f1: Optional[float]
    steps: int
    trajectory: List[Dict[str, Any]]
    decision_trace: List[Dict[str, Any]]
    image_path: Optional[str]
    image_paths: List[str]
    metadata: Dict[str, Any]
    visual_artifacts: List[Dict[str, Any]]
    judge_context: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgriRolloutRunner:
    def __init__(self, env: AgriEnv, agent: LLMAgent, max_steps: int = 8):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.results: List[RolloutResult] = []

    def run_task(self, task_idx: int) -> RolloutResult:
        obs, info = self.env.reset(task_idx)
        self.agent.reset()

        final_answer = ""
        reward = 0.0
        done = False
        exact_match = None
        token_f1 = None
        decision_trace: List[Dict[str, Any]] = []
        judge_context_builder = (
            JudgeContextBuilder(self.env.current_task, self.max_steps)
            if self.env.current_task is not None
            else None
        )

        for _ in range(self.max_steps):
            obs_before = obs
            action, params, raw = self.agent.act(
                obs=obs,
                permitted_actions=self.env.permitted_actions(),
                action_schemas=self.env.action_schemas(),
            )

            obs, reward, done, truncated, step_info = self.env.step(action, **params)
            decision_metadata = self._extract_decision_metadata(raw)
            step_visual_artifact = self._extract_visual_artifact(obs.get("env_message", ""))

            decision_trace.append(
                {
                    "step": len(decision_trace) + 1,
                    "obs_type_before": obs_before.get("type"),
                    "llm_raw": raw,
                    "decision_metadata": decision_metadata,
                    "action": action,
                    "params": params,
                    "obs_type_after": obs.get("type"),
                    "env_message": obs.get("env_message", ""),
                    "tool_success": step_info.get("tool_success"),
                    "tool_error": step_info.get("tool_error"),
                    "visual_artifact": step_visual_artifact,
                    "agent_visual_artifacts": list(getattr(self.agent, "visual_artifacts", [])),
                    "agent_accepted_evidence": list(getattr(self.agent, "accepted_evidence", [])),
                    "agent_rejected_evidence": list(getattr(self.agent, "rejected_evidence", [])),
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                }
            )
            if judge_context_builder is not None:
                judge_context_builder.record_step(
                    step=len(decision_trace),
                    action=action,
                    params=params,
                    observation=obs,
                    decision_metadata=decision_metadata,
                    tool_success=step_info.get("tool_success"),
                    tool_error=step_info.get("tool_error"),
                    visual_artifact=step_visual_artifact,
                    agent_visual_artifacts=list(getattr(self.agent, "visual_artifacts", [])),
                    agent_accepted_evidence=list(getattr(self.agent, "accepted_evidence", [])),
                    agent_rejected_evidence=list(getattr(self.agent, "rejected_evidence", [])),
                    reward=reward,
                    done=done,
                    truncated=truncated,
                )

            if action == "submit_answer":
                final_answer = str(params.get("answer", ""))
                exact_match = step_info.get("exact_match")
                token_f1 = step_info.get("token_f1")

            if done or truncated:
                break

        if not final_answer and self.env.current_task is not None:
            final_answer = ""

        result = RolloutResult(
            task_id=info["task_id"],
            query=info["query"],
            gt_answer=info["gt_answer"],
            final_answer=final_answer,
            reward=reward,
            success=reward >= 0.5,
            exact_match=exact_match,
            token_f1=token_f1,
            steps=len(self.env.history),
            trajectory=self.env.get_history(),
            decision_trace=decision_trace,
            image_path=self.env.current_task.image_path if self.env.current_task else None,
            image_paths=list(self.env.current_task.image_paths or []) if self.env.current_task else [],
            metadata=dict(self.env.current_task.metadata or {}) if self.env.current_task else {},
            visual_artifacts=self._collect_visual_artifacts(self.env.get_history()),
            judge_context=judge_context_builder.to_dict(final_answer=final_answer) if judge_context_builder else {},
            timestamp=datetime.now().isoformat(),
        )

        self.results.append(result)
        return result

    @staticmethod
    def _extract_visual_artifact(env_message: Any) -> Dict[str, Any]:
        if not env_message:
            return {}
        if isinstance(env_message, dict):
            payload = env_message
        else:
            try:
                payload = json.loads(str(env_message))
            except Exception:
                return {}
        if not isinstance(payload, dict) or not payload.get("generated_image_path"):
            return {}
        return {
            "tool_name": payload.get("tool_name"),
            "action": payload.get("action"),
            "operation": payload.get("operation"),
            "source_image_path": payload.get("source_image_path"),
            "generated_image_path": payload.get("generated_image_path"),
            "bbox_pixels": payload.get("bbox_pixels") or payload.get("crop_bbox_pixels") or payload.get("boxes_pixels"),
            "bbox_normalized": payload.get("bbox_normalized") or payload.get("boxes_normalized"),
        }

    @classmethod
    def _collect_visual_artifacts(cls, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        for step in trajectory:
            observation = step.get("observation") or {}
            artifact = cls._extract_visual_artifact(observation.get("env_message", ""))
            if artifact:
                artifact["step"] = step.get("step")
                artifacts.append(artifact)
        return artifacts

    @staticmethod
    def _extract_decision_metadata(raw: str) -> Dict[str, Any]:
        try:
            payload = json.loads(raw.strip())
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(raw[start:end + 1])
                except Exception:
                    return {}
            else:
                return {}
        if not isinstance(payload, dict):
            return {}
        return {
            key: payload.get(key)
            for key in (
                "working_hypothesis",
                "evidence_gap",
                "accepted_evidence",
                "rejected_evidence",
                "why_this_tool",
                "expected_contribution",
            )
            if key in payload
        }

    def run_range(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Dict[str, Any]:
        if end_idx is None:
            end_idx = len(self.env.tasks)

        for idx in range(start_idx, end_idx):
            self.run_task(idx)

        total = len(self.results)
        avg_reward = sum(r.reward for r in self.results) / total if total else 0.0
        success_rate = sum(1 for r in self.results if r.success) / total if total else 0.0

        return {
            "total": total,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
        }

    def save(self, output_dir: str, tag: str = "") -> Dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        result_file = output / f"env_rollout_results{suffix}.json"
        traj_file = output / f"env_rollout_trajectories{suffix}.jsonl"
        stats_file = output / f"env_rollout_stats{suffix}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)

        with open(traj_file, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps({"task_id": r.task_id, "trajectory": r.trajectory}, ensure_ascii=False) + "\n")

        total = len(self.results)
        stats = {
            "total": total,
            "avg_reward": (sum(r.reward for r in self.results) / total) if total else 0.0,
            "success_rate": (sum(1 for r in self.results if r.success) / total) if total else 0.0,
            "exact_match_rate": (
                sum(1 for r in self.results if r.exact_match is True)
                / max(1, sum(1 for r in self.results if r.exact_match is not None))
            ),
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return {
            "results_file": str(result_file),
            "trajectory_file": str(traj_file),
            "stats_file": str(stats_file),
        }
