"""
AgriEnv: MedAgentGym-style environment for agricultural QA.

将任务执行、工具调用、答案验证统一到 env.step(action, **params)。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

from tools import ToolRegistry


@dataclass
class AgriTask:
    """单个农业任务。"""

    task_id: str
    query: str
    gt_answer: str
    image_path: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    image_inputs: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    """环境中单步记录。"""

    step: int
    action: str
    params: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgriEnv:
    """面向知识库工具交互的轻量环境。"""

    def __init__(
        self,
        tasks: List[AgriTask],
        tool_registry: ToolRegistry,
        max_steps: int = 8,
    ):
        if not tasks:
            raise ValueError("AgriEnv requires at least one task")

        self.tasks = tasks
        self.tool_registry = tool_registry
        self.max_steps = max_steps

        self.current_task: Optional[AgriTask] = None
        self.current_index: Optional[int] = None
        self.step_count: int = 0
        self.done: bool = False
        self.history: List[StepRecord] = []

    def permitted_actions(self) -> List[str]:
        """返回允许动作。"""
        return list(self.tool_registry.tools.keys()) + ["submit_answer"]

    def action_schemas(self) -> List[Dict[str, Any]]:
        """返回动作 schema。"""
        schemas = self.tool_registry.list_tools()
        schemas.append(
            {
                "name": "submit_answer",
                "description": "Submit final evidence-grounded answer and finish task",
                "parameters": {
                    "type": "object",
                    "required": ["answer", "evidence_used", "confidence", "remaining_uncertainty"],
                    "properties": {
                        "answer": {"type": "string", "description": "Final answer"},
                        "evidence_used": {
                            "type": "array",
                            "description": (
                                "Evidence supporting the final answer. Each item should identify the source "
                                "(visual_tool, retrieval, management_tool, knowledge_base, diagnosis_tool, or tool_feedback) "
                                "and the claim it supports."
                            ),
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Confidence after considering available evidence.",
                        },
                        "remaining_uncertainty": {
                            "type": "string",
                            "description": "Important uncertainty or limitation in the final answer.",
                        },
                    },
                },
            }
        )
        return schemas

    def reset(self, task_id: int | str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置到新任务。"""
        if isinstance(task_id, int):
            if task_id < 0 or task_id >= len(self.tasks):
                raise IndexError(f"task_id out of range: {task_id}")
            self.current_index = task_id
            self.current_task = self.tasks[task_id]
        else:
            mapping = {t.task_id: idx for idx, t in enumerate(self.tasks)}
            if task_id not in mapping:
                raise KeyError(f"Unknown task_id: {task_id}")
            self.current_index = mapping[task_id]
            self.current_task = self.tasks[self.current_index]

        self.step_count = 0
        self.done = False
        self.history = []

        obs = {
            "type": "initial_observation",
            "steps_remaining": self.max_steps,
            "env_state": self._state_summary(current_action=None),
            "info": {
                "instruction": "",
                "task_goal": self.current_task.query,
                "task_id": self.current_task.task_id,
                "image_path": self.current_task.image_path,
                "image_paths": self.current_task.image_paths,
                "image_inputs": self.current_task.image_inputs,
                "metadata": self.current_task.metadata,
                "max_steps": self.max_steps,
            },
        }
        info = {
            "task_id": self.current_task.task_id,
            "query": self.current_task.query,
            "gt_answer": self.current_task.gt_answer,
        }
        return obs, info

    def _state_summary(self, current_action: Optional[str]) -> Dict[str, Any]:
        previous_actions = [r.action for r in self.history]
        actions_with_current = previous_actions + ([current_action] if current_action else [])
        tool_actions = [a for a in actions_with_current if a in self.tool_registry.tools]
        repeated_action = current_action is not None and current_action in previous_actions
        steps_remaining = max(0, self.max_steps - self.step_count)
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "steps_remaining": steps_remaining,
            "previous_actions": previous_actions,
            "actions_so_far": actions_with_current,
            "tool_calls_so_far": tool_actions,
            "tool_call_count": len(tool_actions),
            "current_action": current_action,
            "repeated_action": repeated_action,
            "must_submit_next": steps_remaining <= 1 and not self.done,
        }

    def step(self, action: str, **kwargs) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行单步动作。"""
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")

        if self.done:
            obs = {"type": "terminal", "env_message": "Episode already finished."}
            return obs, 0.0, True, False, {"message": "already_done"}

        self.step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {"action": action, "step": self.step_count}

        if action in self.tool_registry.tools:
            output = self.tool_registry.execute(action, **kwargs)
            if output.success:
                payload = output.data
                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload.setdefault("tool_name", action)
                    payload.setdefault("action", action)
                else:
                    payload = {
                        "tool_name": action,
                        "action": action,
                        "result": payload,
                    }
                obs = {
                    "type": "tool_feedback",
                    "env_message": json.dumps(payload, ensure_ascii=False),
                    "steps_remaining": max(0, self.max_steps - self.step_count),
                }
                info["tool_success"] = True
                info["tool_error"] = None
            else:
                error_payload = {
                    "tool_name": action,
                    "action": action,
                    "error": output.error,
                    "retrieval_diagnostics": {
                        "status": "error",
                    },
                }
                obs = {
                    "type": "tool_feedback",
                    "env_message": json.dumps(error_payload, ensure_ascii=False),
                    "steps_remaining": max(0, self.max_steps - self.step_count),
                }
                info["tool_success"] = False
                info["tool_error"] = output.error

        elif action == "request_info":
            question = kwargs.get("question", "")
            obs = {
                "type": "requested_info",
                "steps_remaining": max(0, self.max_steps - self.step_count),
                "env_message": (
                    f"Current query: {self.current_task.query}\n"
                    f"Task metadata: {json.dumps(self.current_task.metadata, ensure_ascii=False)}\n"
                    f"Your question: {question}"
                ),
            }

        elif action == "submit_answer":
            answer = str(kwargs.get("answer", "")).strip()
            score = self._score_answer(answer, self.current_task.gt_answer)
            reward = score["score"]
            self.done = True
            obs = {
                "type": "final_feedback",
                "steps_remaining": max(0, self.max_steps - self.step_count),
                "env_message": score["message"],
            }
            info.update(score)
            info["final_answer"] = answer
            info["gt_answer"] = self.current_task.gt_answer

        else:
            obs = {
                "type": "error_message",
                "steps_remaining": max(0, self.max_steps - self.step_count),
                "env_message": f"Unsupported action: {action}",
            }
            info["error"] = "unsupported_action"

        truncated = False
        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            truncated = True
            obs["truncated"] = True
            obs["terminal_reason"] = "max_steps_reached"
            obs["env_message"] = (
                f"{obs.get('env_message', '')}\n\n"
                f"Max steps reached ({self.max_steps}) after action={action}. Episode ended without submit_answer."
            )

        obs["env_state"] = self._state_summary(current_action=action)

        record = StepRecord(
            step=self.step_count,
            action=action,
            params=kwargs,
            observation=obs,
            reward=reward,
            done=self.done,
        )
        self.history.append(record)
        return obs, reward, self.done, truncated, info

    def get_history(self) -> List[Dict[str, Any]]:
        """返回可序列化历史。"""
        return [asdict(r) for r in self.history]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _score_answer(self, pred: str, gt: str) -> Dict[str, Any]:
        """No token-level scoring; defer to LLM judge."""
        return {
            "score": 0.0,
            "exact_match": None,
            "token_f1": None,
            "message": "Scoring deferred to LLM judge.",
        }
