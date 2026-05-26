"""
LLM-based Agricultural Agent

使用 VLM 进行推理、决策和生成答案
支持工具调用轨迹记录
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
import re
from pathlib import Path

from vlm import BaseVLM, VLMMessage, VLMConfig, VLMFactory, VLMType


VISUAL_AUGMENTATION_TOOLS = {"crop_image", "zoom_image", "draw_bounding_box"}


class AgentActionType(Enum):
    """Agent 动作类型"""
    THINK = "think"
    CALL_TOOL = "call_tool"
    RESPOND = "respond"
    ERROR = "error"


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentAction:
    """Agent 执行的单个动作"""
    action_type: AgentActionType
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    """Agent 执行的完整轨迹（trajectory）"""
    query: str
    query_image: Optional[str] = None
    actions: List[AgentAction] = field(default_factory=list)
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class LLMAgent:
    """
    基于 LLM 的 Agent
    
    使用 VLM 进行推理和决策
    """
    
    def __init__(
        self,
        vlm: BaseVLM,
        agent_id: str = "agent_001",
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        parse_retry: int = 2,
        max_prompt_images: int = -1,
        max_history_messages: int = 8,
    ):
        """
        初始化 Agent
        
        Args:
            vlm: Vision Language Model 实例
            agent_id: Agent ID
            system_prompt: 系统提示
            max_steps: 最大步数
        """
        self.vlm = vlm
        self.agent_id = agent_id
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_steps = max_steps
        self.parse_retry = parse_retry
        self.max_prompt_images = int(max_prompt_images)
        self.small_context_mode = self._is_small_context_model(getattr(self.vlm.config, "model_name", ""))
        self.max_history_messages = max(2, int(max_history_messages))
        if self.small_context_mode:
            self.max_history_messages = min(self.max_history_messages, 2)
        self.conversation_history: List[VLMMessage] = []
        self.evidence_memory: List[str] = []
        self.task_goal: str = ""
        self.working_hypothesis: str = "No hypothesis yet."
        self.action_memory: List[Dict[str, Any]] = []
        self.visual_artifacts: List[str] = []
        self.accepted_evidence: List[Dict[str, Any]] = []
        self.rejected_evidence: List[Dict[str, Any]] = []
        self.last_decision_payload: Dict[str, Any] = {}
        self.primary_image_path: str = ""
        self.primary_image_input: Any = None
        self.image_paths: List[str] = []
        self.image_inputs: List[Any] = []
        self.current_env_state: Dict[str, Any] = {}

    def act(
        self,
        obs: Dict[str, Any],
        permitted_actions: List[str],
        action_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any], str]:
        """
        MedAgentGym 风格动作接口。

        输入 observation，输出 action + params。
        """
        self.current_env_state = obs.get("env_state") or {}
        if not self.conversation_history:
            system_message = self._build_action_system_prompt(
                permitted_actions=permitted_actions,
                action_schemas=action_schemas,
            )
            self.conversation_history.append(VLMMessage(role="system", content=system_message))

            if "info" in obs:
                instruction = obs["info"].get("instruction", "")
                task_goal = obs["info"].get("task_goal", "")
                self.task_goal = str(task_goal)
                metadata = self._metadata_for_prompt(obs["info"].get("metadata", {}))
                image_path = obs["info"].get("image_path", "")
                image_paths = obs["info"].get("image_paths", []) or []
                image_inputs = obs["info"].get("image_inputs", []) or []
                self.primary_image_path = str(image_path or "")
                self.primary_image_input = image_inputs[0] if image_inputs else None
                self.image_paths = [str(x) for x in image_paths]
                self.image_inputs = list(image_inputs)
                max_steps = obs["info"].get("max_steps", self.max_steps)
                instruction_block = f"Task instruction:\n{instruction}\n\n" if str(instruction).strip() else ""
                user_text = (
                    f"{instruction_block}"
                    f"Current goal:\n{task_goal}\n\n"
                    f"Metadata:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
                    f"Primary image path: {image_path}\n"
                    f"All available image paths: {json.dumps(image_paths, ensure_ascii=False)}\n"
                    f"Max steps: {max_steps}."
                )
                user_content = self._message_with_images(user_text, self.image_paths, self.image_inputs)
            else:
                user_content = json.dumps(obs, ensure_ascii=False)

            self.conversation_history.append(VLMMessage(role="user", content=user_content))
        else:
            env_message = obs.get("env_message", "")
            if isinstance(env_message, (dict, list)):
                env_message = json.dumps(env_message, ensure_ascii=False)
            obs_type = obs.get("type", "")
            steps_remaining = obs.get("steps_remaining")
            env_state = obs.get("env_state") or {}

            if obs_type == "tool_feedback":
                summary = self._summarize_tool_feedback(env_message)
                evidence_selection = self._select_evidence(env_message)
                self.accepted_evidence.extend(evidence_selection.get("accepted", []))
                self.rejected_evidence.extend(evidence_selection.get("rejected", []))
                self.accepted_evidence = self.accepted_evidence[-8:]
                self.rejected_evidence = self.rejected_evidence[-8:]
                if summary:
                    self.evidence_memory.append(summary)
                    if len(self.evidence_memory) > 8:
                        self.evidence_memory = self.evidence_memory[-8:]
                    self.working_hypothesis = self._update_working_hypothesis(summary)

                evidence_block = "\n".join([f"- {x}" for x in self.evidence_memory[-5:]]) or "- (no evidence yet)"
                memory_block = self._format_memory_block(env_state)
                feedback_text = (
                    "Tool feedback received. Update your working hypothesis from evidence, then choose the next best action.\n"
                    f"Steps remaining after this feedback: {steps_remaining}\n"
                    f"{memory_block}\n\n"
                    f"Latest tool feedback summary:\n{summary}\n\n"
                    f"Evidence selection from latest feedback:\n{json.dumps(evidence_selection, ensure_ascii=False)}\n\n"
                    "Tool-derived evidence ledger:\n"
                    f"- accepted_evidence: {json.dumps(self.accepted_evidence[-8:], ensure_ascii=False)}\n"
                    f"- rejected_evidence: {json.dumps(self.rejected_evidence[-8:], ensure_ascii=False)}\n"
                    "Use accepted_evidence as support. Do not cite rejected_evidence as support; mention it only as a limitation or contradiction.\n\n"
                    f"Accumulated evidence:\n{evidence_block}\n\n"
                    f"{self._tool_guide_text(permitted_actions, compact=self.small_context_mode)}\n"
                    f"{self._tool_rules_text(compact=self.small_context_mode)}"
                )
                if self.small_context_mode:
                    feedback_text = (
                        "Tool feedback received. Update the hypothesis and choose the next action.\n"
                        f"Steps remaining: {steps_remaining}\n"
                        f"{memory_block}\n\n"
                        f"Latest tool feedback summary:\n{summary[:900]}\n\n"
                        f"accepted_evidence: {json.dumps(self.accepted_evidence[-3:], ensure_ascii=False)}\n"
                        f"rejected_evidence: {json.dumps(self.rejected_evidence[-3:], ensure_ascii=False)}\n"
                        "Use accepted evidence as support; rejected evidence is only a limitation.\n\n"
                        f"Accumulated evidence:\n{evidence_block[:600]}\n\n"
                        f"{self._tool_guide_text(permitted_actions, compact=True)}\n"
                        f"{self._tool_rules_text(compact=True)}"
                    )
                visual_artifact = self._extract_visual_artifact(env_message)
                generated_image_path = visual_artifact.get("generated_image_path")
                if generated_image_path:
                    self.visual_artifacts.append(generated_image_path)
                    feedback_text += (
                        "\n\nGenerated visual artifact is attached as the image input for this turn. "
                        f"operation={visual_artifact.get('operation')}; path={generated_image_path}; "
                        "inspect this generated crop/zoom/annotation directly and update visual observations before choosing the next action."
                    )
                self.conversation_history.append(
                    VLMMessage(
                        role="user",
                        content=self._message_with_optional_image(feedback_text, generated_image_path),
                    )
                )
            else:
                self.conversation_history.append(
                    VLMMessage(
                        role="user",
                        content=(
                            f"Environment feedback:\n{env_message}\n\n"
                            f"{self._format_memory_block(env_state)}\n"
                            f"{self._tool_guide_text(permitted_actions, compact=self.small_context_mode)}\n"
                            f"{self._tool_rules_text(compact=self.small_context_mode)}"
                        ),
                    )
                )

        action = "request_info"
        params: Dict[str, Any] = {}
        raw_text = ""

        validation_error = ""
        for _ in range(self.parse_retry + 2):
            response = self.vlm.generate(self._messages_for_generation())
            raw_text = response.text

            action, params = self._parse_action_response(
                response.text,
                permitted_actions=permitted_actions,
            )

            self.conversation_history.append(VLMMessage(role="assistant", content=response.text))
            validation_error = self._validate_next_action(action, params)
            if action != "error" and not validation_error:
                break

            correction = validation_error or "Invalid output format."
            if self.small_context_mode:
                correction_content = (
                    f"{correction}\n"
                    "Return one valid compact JSON object only: "
                    '{"action":"<allowed_action>","params":{},"working_hypothesis":"<=25 words",'
                    '"evidence_gap":"<=15 words","why_this_tool":"<=15 words",'
                    '"expected_contribution":"<=15 words","accepted_evidence":[],"rejected_evidence":[]}'
                )
            else:
                correction_content = (
                    f"{correction}\n"
                    "Choose a different valid next action. Return compact JSON only, with action and params first:\n"
                    '{"action": "<action_name>", "params": {"key": "value"}, '
                    '"working_hypothesis": "max 40 words", "evidence_gap": "max 25 words", '
                    '"why_this_tool": "why this tool/action is needed now", "expected_contribution": "max 25 words", '
                    '"accepted_evidence": [], "rejected_evidence": []}'
                )
            self.conversation_history.append(VLMMessage(role="user", content=correction_content))

        if action == "error":
            action = "request_info"
            params = {"reason": "parse_failed", "raw": raw_text[:1000]}

        params = self._repair_tool_params(action, params)

        if validation_error and self._validation_requests_visual_augmentation(validation_error):
            action, params = self._fallback_visual_augmentation_action()

        if self.current_env_state.get("must_submit_next") and action != "submit_answer":
            action = "submit_answer"
            params = self._fallback_submit_params()

        if action == "submit_answer":
            final_validation = self._validate_next_action(action, params)
            if final_validation:
                params = self._repair_submit_params({"answer": self._repair_final_answer(str(params.get("answer", "")))})

        self.action_memory.append({"action": action, "params": params})

        return action, params, raw_text

    def _validate_next_action(self, action: str, params: Dict[str, Any]) -> str:
        if action == "error":
            return "Invalid output format."
        if self.current_env_state.get("must_submit_next") and action != "submit_answer":
            return (
                "This is the final available step. You must call submit_answer now using the best current evidence, "
                "including uncertainty if needed."
            )
        if action != "submit_answer" and self._is_duplicate_action(action, params):
            return (
                f"You repeated the same action and params: {action}. "
                "Repeated tool calls do not add evidence value. Choose a different tool/params or submit a cautious answer."
            )
        if action != "submit_answer":
            why = str(self.last_decision_payload.get("why_this_tool") or "").strip()
            gap = str(self.last_decision_payload.get("evidence_gap") or "").strip()
            if not why or not gap:
                return (
                    "Every tool call requires evidence-based metadata: include evidence_gap and why_this_tool "
                    "to explain what evidence is missing and why this tool is the right next call."
                )
        evidence_tools = {
            "triage_agri_issue",
            "visual_retrieve",
            "retrieve_multimodal_evidence",
            "search_knowledge_base",
            "retrieve_by_embedding",
            "crop_image",
            "zoom_image",
            "draw_bounding_box",
            "identify_disease",
            "get_management_advice",
        }
        used_evidence_tool = any(a.get("action") in evidence_tools for a in self.action_memory)
        if action == "submit_answer":
            answer = str(params.get("answer", ""))
            if not answer.strip():
                return "submit_answer requires a non-empty answer."
            if not used_evidence_tool:
                return (
                    "Do not submit yet. This benchmark requires at least one evidence-gathering tool result before submit_answer."
                )
            evidence_used = params.get("evidence_used")
            if not isinstance(evidence_used, list) or not evidence_used:
                return (
                    "submit_answer requires params.evidence_used as a non-empty list. "
                    "Each item must cite source_type/source_id and the claim supported by tool evidence."
                )
            if not self._evidence_cites_tool(evidence_used):
                return (
                    "params.evidence_used must cite at least one non-rejected tool result, not only initial_image, "
                    "metadata, prior_knowledge, recall_only evidence, or unsupported observation."
                )
            evidence_match_error = self._validate_evidence_used_against_ledger(evidence_used)
            if evidence_match_error:
                return evidence_match_error
            if not str(params.get("confidence", "")).strip():
                return "submit_answer requires params.confidence: low, medium, or high."
            if not str(params.get("remaining_uncertainty", "")).strip():
                return "submit_answer requires params.remaining_uncertainty."
        return ""

    def _validate_evidence_used_against_ledger(self, evidence_used: Any) -> str:
        if not isinstance(evidence_used, list):
            return ""

        used_ids = self._source_ids_from_evidence_used(evidence_used)
        accepted_ids = self._ledger_source_ids(self.accepted_evidence)
        rejected_ids = self._ledger_source_ids(self.rejected_evidence)

        rejected_used = sorted(used_ids & rejected_ids)
        if rejected_used:
            return (
                "params.evidence_used cites rejected evidence "
                f"{rejected_used[:3]}. Cite accepted tool evidence instead, or submit with lower confidence and explain the limitation."
            )

        if accepted_ids and not (used_ids & accepted_ids):
            return (
                "params.evidence_used must cite at least one accepted tool evidence id from the evidence ledger "
                f"(for example: {sorted(accepted_ids)[:3]})."
            )
        return ""

    @staticmethod
    def _source_ids_from_evidence_used(evidence_used: Any) -> set[str]:
        source_ids: set[str] = set()
        if not isinstance(evidence_used, list):
            return source_ids
        for item in evidence_used:
            if not isinstance(item, dict):
                continue
            for key in ("source_id", "faq_id", "case_id"):
                value = item.get(key)
                if value not in (None, ""):
                    source_ids.add(str(value).strip())
        return source_ids

    @staticmethod
    def _ledger_source_ids(items: Any) -> set[str]:
        source_ids: set[str] = set()
        if not isinstance(items, list):
            return source_ids
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("faq_id", "source_id", "case_id"):
                value = item.get(key)
                if value not in (None, ""):
                    source_ids.add(str(value).strip())
        return source_ids

    @staticmethod
    def _evidence_cites_tool(evidence_used: Any) -> bool:
        if not isinstance(evidence_used, list):
            return False
        tool_sources = {
            "tool",
            "tool_feedback",
            "visual_tool",
            "retrieval",
            "management_tool",
            "knowledge_base",
            "disease_tool",
            "multimodal_retrieval",
        }
        for item in evidence_used:
            if isinstance(item, dict):
                source_type = str(item.get("source_type") or item.get("type") or "").strip().lower()
                source_id = str(item.get("source_id") or item.get("source") or "").strip().lower()
                if source_type in tool_sources:
                    return True
                if any(token in source_id for token in ("tool", "faq", "kb", "retriev", "visual", "advice")):
                    return True
            elif isinstance(item, str):
                text = item.lower()
                if any(token in text for token in ("tool", "faq", "kb", "retriev", "visual", "advice")):
                    return True
        return False

    def _repair_tool_params(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params or {})
        if action in {"inspect_image_metadata", "visual_retrieve", "retrieve_multimodal_evidence", "retrieve_by_embedding"}:
            if self.primary_image_path and not params.get("image_path"):
                params["image_path"] = self.primary_image_path
            if self.image_paths and action in {"visual_retrieve", "retrieve_multimodal_evidence", "retrieve_by_embedding"}:
                params.setdefault("image_paths", self.image_paths)
        if action == "crop_image":
            if self.primary_image_path and not params.get("image_path"):
                params["image_path"] = self.primary_image_path
            if not params.get("bbox"):
                params["bbox"] = self._default_visual_bbox()
                params.setdefault("normalized", True)
            else:
                params["normalized"] = self._infer_normalized(params.get("bbox"), params.get("normalized"))
        elif action == "zoom_image":
            if self.primary_image_path and not params.get("image_path"):
                params["image_path"] = self.primary_image_path
            if not params.get("center"):
                params["center"] = self._default_visual_center()
                params.setdefault("normalized", True)
            else:
                params["normalized"] = self._infer_normalized(params.get("center"), params.get("normalized"))
            params.setdefault("scale", 2.5)
        elif action == "draw_bounding_box":
            if self.primary_image_path and not params.get("image_path"):
                params["image_path"] = self.primary_image_path
            if not params.get("boxes"):
                params["boxes"] = [self._default_visual_bbox()]
                params.setdefault("labels", ["region of interest"])
                params.setdefault("normalized", True)
            else:
                boxes = params.get("boxes") or []
                if isinstance(boxes, list) and len(boxes) == 4 and all(isinstance(v, (int, float, str)) for v in boxes):
                    params["boxes"] = [boxes]
                    first_box = boxes
                else:
                    first_box = boxes[0] if isinstance(boxes, list) and boxes else None
                params["normalized"] = self._infer_normalized(first_box, params.get("normalized"))
        return params

    @staticmethod
    def _infer_normalized(coords: Any, current: Any = None) -> bool:
        if isinstance(coords, (list, tuple)) and coords:
            try:
                values = [float(x) for x in coords]
            except Exception:
                return bool(current) if current is not None else True
            if values and all(0.0 <= x <= 1.0 for x in values):
                return True
            if any(abs(x) > 1.0 for x in values):
                return False
        return bool(current) if current is not None else True

    def _default_visual_center(self) -> List[float]:
        task = self.task_goal.lower()
        if any(token in task for token in ("bark", "scrap", "rubbed", "stripped", "trunk")):
            return [0.48, 0.45]
        return [0.5, 0.5]

    def _default_visual_bbox(self) -> List[float]:
        cx, cy = self._default_visual_center()
        half_w = 0.28
        half_h = 0.28
        return [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        ]

    @staticmethod
    def _validation_requests_visual_augmentation(message: str) -> bool:
        text = (message or "").lower()
        return (
            "crop_image" in text
            or "zoom_image" in text
            or "draw_bounding_box" in text
            or "visual artifact" in text
        )

    def _fallback_visual_augmentation_action(self) -> Tuple[str, Dict[str, Any]]:
        image_path = self.primary_image_path
        center = [0.5, 0.5]
        scale = 2.5
        return (
            "zoom_image",
            {
                "image_path": image_path,
                "center": center,
                "scale": scale,
                "normalized": True,
            },
        )

    def _is_duplicate_action(self, action: str, params: Dict[str, Any]) -> bool:
        key = (action, json.dumps(params, sort_keys=True, ensure_ascii=False))
        for item in self.action_memory:
            prev = (item.get("action"), json.dumps(item.get("params", {}), sort_keys=True, ensure_ascii=False))
            if prev == key:
                return True
        return False

    def _format_memory_block(self, env_state: Dict[str, Any]) -> str:
        recent_actions = self.action_memory[-5:]
        return (
            "Agent working memory:\n"
            f"- task_goal: {self.task_goal}\n"
            f"- working_hypothesis: {self.working_hypothesis}\n"
            f"- recent_actions: {json.dumps(recent_actions, ensure_ascii=False)}\n"
            f"- visual_artifacts: {json.dumps(self.visual_artifacts[-3:], ensure_ascii=False)}\n"
            f"- accepted_evidence: {json.dumps(self.accepted_evidence[-5:], ensure_ascii=False)}\n"
            f"- rejected_evidence: {json.dumps(self.rejected_evidence[-5:], ensure_ascii=False)}\n"
            f"- env_state: {json.dumps(env_state, ensure_ascii=False)}"
        )

    def _fallback_final_answer(self) -> str:
        return self._repair_final_answer("")

    def _fallback_submit_params(self) -> Dict[str, Any]:
        return self._repair_submit_params({"answer": self._fallback_final_answer()})

    def _repair_final_answer(self, answer: str) -> str:
        if answer.strip():
            return answer
        evidence = "; ".join(self.evidence_memory[-3:])
        hypothesis = self.working_hypothesis if self.working_hypothesis != "No hypothesis yet." else ""
        if hypothesis or evidence:
            return (
                f"Based on the available image and tool evidence, the most plausible answer is: "
                f"{(hypothesis or evidence)[:700]}. "
                "This answer is uncertain because the trajectory did not collect enough decisive evidence."
            )
        return "I could not determine a reliable answer from the available image and evidence."

    def _repair_submit_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        fixed = dict(params or {})
        fixed["answer"] = self._repair_final_answer(str(fixed.get("answer", "")))
        evidence_used = fixed.get("evidence_used")
        if not isinstance(evidence_used, list) or not evidence_used:
            evidence_items: List[Dict[str, str]] = []
            for item in self.accepted_evidence[-3:]:
                source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
                if not source_id:
                    continue
                evidence_items.append(
                    {
                        "source_type": "retrieval",
                        "source_id": source_id,
                        "claim_supported": str(item.get("reason") or item)[:240],
                    }
                )
            for idx, item in enumerate(self.evidence_memory[-3:], start=1):
                if evidence_items:
                    break
                evidence_items.append(
                    {
                        "source_type": "tool_feedback",
                        "source_id": f"recent_tool_{idx}",
                        "claim_supported": str(item)[:240],
                    }
                )
            if not evidence_items and self.visual_artifacts:
                evidence_items.append(
                    {
                        "source_type": "visual_tool",
                        "source_id": self.visual_artifacts[-1],
                        "claim_supported": "Generated visual artifact was available for final visual reasoning.",
                    }
                )
            if not evidence_items:
                evidence_items.append(
                    {
                        "source_type": "tool_feedback",
                        "source_id": "no_tool_evidence_available",
                        "claim_supported": "Forced final step without decisive tool evidence.",
                    }
                )
            fixed["evidence_used"] = evidence_items
        confidence = str(fixed.get("confidence", "")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            fixed["confidence"] = "medium" if self.evidence_memory or self.visual_artifacts else "low"
        if not str(fixed.get("remaining_uncertainty", "")).strip():
            fixed["remaining_uncertainty"] = (
                "Tool evidence is supportive but may be incomplete or analogical."
                if self.evidence_memory or self.visual_artifacts
                else "No decisive tool evidence was collected before the forced final answer."
            )
        return fixed

    def _update_working_hypothesis(self, summary: str) -> str:
        if not summary:
            return self.working_hypothesis
        if "retrieval_status=empty" in summary or "status=empty" in summary:
            return "Current evidence is weak or empty; broaden retrieval or inspect the image."
        return f"Best current evidence: {summary[:600]}"

    def _select_evidence(self, env_message: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            payload = json.loads(env_message)
        except Exception:
            return {"accepted": [], "rejected": []}
        if not isinstance(payload, dict):
            return {"accepted": [], "rejected": []}

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        rows = (
            payload.get("answer_evidence")
            or payload.get("candidates")
            or payload.get("results")
            or payload.get("advices")
            or payload.get("possible_diseases")
            or []
        )
        if not isinstance(rows, list):
            rows = []

        requested_species = str(payload.get("species") or "").lower()
        requested_issue = " ".join(
            str(payload.get(k) or "")
            for k in ("query", "issue", "symptom", "field")
        ).lower()

        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            faq_id = row.get("faq_id")
            species = str(row.get("species") or "").lower()
            issue_type = str(row.get("issue_type") or "")
            text = json.dumps(row, ensure_ascii=False).lower()
            relevance = row.get("relevance") or {}

            reasons = []
            mismatch = False
            should_use = relevance.get("should_agent_use")
            if requested_species and species and requested_species not in species and species not in requested_species:
                if not (set(requested_species.split()) & set(species.replace(",", " ").split())):
                    mismatch = True
                    reasons.append("species differs from current query hint; use only transferable symptoms/management")
            if "bark" in requested_issue and issue_type == "disease":
                reasons.append("bark scraping often needs wildlife/physical-damage check, not only disease")
            if "white" in requested_issue and "lichen" in text:
                reasons.append("mentions lichen; distinguish cosmetic lichen from disease")
            if "management_instructions" in text or row.get("advice"):
                reasons.append("contains management advice")
            if row.get("fused_score") is not None:
                reasons.append(f"fused_score={float(row.get('fused_score') or 0):.3f}")
            if row.get("image_score") is not None:
                reasons.append(f"image_score={float(row.get('image_score') or 0):.3f}")
            if relevance:
                reasons.append(f"use_as={relevance.get('use_as')}")
                if relevance.get("mismatch_reason"):
                    reasons.append(str(relevance.get("mismatch_reason")))

            entry = {
                "faq_id": faq_id,
                "species": row.get("species"),
                "issue_type": row.get("issue_type"),
                "reason": "; ".join(reasons)[:240] or "potentially relevant",
            }
            use_as = str(relevance.get("use_as") or "")
            mismatch_reason = str(relevance.get("mismatch_reason") or "")
            query_term_match = float(relevance.get("query_term_match") or 0.0)
            has_species_mismatch = "species_mismatch" in mismatch_reason or mismatch
            has_field_mismatch = "field_mismatch" in mismatch_reason
            no_query_overlap = "no_query_term_overlap" in mismatch_reason or query_term_match == 0.0
            severe_mismatch = (
                use_as == "recall_only"
                or should_use is False
                or (has_species_mismatch and no_query_overlap)
                or (has_field_mismatch and no_query_overlap)
            )
            if severe_mismatch:
                entry["evidence_role"] = "rejected"
                rejected.append(entry)
            elif should_use is True and use_as != "recall_only":
                entry["evidence_role"] = "analogical" if (has_species_mismatch or has_field_mismatch) else "direct"
                accepted.append(entry)
            elif has_species_mismatch or has_field_mismatch:
                entry["evidence_role"] = "analogical"
                accepted.append(entry)
            else:
                entry["evidence_role"] = "direct"
                accepted.append(entry)

        return {"accepted": accepted[:4], "rejected": rejected[:4]}

    def _message_with_optional_image(self, text: str, image_path: Optional[str], image_input: Any = None) -> Any:
        if image_input is not None:
            if self._is_usable_vlm_image(image_input):
                return [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": text},
                ]
            return text + "\n\n[Image omitted from VLM input because its dimensions/aspect ratio are invalid for the processor.]"
        if image_path:
            try:
                p = Path(str(image_path))
                if p.exists() and p.is_file():
                    if not self._is_usable_vlm_image_path(p):
                        return text + f"\n\n[Image path retained but not attached due to invalid dimensions: {p}]"
                    return [
                        {"type": "image", "image": self.vlm.encode_image(str(p))},
                        {"type": "text", "text": text},
                    ]
            except Exception:
                pass
        return text

    def _message_with_images(self, text: str, image_paths: List[str], image_inputs: List[Any]) -> Any:
        content: List[Dict[str, Any]] = []
        omitted = 0
        attached = 0
        max_images = self.max_prompt_images
        unlimited_images = max_images < 0

        for image in image_inputs:
            if not unlimited_images and attached >= max_images:
                omitted += 1
                continue
            if self._is_usable_vlm_image(image):
                content.append({"type": "image", "image": image})
                attached += 1
            else:
                omitted += 1

        if not content:
            for image_path in image_paths:
                if not unlimited_images and attached >= max_images:
                    omitted += 1
                    continue
                try:
                    p = Path(str(image_path))
                    if p.exists() and p.is_file() and self._is_usable_vlm_image_path(p):
                        content.append({"type": "image", "image": self.vlm.encode_image(str(p))})
                        attached += 1
                    elif image_path:
                        omitted += 1
                except Exception:
                    omitted += 1

        if content:
            note = ""
            if omitted:
                note = (
                    f"\n\n[{omitted} image(s) not directly attached to this prompt because of invalid dimensions, "
                    "unavailable files, or an explicit --max-prompt-images limit. Their paths remain listed above; use visual tools "
                    "with the relevant image_path/image_index when needed.]"
                )
            content.append({"type": "text", "text": text + note})
            return content

        if omitted:
            return (
                text
                + f"\n\n[{omitted} image(s) not directly attached because of invalid dimensions, unavailable files, or an explicit --max-prompt-images limit. "
                "Use listed paths with visual tools when needed.]"
            )
        return text

    def _messages_for_generation(self) -> List[VLMMessage]:
        """Return a compact generation view without mutating full trajectory history."""
        if len(self.conversation_history) <= 2:
            return self.conversation_history

        system_messages = [m for m in self.conversation_history[:1] if m.role == "system"]
        non_system = [m for m in self.conversation_history if m.role != "system"]
        recent = non_system[-self.max_history_messages :]

        first_non_system = non_system[0] if non_system else None
        if first_non_system is not None and not any(m is first_non_system for m in recent):
            recent = [self._compact_initial_message(first_non_system)] + recent

        compacted: List[VLMMessage] = []
        for idx, msg in enumerate(system_messages + recent):
            keep_images = idx == len(system_messages + recent) - 1
            view = self._strip_images_from_message(msg, keep_images=keep_images)
            if self.small_context_mode:
                view = self._truncate_generation_message(view, keep_images=keep_images)
            compacted.append(view)
        return compacted

    def _truncate_generation_message(self, msg: VLMMessage, keep_images: bool = False) -> VLMMessage:
        if msg.role == "system":
            limit = 1800
        elif msg.role == "assistant":
            limit = 500
        else:
            limit = 1100 if keep_images else 900

        if isinstance(msg.content, str):
            text = msg.content
            if len(text) <= limit:
                return msg
            return VLMMessage(role=msg.role, content=text[:limit] + "\n[truncated for small-context model]")

        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", ""))
                    if len(text) > limit:
                        item = dict(item)
                        item["text"] = text[:limit] + "\n[truncated for small-context model]"
                elif not keep_images and isinstance(item, dict) and item.get("type") == "image":
                    continue
                content.append(item)
            return VLMMessage(role=msg.role, content=content)
        return msg

    def _compact_initial_message(self, msg: VLMMessage) -> VLMMessage:
        text = self._message_text(msg.content)
        path_note = ""
        if self.image_paths:
            path_note = "\nAvailable image paths: " + json.dumps(self.image_paths, ensure_ascii=False)
        return VLMMessage(
            role=msg.role,
            content=(
                text[:800 if self.small_context_mode else 3000]
                + path_note
                + "\n[Original task image payloads were attached only in the initial turn to control context length. "
                "Use visual tools for further image inspection.]"
            ),
        )

    def _strip_images_from_message(self, msg: VLMMessage, keep_images: bool = False) -> VLMMessage:
        if keep_images or not isinstance(msg.content, list):
            return msg
        text = self._message_text(msg.content)
        return VLMMessage(role=msg.role, content=text + "\n[Older image payload omitted from repeated VLM context.]")

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return "\n".join([t for t in texts if t]).strip()
        return str(content)

    @staticmethod
    def _is_usable_vlm_image(image: Any) -> bool:
        size = getattr(image, "size", None)
        if not size or len(size) != 2:
            return True
        width, height = [int(x) for x in size]
        if width < 16 or height < 16:
            return False
        ratio = max(width / max(1, height), height / max(1, width))
        return ratio <= 20

    @classmethod
    def _is_usable_vlm_image_path(cls, path: Path) -> bool:
        try:
            from PIL import Image

            with Image.open(path) as img:
                return cls._is_usable_vlm_image(img)
        except Exception:
            return False

    @staticmethod
    def _extract_visual_artifact(env_message: str) -> Dict[str, Any]:
        try:
            payload = json.loads(env_message)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        value = payload.get("generated_image_path")
        if value:
            return {
                "generated_image_path": str(value),
                "operation": payload.get("operation") or payload.get("action"),
                "bbox_pixels": payload.get("bbox_pixels") or payload.get("crop_bbox_pixels") or payload.get("boxes_pixels"),
            }
        for key in ("result", "data"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested.get("generated_image_path"):
                return {
                    "generated_image_path": str(nested["generated_image_path"]),
                    "operation": nested.get("operation") or nested.get("action"),
                    "bbox_pixels": nested.get("bbox_pixels") or nested.get("crop_bbox_pixels") or nested.get("boxes_pixels"),
                }
        return {}

    @staticmethod
    def _is_small_context_model(model_name: str) -> bool:
        model_key = (model_name or "").lower().replace("_", "-")
        # DeepSeek-VL2 advertises a 4096-token text context in its HF config,
        # and vLLM rejects larger max_model_len values. Keep the compact agent
        # prompt/history path for this family so run_agent_inference remains stable.
        return "deepseek-vl2" in model_key

    def _metadata_for_prompt(self, metadata: Any) -> Any:
        if not self.small_context_mode or not isinstance(metadata, dict):
            return metadata

        keep_keys = {
            "source",
            "subset",
            "split",
            "category",
            "domain",
            "species",
            "entity",
            "plant",
            "crop",
            "image_paths",
        }
        compact: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key not in keep_keys and len(compact) >= 8:
                continue
            if isinstance(value, str):
                compact[key] = value[:220]
            elif isinstance(value, (int, float, bool)) or value is None:
                compact[key] = value
            elif isinstance(value, list):
                compact[key] = [str(x)[:120] for x in value[:4]]
            elif isinstance(value, dict):
                compact[key] = {str(k): str(v)[:120] for k, v in list(value.items())[:6]}
            elif key in keep_keys:
                compact[key] = str(value)[:160]
        return compact

    def _build_action_system_prompt(
        self,
        permitted_actions: List[str],
        action_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建动作决策提示词。"""
        if self.small_context_mode:
            return self._build_small_context_system_prompt(permitted_actions, action_schemas)

        schema_text = ""
        tool_set_text = ""
        tool_guide = self._tool_guide_text(permitted_actions)
        if action_schemas:
            compact_schemas = self._compact_action_schemas(action_schemas)
            schema_text = "\n\nAvailable tool schemas (compact):\n" + json.dumps(
                compact_schemas,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            callable_tools = [
                s.get("name")
                for s in action_schemas
                if isinstance(s, dict) and s.get("name") not in {"request_info", "submit_answer"}
            ]
            callable_tools = [str(x) for x in callable_tools if x]
            tool_set_text = "Callable tool set: " + ", ".join(callable_tools)

        return (
            "You are an agricultural tool-using agent. Each step must select exactly one allowed action and return JSON only.\n"
            f"Allowed actions: {', '.join(permitted_actions)}\n"
            f"{tool_set_text}\n"
            f"{tool_guide}\n"
            f"{self._tool_rules_text()}\n"
            "Required action schema:\n"
            "{"
            "\"action\": \"<action_name>\", "
            "\"params\": {...}, "
            "\"working_hypothesis\": \"current best diagnosis/cause and confidence\", "
            "\"evidence_gap\": \"what is still missing or uncertain\", "
            "\"accepted_evidence\": [\"tool result id or observation: why accepted\"], "
            "\"rejected_evidence\": [\"tool result id or observation: why rejected\"], "
            "\"why_this_tool\": \"why this action is useful now\", "
            "\"expected_contribution\": \"what this action should add to the trajectory\""
            "}"
            "\nFor submit_answer, params must be: {\"answer\": str, \"evidence_used\": [{\"source_type\": \"visual_tool|retrieval|management_tool|knowledge_base|tool_feedback\", \"source_id\": \"tool name or returned KB entry id\", \"claim_supported\": str}], \"confidence\": \"low|medium|high\", \"remaining_uncertainty\": str}."
            f"{schema_text}"
        )

    def _build_small_context_system_prompt(
        self,
        permitted_actions: List[str],
        action_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        schema_hint = ""
        if action_schemas:
            schema_hint = "\nTool params: " + json.dumps(
                self._compact_action_schema_hints(action_schemas),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return (
            "You are an agricultural tool-using agent. Return one compact JSON object only.\n"
            f"Allowed actions: {', '.join(permitted_actions)}\n"
            f"{self._tool_guide_text(permitted_actions, compact=True)}\n"
            f"{self._tool_rules_text(compact=True)}\n"
            "JSON keys: action, params, working_hypothesis, evidence_gap, accepted_evidence, rejected_evidence, why_this_tool, expected_contribution.\n"
            "For submit_answer params: answer, evidence_used, confidence, remaining_uncertainty."
            f"{schema_hint}"
        )

    @staticmethod
    def _tool_rules_text(compact: bool = False) -> str:
        if compact:
            return (
                "Tool-use rules:\n"
                "- Use an exact allowed action name; there is no call_tool wrapper.\n"
                "- Gather tool evidence before submit_answer; choose tools that reduce the evidence_gap.\n"
                "- For image QA, use retrieval/visual tools when useful; for multi-image tasks pass image_paths when comparing all images.\n"
                "- Track accepted_evidence and rejected_evidence; rejected evidence cannot support the answer.\n"
                "- Similar KB cases may be analogical evidence if you state uncertainty.\n"
                "- Do not repeat identical tool params. Submit when evidence is sufficient. JSON only."
            )
        return (
            "Tool-use rules:\n"
            "1) To use a tool, set action to the exact tool name, for example search_knowledge_base or retrieve_multimodal_evidence. There is no wrapper action named call_tool.\n"
            "2) Use tools to complete the agricultural consultation or QA task; do not submit from prior knowledge alone.\n"
            "3) Call at least one evidence-gathering tool before submit_answer. More tools are useful only if they reduce a real evidence_gap.\n"
            "4) Choose the smallest useful tool set: visual detail tools for unclear image regions, retrieval tools for similar cases, diagnosis/management tools for cause and treatment advice.\n"
            "5) Maintain an evidence ledger. accepted_evidence means a tool result adds useful direct or transferable information; rejected_evidence means a result is recall_only, should_agent_use=false, or has severe species/field mismatch.\n"
            "6) Rejected evidence must not support the final answer. If most retrieved evidence is rejected or weak, lower confidence, change the hypothesis, call a different useful tool, or answer cautiously with the limitation.\n"
            "7) Similar KB cases can be useful without exact species match; use them as analogical evidence only when you state what transfers and what remains uncertain. If evidence_used is mainly analogical, confidence should be low or medium, not high.\n"
            "8) After each tool result, update working_hypothesis, accepted_evidence, rejected_evidence, and evidence_gap. Do not keep an initial hypothesis if tool evidence contradicts it.\n"
            "9) Do not repeat the same tool with identical params unless why_this_tool states a new reason.\n"
            "10) submit_answer is valid only when params.evidence_used cites at least one accepted tool result when accepted evidence exists, and the answer directly addresses the user's query.\n"
            "11) For multi-image tasks, consider all listed images. Retrieval tools can receive image_paths with all available image references; use image_path only when intentionally focusing on one image.\n"
            "When tool evidence is sufficient, call submit_answer. Match the user's requested answer format; do not force a consultation template when the query asks for a short identification or one paragraph.\n"
            "Output must be a single JSON object with no extra text.\n"
            "Keep all non-answer metadata fields concise: working_hypothesis <= 40 words, evidence_gap/why_this_tool/expected_contribution <= 25 words."
        )

    @staticmethod
    def _tool_guide_text(permitted_actions: List[str], compact: bool = False) -> str:
        descriptions = {
            "inspect_image_metadata": "read image size/metadata only; use for dimensions, not diagnosis.",
            "crop_image": "create a crop of a specific region when symptoms, pests, lesions, or objects are localized.",
            "zoom_image": "create a zoomed view when the image detail is too small or unclear.",
            "draw_bounding_box": "mark regions of interest for later visual reasoning or trajectory inspection.",
            "triage_agri_issue": "route broad agricultural questions into likely issue types and useful search terms.",
            "search_knowledge_base": "text search AGMMU facts for symptoms, species, causes, pests, diseases, or management.",
            "visual_retrieve": "retrieve visually similar KB cases from the image embedding index; pass image_paths for multi-image tasks.",
            "retrieve_multimodal_evidence": "retrieve evidence using both query text and image similarity; good default for image QA; pass image_paths for multi-image tasks.",
            "retrieve_by_embedding": "semantic text/image retrieval over the mounted embedding index; pass image_paths for multi-image tasks.",
            "identify_disease": "look up candidate diseases or issue diagnoses from symptom and species text.",
            "get_management_advice": "look up treatment, prevention, or care advice for a suspected issue and species.",
        }
        if compact:
            short = {
                "inspect_image_metadata": "image metadata only",
                "crop_image": "crop a localized visual region",
                "zoom_image": "zoom small/unclear details",
                "draw_bounding_box": "mark visual evidence",
                "triage_agri_issue": "triage issue type/search terms",
                "search_knowledge_base": "text-search KB facts",
                "visual_retrieve": "image-similar KB cases; use image_paths for multi-image",
                "retrieve_multimodal_evidence": "text+image KB evidence; good default for image QA",
                "retrieve_by_embedding": "semantic/image embedding retrieval",
                "identify_disease": "candidate diseases/issues from symptoms",
                "get_management_advice": "management/treatment advice",
            }
            lines = ["Tool guide:"]
            for name in permitted_actions:
                if name in short:
                    lines.append(f"- {name}: {short[name]}")
            if any(name in permitted_actions for name in {"search_knowledge_base", "visual_retrieve", "retrieve_multimodal_evidence", "retrieve_by_embedding"}):
                lines.append("- faq_id appears only after retrieval and identifies a returned KB case.")
            return "\n".join(lines)

        lines = ["Tool guide:"]
        if any(name in permitted_actions for name in {"search_knowledge_base", "visual_retrieve", "retrieve_multimodal_evidence", "retrieve_by_embedding"}):
            lines.append(
                "- KB entries are agricultural case records returned by retrieval tools. "
                "If a tool returns a faq_id, treat it as that retrieved case id; do not invent faq_id values before a tool returns them."
            )
        for name in permitted_actions:
            if name in descriptions:
                lines.append(f"- {name}: {descriptions[name]}")
        return "\n".join(lines)

    @staticmethod
    def _compact_action_schema_hints(action_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hints: List[Dict[str, Any]] = []
        for schema in action_schemas:
            if not isinstance(schema, dict):
                continue
            name = schema.get("name")
            params = schema.get("parameters") or schema.get("input_schema") or {}
            required = set(params.get("required") or []) if isinstance(params, dict) else set()
            properties = params.get("properties") if isinstance(params, dict) else {}
            if isinstance(params, dict) and not isinstance(properties, dict):
                properties = {
                    k: v
                    for k, v in params.items()
                    if k not in {"type", "required", "properties", "additionalProperties"}
                }
            param_hint = {}
            if isinstance(properties, dict):
                for key, value in properties.items():
                    if isinstance(value, dict):
                        typ = value.get("type") or "any"
                        enum = value.get("enum")
                        marker = "*" if key in required else ""
                        if enum:
                            typ = "|".join([str(x) for x in enum[:6]]) if isinstance(enum, list) else str(enum)
                        param_hint[key] = f"{typ}{marker}"
                    else:
                        param_hint[key] = "any" + ("*" if key in required else "")
            hints.append({"name": name, "params": param_hint})
        return hints

    @staticmethod
    def _compact_action_schemas(action_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for schema in action_schemas:
            if not isinstance(schema, dict):
                continue
            item: Dict[str, Any] = {"name": schema.get("name")}
            description = str(schema.get("description") or "")
            if description:
                item["description"] = description[:180]

            params = schema.get("parameters") or schema.get("input_schema") or {}
            if isinstance(params, dict):
                required = set(params.get("required") or [])
                properties = params.get("properties")
                if not isinstance(properties, dict):
                    properties = {
                        k: v
                        for k, v in params.items()
                        if k not in {"type", "required", "properties", "additionalProperties"}
                    }
                compact_params: Dict[str, Any] = {}
                for key, value in properties.items():
                    if not isinstance(value, dict):
                        compact_params[key] = {"type": type(value).__name__}
                        continue
                    spec: Dict[str, Any] = {}
                    if value.get("type") is not None:
                        spec["type"] = value.get("type")
                    if value.get("enum") is not None:
                        enum_values = value.get("enum")
                        spec["enum"] = enum_values[:12] if isinstance(enum_values, list) else enum_values
                    if key in required:
                        spec["required"] = True
                    if value.get("default") is not None:
                        spec["default"] = value.get("default")
                    compact_params[key] = spec or {"type": "any"}
                if compact_params:
                    item["params"] = compact_params
            compact.append(item)
        return compact

    def _summarize_tool_feedback(self, env_message: str) -> str:
        """Compress tool feedback into a short evidence summary for iterative reasoning."""
        try:
            payload = json.loads(env_message)
        except Exception:
            return str(env_message)[:400]

        if not isinstance(payload, dict):
            return str(payload)[:400]

        parts: List[str] = []

        if "query" in payload:
            parts.append(f"query={payload.get('query')}")
        if "species" in payload:
            parts.append(f"species={payload.get('species')}")
        if "field" in payload and payload.get("field") is not None:
            parts.append(f"field={payload.get('field')}")
        if "broad_issue_types" in payload:
            parts.append("broad_issue_types=" + ", ".join([str(x) for x in payload.get("broad_issue_types", [])[:4]]))
        if "recommended_search_terms" in payload:
            parts.append("recommended_search_terms=" + " | ".join([str(x) for x in payload.get("recommended_search_terms", [])[:4]]))

        answer_evidence = payload.get("answer_evidence") or []
        if isinstance(answer_evidence, list) and answer_evidence:
            snippets = []
            for c in answer_evidence[:3]:
                if not isinstance(c, dict):
                    continue
                facts = c.get("all_facts") or c.get("facts") or []
                fact_bits = []
                for f in facts:
                    if not isinstance(f, dict):
                        continue
                    if f.get("field") in {"disease_or_issue_identification", "insect_or_pest", "symptom_description", "management_instructions", "image_description"}:
                        fact_bits.append(f"{f.get('field')}={str(f.get('answer', ''))[:70]}")
                    if len(fact_bits) >= 3:
                        break
                snippets.append(f"{c.get('faq_id')}:{c.get('species')}:{' / '.join(fact_bits)}")
            if snippets:
                parts.append("answer_evidence=" + " | ".join(snippets))

        results = payload.get("results") or []
        if isinstance(results, list) and results:
            top = results[:3]
            snippets = []
            for r in top:
                if not isinstance(r, dict):
                    continue
                faq = r.get("faq_id", "")
                ans = r.get("answer") or r.get("advice") or r.get("disease") or ""
                if not ans:
                    mf = r.get("matched_facts") or []
                    if isinstance(mf, list) and mf and isinstance(mf[0], dict):
                        ans = mf[0].get("answer", "")
                snippets.append(f"{faq}:{str(ans)[:90]}")
            if snippets:
                parts.append("top_results=" + " | ".join(snippets))

        case_evidence = payload.get("case_evidence") or []
        if isinstance(case_evidence, dict):
            case_evidence = [case_evidence]
        if isinstance(case_evidence, list) and case_evidence:
            snippets = []
            for c in case_evidence[:3]:
                if not isinstance(c, dict):
                    continue
                facts = c.get("facts") or []
                diagnosis = ""
                management = ""
                for f in facts:
                    if not isinstance(f, dict):
                        continue
                    field = f.get("field", "")
                    if field == "disease_or_issue_identification" and not diagnosis:
                        diagnosis = f.get("answer", "")
                    if field == "insect_or_pest" and not diagnosis:
                        diagnosis = f.get("answer", "")
                    if field == "management_instructions" and not management:
                        management = f.get("answer", "")
                snippets.append(
                    f"{c.get('faq_id')}:{c.get('species')}:{c.get('issue_type')}:{str(diagnosis or management)[:100]}"
                )
            if snippets:
                parts.append("case_evidence=" + " | ".join(snippets))

        candidates = payload.get("candidates") or []
        if isinstance(candidates, list) and candidates:
            snippets = []
            for c in candidates[:3]:
                if not isinstance(c, dict):
                    continue
                faq = c.get("faq_id", "")
                fused = c.get("fused_score", 0.0)
                issue_type = c.get("issue_type", "")
                txt = (c.get("text_evidence") or {}).get("answer", "")
                if not txt:
                    facts = c.get("facts") or []
                    if facts and isinstance(facts[0], dict):
                        txt = facts[0].get("answer", "")
                relevance = c.get("relevance") or {}
                rel_bits = []
                if isinstance(relevance, dict):
                    if relevance.get("use_as"):
                        rel_bits.append(f"use={relevance.get('use_as')}")
                    if relevance.get("mismatch_reason"):
                        rel_bits.append(f"mismatch={relevance.get('mismatch_reason')}")
                    if relevance.get("should_agent_use") is False:
                        rel_bits.append("do_not_use_as_answer")
                rel = ":" + ",".join(rel_bits) if rel_bits else ""
                snippets.append(f"{faq}:score={fused:.3f}:{issue_type}{rel}:{str(txt)[:70]}")
            if snippets:
                parts.append("top_candidates=" + " | ".join(snippets))

        case_results = payload.get("case_results") or []
        if isinstance(case_results, list) and case_results:
            snippets = []
            for c in case_results[:3]:
                if not isinstance(c, dict):
                    continue
                facts = c.get("facts") or []
                txt = ""
                if facts and isinstance(facts[0], dict):
                    txt = facts[0].get("answer", "")
                snippets.append(f"{c.get('faq_id')}:score={float(c.get('fused_score', 0.0)):.3f}:{str(txt)[:80]}")
            if snippets:
                parts.append("embed_cases=" + " | ".join(snippets))

        diseases = payload.get("possible_diseases") or []
        if isinstance(diseases, list) and diseases:
            snippets = []
            for d in diseases[:3]:
                if isinstance(d, dict):
                    snippets.append(str(d.get("disease", ""))[:90])
            if snippets:
                parts.append("possible_diseases=" + " | ".join(snippets))

        advices = payload.get("advices") or []
        if isinstance(advices, list) and advices:
            snippets = []
            for a in advices[:3]:
                if isinstance(a, dict):
                    snippets.append(str(a.get("advice", ""))[:90])
            if snippets:
                parts.append("top_advices=" + " | ".join(snippets))

        text_results = payload.get("text_results") or []
        if isinstance(text_results, list) and text_results:
            snippets = []
            for r in text_results[:3]:
                if not isinstance(r, dict):
                    continue
                faq = r.get("faq_id", "")
                ans = r.get("answer") or ""
                snippets.append(f"{faq}:{str(ans)[:90]}")
            if snippets:
                parts.append("embed_text_top=" + " | ".join(snippets))

        image_results = payload.get("image_results") or []
        if isinstance(image_results, list) and image_results:
            snippets = []
            for r in image_results[:3]:
                if not isinstance(r, dict):
                    continue
                faq = r.get("faq_id", "")
                img = r.get("image_path", "")
                snippets.append(f"{faq}:{str(img)[:80]}")
            if snippets:
                parts.append("embed_image_top=" + " | ".join(snippets))

        diag = payload.get("retrieval_diagnostics") or {}
        if isinstance(diag, dict):
            status = diag.get("status")
            if status:
                parts.append(f"retrieval_status={status}")
            suggested = diag.get("suggested_species") or []
            if isinstance(suggested, list) and suggested:
                parts.append("suggested_species=" + ", ".join([str(x) for x in suggested[:3]]))

        if not parts:
            return str(payload)[:400]
        return "; ".join(parts)[:900]

    def _parse_action_response(
        self,
        text: str,
        permitted_actions: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """解析 LLM 输出为 action + params。"""
        text = text.strip()

        # 优先解析纯 JSON
        try:
            payload = json.loads(text)
            self.last_decision_payload = payload if isinstance(payload, dict) else {}
            action = str(payload.get("action", "")).strip()
            params = payload.get("params", {}) or {}
            if action in permitted_actions and isinstance(params, dict):
                return action, params
        except Exception:
            pass

        # 从文本中提取 JSON 片段
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                payload = json.loads(match.group(0))
                self.last_decision_payload = payload if isinstance(payload, dict) else {}
                action = str(payload.get("action", "")).strip()
                params = payload.get("params", {}) or {}
                if action in permitted_actions and isinstance(params, dict):
                    return action, params
            except Exception:
                pass

        # 兼容旧格式: CALL_TOOL: tool {json}
        if text.startswith("CALL_TOOL:"):
            self.last_decision_payload = {}
            body = text.replace("CALL_TOOL:", "", 1).strip()
            parts = body.split(" ", 1)
            action = parts[0].strip()
            if action in permitted_actions:
                if len(parts) == 1:
                    return action, {}
                try:
                    params = json.loads(parts[1].strip())
                    if isinstance(params, dict):
                        return action, params
                except Exception:
                    return action, {"query": parts[1].strip()}

        # 兼容旧格式: RESPOND: xxx
        if text.startswith("RESPOND:") and "submit_answer" in permitted_actions:
            self.last_decision_payload = {}
            answer = text.replace("RESPOND:", "", 1).strip()
            return "submit_answer", {"answer": answer}

        self.last_decision_payload = {}
        return "error", {}
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return """You are an agricultural tool-using assistant.

    Your tasks are:
    1. Understand the user's agricultural question.
    2. Call available tools to collect evidence before answering.
    3. Use visual, retrieval, diagnosis, or management tools according to the evidence gap.
    4. Submit a final answer only when it cites tool evidence, confidence, and remaining uncertainty.

    Available tools may include:
    - search_knowledge_base
    - identify_disease
    - get_management_advice

    This benchmark evaluates tool use, so do not answer from prior knowledge alone.
    Keep each step concise and action-oriented."""
    
    def run(
        self,
        query: str,
        query_image: Optional[str] = None,
        return_trajectory: bool = True
    ) -> Dict[str, Any]:
        """
        执行 Agent
        
        Args:
            query: 用户查询
            query_image: 查询图像路径（可选）
            return_trajectory: 是否返回完整轨迹
            
        Returns:
            包含答案和轨迹的字典
        """
        # 初始化轨迹
        trajectory = AgentTrajectory(
            query=query,
            query_image=query_image,
            metadata={
                "vlm_model": self.vlm.config.model_name,
                "start_time": datetime.now().isoformat()
            }
        )
        
        # 重置对话历史
        self.conversation_history = []
        
        # 第一步：思考
        think_action = self._think(query, query_image)
        trajectory.actions.append(think_action)
        
        if think_action.action_type == AgentActionType.ERROR:
            trajectory.final_answer = f"Error: {think_action.content}"
            return self._format_output(trajectory, return_trajectory)
        
        # 第二步：根据思考结果决定是否调用工具
        step_count = 1
        while step_count < self.max_steps:
            # 检查上一步是否已经给出最终答案
            if trajectory.actions[-1].action_type == AgentActionType.RESPOND:
                trajectory.final_answer = trajectory.actions[-1].content
                break
            
            # 决定下一步动作
            next_action = self._decide_next_action(trajectory)
            trajectory.actions.append(next_action)
            
            if next_action.action_type == AgentActionType.RESPOND:
                trajectory.final_answer = next_action.content
                break
            elif next_action.action_type == AgentActionType.CALL_TOOL:
                # 这里会调用实际的工具（之后实现）
                pass
            
            step_count += 1
        
        trajectory.metadata["end_time"] = datetime.now().isoformat()
        trajectory.metadata["steps_taken"] = len(trajectory.actions)
        
        return self._format_output(trajectory, return_trajectory)
    
    def _think(self, query: str, query_image: Optional[str]) -> AgentAction:
        """
        第一步：LLM 思考
        
        Returns:
            AgentAction
        """
        try:
            # 构建消息
            messages = [
                VLMMessage(role="system", content=self.system_prompt)
            ]
            
            # 添加用户查询
            if query_image:
                content = [
                    {"type": "image", "image": self.vlm.encode_image(query_image)},
                    {"type": "text", "text": f"请思考这个问题：\n{query}"}
                ]
            else:
                content = f"请思考这个问题：\n{query}"
            
            messages.append(VLMMessage(role="user", content=content))
            
            # 调用 VLM
            response = self.vlm.generate(messages)
            
            # 保存到历史
            self.conversation_history.extend(messages)
            self.conversation_history.append(
                VLMMessage(role="assistant", content=response.text)
            )
            
            return AgentAction(
                action_type=AgentActionType.THINK,
                content=response.text,
                metadata={"model": self.vlm.config.model_name}
            )
        
        except Exception as e:
            return AgentAction(
                action_type=AgentActionType.ERROR,
                content=str(e)
            )
    
    def _decide_next_action(self, trajectory: AgentTrajectory) -> AgentAction:
        """
        根据当前轨迹决定下一步动作
        
        Returns:
            AgentAction
        """
        try:
            # 构建提示，包含当前轨迹
            actions_summary = "\n".join([
                f"步骤 {i+1}: {action.action_type.value}\n{action.content}"
                for i, action in enumerate(trajectory.actions)
            ])
            
            decision_prompt = f"""
Based on the current trajectory, decide the next action.

Current trajectory:
{actions_summary}

Please decide:
1. Whether to call a tool to gather more information?
2. If so, which tool to call and what are the parameters?
3. Or are you ready to generate the final answer?

Format output:
- If calling a tool: reply "CALL_TOOL: [Tool Name] [Parameters]"
- If generating an answer: reply "RESPOND: [Final Answer]"
"""
            
            messages = self.conversation_history + [
                VLMMessage(role="user", content=decision_prompt)
            ]
            
            response = self.vlm.generate(messages)
            
            # 更新历史
            self.conversation_history.append(
                VLMMessage(role="user", content=decision_prompt)
            )
            self.conversation_history.append(
                VLMMessage(role="assistant", content=response.text)
            )
            
            # 解析响应
            if response.text.startswith("CALL_TOOL:"):
                return AgentAction(
                    action_type=AgentActionType.CALL_TOOL,
                    content=response.text,
                    metadata={"parsed": True}
                )
            elif response.text.startswith("RESPOND:"):
                answer = response.text.replace("RESPOND:", "").strip()
                return AgentAction(
                    action_type=AgentActionType.RESPOND,
                    content=answer,
                    metadata={"parsed": True}
                )
            else:
                # 如果格式不符合，让 LLM 重新思考
                return AgentAction(
                    action_type=AgentActionType.THINK,
                    content=response.text,
                    metadata={"unparsed": True}
                )
        
        except Exception as e:
            return AgentAction(
                action_type=AgentActionType.ERROR,
                content=str(e)
            )
    
    def _format_output(
        self,
        trajectory: AgentTrajectory,
        return_trajectory: bool
    ) -> Dict[str, Any]:
        """格式化输出"""
        output = {
            "agent_id": self.agent_id,
            "query": trajectory.query,
            "final_answer": trajectory.final_answer,
            "steps": len(trajectory.actions),
            "metadata": trajectory.metadata
        }
        
        if return_trajectory:
            output["trajectory"] = trajectory.to_dict()
        
        return output
    
    def reset(self):
        """重置 Agent 状态"""
        self.conversation_history = []
        self.evidence_memory = []
        self.task_goal = ""
        self.working_hypothesis = "No hypothesis yet."
        self.action_memory = []
        self.visual_artifacts = []
        self.accepted_evidence = []
        self.rejected_evidence = []
        self.last_decision_payload = {}
        self.primary_image_path = ""
        self.primary_image_input = None
        self.image_paths = []
        self.image_inputs = []
        self.current_env_state = {}
