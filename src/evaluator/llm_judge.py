"""Trajectory-aware LLM judge for the main MMST inference workflow."""

import json
import re
from typing import Any, Dict, List

from vlm import BaseVLM, VLMMessage


class LLMJudge:
    """Trajectory-aware evaluator used by the main benchmark runner."""

    JUDGE_MAX_TOKENS = 800
    JUDGE_SCORE_ONLY_MAX_TOKENS = 800
    JUDGE_DIAGNOSTIC_MAX_TOKENS = 2400
    JUDGE_DEFAULT_EVIDENCE_ROWS = 3
    JUDGE_MAX_DISEASE_ROWS = 5

    def __init__(self, vlm: BaseVLM, judge_mode: str = "score_only"):
        self.vlm = vlm
        if judge_mode not in {"score_only", "diagnostic"}:
            raise ValueError(f"Unsupported judge_mode: {judge_mode}")
        self.judge_mode = judge_mode
        self.max_tokens = (
            self.JUDGE_SCORE_ONLY_MAX_TOKENS
            if judge_mode == "score_only"
            else self.JUDGE_DIAGNOSTIC_MAX_TOKENS
        )

    def judge(self, result: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
        request = self.build_judge_request(result, tool_names)
        if request.get("precomputed_judge") is not None:
            return request["precomputed_judge"]

        try:
            resp = self.vlm.generate(request["messages"], temperature=0.0, max_tokens=self.max_tokens)
            raw = resp.text.strip()
        except RuntimeError as e:
            if "context" not in str(e).lower():
                raise
            request = self.build_judge_request(result, tool_names, ultra_compact=True)
            if request.get("precomputed_judge") is not None:
                return request["precomputed_judge"]
            try:
                resp = self.vlm.generate(request["messages"], temperature=0.0, max_tokens=self.max_tokens)
                raw = resp.text.strip()
            except RuntimeError:
                state = request["state"]
                return self._context_fallback_judge(
                    state["deterministic_flags"],
                    state["compact_trace"],
                    state["compact_decisions"],
                )

        return self.parse_judge_response(raw, request["state"])

    def build_judge_request(
        self,
        result: Dict[str, Any],
        tool_names: List[str],
        *,
        ultra_compact: bool = False,
    ) -> Dict[str, Any]:
        """Build judge messages and deterministic state without calling the VLM."""
        state = self._prepare_judge_state(result, tool_names)
        if state.get("precomputed_judge") is not None:
            return {"messages": [], "state": state, "precomputed_judge": state["precomputed_judge"]}

        compact_trace = state["compact_trace"]
        compact_decisions = state["compact_decisions"]
        deterministic_flags = list(state["deterministic_flags"])
        if ultra_compact:
            compact_trace = self._ultra_compact_trace(compact_trace)
            compact_decisions = self._ultra_compact_decisions(compact_decisions)
            deterministic_flags.append("judge_ultra_compact_retry")

        judge_prompt = self._build_judge_prompt(
            result=state["result"],
            tool_names=tool_names,
            compact_trace=compact_trace,
            compact_decisions=compact_decisions,
            final_cited_facts=state["final_cited_facts"],
            deterministic_flags=deterministic_flags,
            has_image=state["has_image"],
            answer=state["answer"],
            gt_answer=state["gt_answer"],
            answer_gt_overlap=state["answer_gt_overlap"],
        )
        messages = [
            VLMMessage(role="system", content="You are a strict evaluator. Output compact JSON only, with no markdown."),
            VLMMessage(role="user", content=judge_prompt),
        ]
        state = dict(state)
        state["compact_trace"] = compact_trace
        state["compact_decisions"] = compact_decisions
        state["deterministic_flags"] = deterministic_flags
        state["judge_mode"] = self.judge_mode
        return {"messages": messages, "state": state, "precomputed_judge": None}

    def parse_judge_response(self, raw: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a judge model response and apply code-owned score computation."""
        parsed = self._extract_json(str(raw or "").strip())
        if parsed is None:
            parsed = {
                "total_score": 0.0,
                "answer_score": 0.0,
                "process_score": 0.0,
                "verdict": "fail",
                "reason": "Judge output parsing failed",
                "subscores": {
                    "answer_correctness": 0.0,
                    "evidence_grounding": 0.0,
                    "trajectory_quality": 0.0,
                    "tool_efficiency": 0.0,
                    "visual_reasoning": 0.0,
                },
                "flags": ["judge_parse_failed"],
            }

        parsed = self._normalize_judge_result(
            parsed,
            state.get("deterministic_flags", []),
            state.get("compact_trace", []),
            state.get("compact_decisions", []),
            include_step_rationales=self.judge_mode == "diagnostic",
        )
        parsed["raw_judge_output"] = str(raw or "")[:2000]
        parsed["judge_mode"] = self.judge_mode
        return parsed

    def _prepare_judge_state(self, result: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
        trajectory = result.get("trajectory", [])
        judge_context = result.get("judge_context") if isinstance(result.get("judge_context"), dict) else {}
        if judge_context:
            context_task = judge_context.get("task") or {}
            context_final = judge_context.get("final") or {}
            result = dict(result)
            result["query"] = result.get("query") or context_task.get("query") or ""
            result["gt_answer"] = result.get("gt_answer") or context_task.get("gt_answer") or ""
            result["final_answer"] = result.get("final_answer") or context_final.get("pred_answer") or ""
            result["image_path"] = result.get("image_path") or context_task.get("image_path")
            result["image_paths"] = result.get("image_paths") or context_task.get("image_paths") or []
            result["metadata"] = result.get("metadata") or context_task.get("metadata") or {}
        if judge_context:
            compact_trace = judge_context.get("compact_trajectory") or []
            compact_decisions = judge_context.get("agent_decision_metadata") or []
            final_cited_facts = judge_context.get("final_cited_evidence_facts") or []
        else:
            important_evidence_ids = self._important_evidence_ids(result)
            compact_trace = self._compact_trajectory(trajectory, important_evidence_ids)
            compact_decisions = self._compact_decision_trace(result.get("decision_trace", []))
            final_cited_facts = self._final_cited_evidence_facts(trajectory, important_evidence_ids)
        deterministic_flags = self._deterministic_flags(result, compact_trace)
        deterministic_flags.extend(self._evidence_ledger_flags(result))

        state = {
            "result": result,
            "tool_names": tool_names,
            "compact_trace": compact_trace,
            "compact_decisions": compact_decisions,
            "final_cited_facts": final_cited_facts,
            "deterministic_flags": deterministic_flags,
            "precomputed_judge": None,
        }

        if not str(result.get("final_answer", "")).strip():
            step_scores = self._fallback_step_scores(compact_trace, compact_decisions)
            state["precomputed_judge"] = {
                "total_score": 0.0,
                "score_source": "code_from_subscores",
                "answer_score": 0.0,
                "process_score": 0.0,
                "verdict": "fail",
                "reason": "No final answer was submitted before the episode ended.",
                "subscores": {
                    "answer_correctness": 0.0,
                    "evidence_grounding": 0.0,
                    "trajectory_quality": 0.0,
                    "tool_efficiency": 0.0,
                    "visual_reasoning": 0.0,
                },
                "flags": ["missing_final_answer"],
                "step_scores": step_scores,
                "step_score_summary": self._summarize_step_scores(step_scores),
                "raw_judge_output": "",
                "judge_mode": self.judge_mode,
            }
            return state

        has_image = bool(result.get("image_path") or result.get("image_paths")) or bool((result.get("metadata") or {}).get("image_path")) or any(
            "image_path" in json.dumps(step, ensure_ascii=False)
            for step in trajectory[:2]
        )
        answer = str(result.get("final_answer", ""))
        gt_answer = str(result.get("gt_answer", ""))
        state.update(
            {
                "has_image": has_image,
                "answer": answer,
                "gt_answer": gt_answer,
                "answer_gt_overlap": self._keyword_overlap(answer, gt_answer),
            }
        )
        return state

    def _judge_output_instructions(self) -> str:
        base = (
            "Scoring:\n"
            "You must score only the five capability subscores below. Do not calculate answer_score, process_score, total_score, pass/fail, or reward; code will compute those deterministically from subscores.\n"
            "Subscore ranges: answer_correctness 0-4, evidence_grounding 0-2, "
            "trajectory_quality 0-2, tool_efficiency 0-1, visual_reasoning 0-1.\n"
            "Code scoring after your response: answer_score = answer_correctness; process_score = evidence_grounding + trajectory_quality + tool_efficiency + visual_reasoning; total_score = answer_score + process_score.\n"
            "Step scores are diagnostic only: 0 harmful/invalid, 0.25 irrelevant/weak, 0.5 partially useful, "
            "0.75 useful, 1.0 decisive. Do not reward a tool merely because it executed.\n"
        )
        if self.judge_mode == "score_only":
            return (
                base
                + "Score-only output mode:\n"
                + "Return compact JSON only with exactly these top-level keys: subscores, step_scores, flags. Do not include prose, explanations, reason, positive_evidence, issues, total_score, answer_score, process_score, verdict, or markdown.\n"
                + "step_scores schema: [{\"step\": int, \"action\": str, \"score\": 0-1}]. Give one step_scores object per compact_trajectory step; do not output a bare numeric list.\n"
                + "subscores must contain exactly these numeric keys: answer_correctness, evidence_grounding, trajectory_quality, tool_efficiency, visual_reasoning. flags must be a list of short strings and may be empty.\n"
            )
        return (
            base
            + "Diagnostic output mode:\n"
            + "Give one step_scores object per compact_trajectory step; keep reasons under 25 words.\n"
            + "step_scores schema: [{\"step\": int, \"action\": str, \"score\": 0-1, \"reason\": str, \"positive_evidence\": [str], \"issues\": [str]}]. Do not output a bare numeric list.\n"
            + "Required JSON keys: reason, subscores, step_scores, flags. subscores must contain exactly these numeric keys: answer_correctness, evidence_grounding, trajectory_quality, tool_efficiency, visual_reasoning.\n"
        )

    def _build_judge_prompt(
        self,
        result: Dict[str, Any],
        tool_names: List[str],
        compact_trace: List[Dict[str, Any]],
        compact_decisions: List[Dict[str, Any]],
        final_cited_facts: List[Dict[str, Any]],
        deterministic_flags: List[str],
        has_image: bool,
        answer: str,
        gt_answer: str,
        answer_gt_overlap: float,
    ) -> str:
        return (
            "You are a strict evaluator for an agricultural VLM-agent benchmark.\n"
            "This benchmark requires the agent to use tools for agricultural consultation or QA. "
            "Judge whether tool evidence helped answer the user query, not whether the agent followed a fixed tool sequence. "
            "Return compact JSON only.\n\n"
            "Rubric:\n"
            "- answer_correctness: Does pred_answer match gt_answer on the main identification, cause, and management decision?\n"
            "- evidence_grounding: Are final claims supported by available tool evidence, visual observations, retrieved facts, or justified expert inference from similar cases?\n"
            "- trajectory_quality: Were tool calls purposeful, adaptive to observations, and reasonable given the evidence available to the agent?\n"
            "- tool_efficiency: Did the agent avoid unnecessary, repeated, or irrelevant tool calls?\n"
            "- visual_reasoning: For image tasks, did the agent inspect/use the provided image evidence relevant to the query?\n\n"
            "Important separation:\n"
            "- answer_correctness is the outcome score. Do not use it to automatically lower evidence_grounding, trajectory_quality, tool_efficiency, or visual_reasoning.\n"
            "- If KB retrieval lacks a directly matching case, a wrong final answer can still receive reasonable process subscores when the agent used tools sensibly, qualified weak evidence, and stated uncertainty.\n"
            "- Penalize process only for process errors: irrelevant/repeated tools, overstated unsupported claims, ignored useful evidence, ignored images, or misuse of rejected evidence.\n\n"
            "Tool-use alignment:\n"
            "- Do not give credit merely because a tool executed; credit tools only when their evidence supports the final answer or a useful uncertainty statement.\n"
            "- Do not require a fixed chain such as retrieval -> diagnosis -> management.\n"
            "- Do not require crop_image/zoom_image/draw_bounding_box unless local image detail is actually needed for the query.\n"
            "- A concise trajectory with one relevant tool call can score well if the answer is correct and grounded.\n"
            "- A long trajectory should be penalized if it accumulates weak or irrelevant evidence and then overstates a conclusion.\n\n"
            "Trajectory input format:\n"
            "- compact_trajectory contains per-step action, core params, env_state, and an outcome-level evidence summary; raw tool JSON is intentionally omitted.\n"
            "- agent_decision_metadata contains the agent's stated hypothesis, evidence gap, reason for the action, expected contribution, and accepted/rejected evidence deltas.\n"
            "- final_cited_evidence_facts contains complete facts only for evidence ids explicitly cited in the final answer.\n"
            "- Judge process quality from this structured context; do not expect raw env_message or every retrieved candidate.\n\n"
            "Retrieval guidance:\n"
            "- Do not over-penalize retrieval simply because it did not find an exact prior case; KB coverage is incomplete and many MMST GT answers do not expose species explicitly.\n"
            "- Species is useful only when it is clearly present in the query, metadata/entity fields, retrieved facts, or image evidence. Do not treat species mismatch as a universal hard failure.\n"
            "- A retrieved case can be DIRECT evidence when issue, symptom, visual-text clues, management logic, or clearly known species/entity are close enough to support the claim.\n"
            "- A mismatched but semantically or visually similar case can be ANALOGICAL evidence: reward cautious inference that adapts the experience to the current query.\n"
            "- Analogical evidence is weaker than direct evidence, but it is valid process evidence when the agent explains what transfers and what remains uncertain.\n"
            "- Treat retrieval as REJECTED only when the tool explicitly marks recall_only/should_agent_use=false, the returned case is unrelated by symptoms/visual clues/management logic, or the agent uses it to make an overconfident unsupported claim.\n"
            "- If an evidence ledger is present, accepted entries are usable support and rejected entries are limitations/contradictions, not support.\n"
            "- Penalize when the agent keeps an initial hypothesis despite clearly contradictory tool evidence, but do not punish uncertainty caused by absent matching cases.\n"
            "- Do not require an exact prior case; experts may infer management from similar cases if uncertainty and limits are clear.\n\n"
            "Multi-image guidance:\n"
            "- For tasks with multiple image_paths, reward tool calls or reasoning that consider all relevant images.\n"
            "- Penalize missed visual context when the answer depends on multiple images but the trajectory focuses on only one without explanation.\n\n"
            f"{self._judge_output_instructions()}\n"
            f"Available tools: {tool_names}\n"
            f"image_task: {has_image}\n"
            f"deterministic_flags: {deterministic_flags}\n"
            f"answer_gt_keyword_overlap: {answer_gt_overlap:.3f}\n"
            f"query: {str(result.get('query', ''))}\n"
            f"task_metadata: {json.dumps(self._task_metadata_for_judge(result.get('metadata', {})), ensure_ascii=False)}\n"
            f"gt_answer: {gt_answer}\n"
            f"pred_answer: {answer}\n"
            f"compact_trajectory: {json.dumps(compact_trace, ensure_ascii=False)}\n"
            f"agent_decision_metadata: {json.dumps(compact_decisions, ensure_ascii=False)}\n"
            f"final_cited_evidence_facts: {json.dumps(final_cited_facts, ensure_ascii=False)}"
        )

    def _compact_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        important_evidence_ids: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        compact_trace = []
        important_evidence_ids = important_evidence_ids or set()
        for step in trajectory:
            observation = step.get("observation", {})
            env_message = str(observation.get("env_message", ""))
            outcome = self._extract_tool_outcome(env_message)
            compact_trace.append(
                {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "params": self._compact_params(step.get("action"), step.get("params", {})),
                    "observation_type": step.get("observation", {}).get("type"),
                    "env_state": self._compact_env_state(observation.get("env_state", {})),
                    "evidence": outcome,
                }
            )
        return compact_trace

    @staticmethod
    def _compact_decision_trace(decision_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen_accepted: set[str] = set()
        seen_rejected: set[str] = set()
        for step in decision_trace:
            accepted_delta = LLMJudge._compact_evidence_ledger_delta(
                step.get("agent_accepted_evidence"),
                seen_accepted,
            )
            rejected_delta = LLMJudge._compact_evidence_ledger_delta(
                step.get("agent_rejected_evidence"),
                seen_rejected,
            )
            out.append(
                {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "decision_metadata": LLMJudge._compact_decision_metadata(step.get("decision_metadata", {})),
                    "tool_success": step.get("tool_success"),
                    "tool_error": step.get("tool_error"),
                    "visual_artifact": LLMJudge._compact_visual_artifact(step.get("visual_artifact", {})),
                    "agent_visual_artifacts": [
                        LLMJudge._compact_visual_artifact(item)
                        for item in (step.get("agent_visual_artifacts") or [])
                        if isinstance(item, dict)
                    ],
                    "tool_evidence_ledger": {
                        "accepted_delta": accepted_delta,
                        "rejected_delta": rejected_delta,
                        "accepted_total": len(LLMJudge._compact_evidence_ledger(step.get("agent_accepted_evidence"))),
                        "rejected_total": len(LLMJudge._compact_evidence_ledger(step.get("agent_rejected_evidence"))),
                    },
                }
            )
        return out

    @staticmethod
    def _extract_tool_outcome(env_message: str) -> Dict[str, Any]:
        try:
            payload = json.loads(env_message)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        outcome: Dict[str, Any] = {}
        for key in (
            "tool_name",
            "action",
            "query",
            "species",
            "field",
            "result_count",
            "advice_count",
            "result_type",
            "image_path",
            "image_paths",
        ):
            if key in payload:
                outcome[key] = LLMJudge._drop_heavy_fields(payload.get(key))

        diagnostics = payload.get("retrieval_diagnostics")
        if isinstance(diagnostics, dict):
            outcome["retrieval_diagnostics"] = LLMJudge._drop_heavy_fields(diagnostics)

        if payload.get("generated_image_path"):
            outcome["visual_evidence"] = {
                key: LLMJudge._drop_heavy_fields(payload.get(key))
                for key in (
                    "source_image_path",
                    "image_path",
                    "generated_image_path",
                    "operation",
                    "center",
                    "scale",
                    "crop_bbox_pixels",
                    "boxes_pixels",
                    "boxes_normalized",
                    "width",
                    "height",
                )
                if key in payload
            }
            outcome["generated_image_path"] = payload.get("generated_image_path")

        candidate_sources = [
            payload.get("candidates"),
            payload.get("results"),
            payload.get("answer_evidence"),
            payload.get("recall_candidates"),
            payload.get("case_results"),
            payload.get("advices"),
            payload.get("possible_diseases"),
        ]
        evidence_rows = []
        for rows in candidate_sources:
            if isinstance(rows, list):
                evidence_rows.extend([row for row in rows if isinstance(row, dict)])
        if evidence_rows:
            ids = []
            direct = 0
            analogical = 0
            recall_only = 0
            top_rows = []
            for row in evidence_rows:
                faq_id = row.get("faq_id")
                if faq_id and faq_id not in ids:
                    ids.append(faq_id)
                relevance = row.get("relevance") or {}
                if isinstance(relevance, dict):
                    use_as = str(relevance.get("use_as") or "")
                    mismatch_reason = str(relevance.get("mismatch_reason") or "")
                    should_use = relevance.get("should_agent_use")
                    if should_use is False or use_as == "recall_only":
                        recall_only += 1
                    elif mismatch_reason:
                        analogical += 1
                    else:
                        direct += 1
                if len(top_rows) < LLMJudge.JUDGE_DEFAULT_EVIDENCE_ROWS:
                    top_rows.append(LLMJudge._compact_evidence_row_header(row))
            outcome["evidence_summary"] = {
                "total_rows": len(evidence_rows),
                "unique_faq_ids": ids,
                "direct_like_count": direct,
                "analogical_like_count": analogical,
                "recall_only_or_unusable_count": recall_only,
                "top_rows": top_rows,
            }
        return outcome

    @staticmethod
    def _extract_evidence_from_env_message(
        env_message: str,
        important_evidence_ids: set[str] | None = None,
        seen_evidence_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        try:
            payload = json.loads(env_message)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        important_evidence_ids = important_evidence_ids or set()
        evidence: Dict[str, Any] = {}
        for key in (
            "query",
            "species",
            "field",
            "tool_name",
            "action",
            "result_count",
            "advice_count",
            "result_type",
            "image_path",
            "image_paths",
        ):
            if key in payload:
                evidence[key] = LLMJudge._drop_heavy_fields(payload.get(key))

        diagnostics = payload.get("retrieval_diagnostics")
        if isinstance(diagnostics, dict):
            evidence["retrieval_diagnostics"] = LLMJudge._drop_heavy_fields(diagnostics)

        seen_evidence_ids = seen_evidence_ids if seen_evidence_ids is not None else set()
        rows_for_candidates = payload.get("candidates") or payload.get("results") or []
        candidates = LLMJudge._compact_candidate_list(
            rows_for_candidates,
            seen_evidence_ids,
            important_evidence_ids=important_evidence_ids,
        )
        if candidates:
            evidence["candidates"] = candidates
        if isinstance(rows_for_candidates, list) and len(candidates) < len(rows_for_candidates):
            evidence["candidates_selection"] = {
                "policy": "kept final-cited/ledger evidence ids plus top-ranked new rows; duplicate evidence ids are not repeated",
                "kept": len(candidates),
                "available": len(rows_for_candidates),
            }

        has_advices = isinstance(payload.get("advices"), list) and bool(payload.get("advices"))
        for source_key in ("answer_evidence", "recall_candidates"):
            if source_key == "answer_evidence" and has_advices:
                continue
            if source_key == "recall_candidates" and (has_advices or candidates):
                continue
            rows = payload.get(source_key) or []
            if isinstance(rows, list) and rows:
                compact_rows = LLMJudge._compact_candidate_list(
                    rows,
                    seen_evidence_ids,
                    important_evidence_ids=important_evidence_ids,
                )
                if compact_rows:
                    evidence[source_key] = compact_rows
                if len(compact_rows) < len(rows):
                    evidence[f"{source_key}_selection"] = {
                        "policy": "kept final-cited/ledger evidence ids plus top-ranked new rows; duplicate evidence ids are not repeated",
                        "kept": len(compact_rows),
                        "available": len(rows),
                    }

        rows = payload.get("possible_diseases") or []
        if isinstance(rows, list) and rows:
            evidence["possible_diseases"] = [
                LLMJudge._drop_heavy_fields(row)
                for row in rows[: LLMJudge.JUDGE_MAX_DISEASE_ROWS]
            ]
            if len(rows) > LLMJudge.JUDGE_MAX_DISEASE_ROWS:
                evidence["possible_diseases_selection"] = {
                    "policy": "kept top-ranked disease hypotheses",
                    "kept": LLMJudge.JUDGE_MAX_DISEASE_ROWS,
                    "available": len(rows),
                }

        rows = payload.get("case_results") or []
        if isinstance(rows, list) and rows:
            compact_rows = LLMJudge._compact_candidate_list(
                rows,
                seen_evidence_ids,
                important_evidence_ids=important_evidence_ids,
            )
            if compact_rows:
                evidence["case_results"] = compact_rows
            if len(compact_rows) < len(rows):
                evidence["case_results_selection"] = {
                    "policy": "kept final-cited/ledger evidence ids plus top-ranked new rows; duplicate evidence ids are not repeated",
                    "kept": len(compact_rows),
                    "available": len(rows),
                }

        advices = payload.get("advices") or []
        if isinstance(advices, list) and advices:
            compact_advices = LLMJudge._compact_advice_list(
                advices,
                important_evidence_ids=important_evidence_ids,
            )
            evidence["advices"] = compact_advices
            if len(compact_advices) < len(advices):
                evidence["advices_selection"] = {
                    "policy": "kept final-cited/ledger evidence ids plus top-ranked rows",
                    "kept": len(compact_advices),
                    "available": len(advices),
                }

        if payload.get("generated_image_path"):
            evidence["visual_evidence"] = {
                key: LLMJudge._drop_heavy_fields(payload.get(key))
                for key in (
                    "source_image_path",
                    "image_path",
                    "generated_image_path",
                    "operation",
                    "center",
                    "scale",
                    "crop_bbox_pixels",
                    "boxes_pixels",
                    "boxes_normalized",
                    "width",
                    "height",
                    "tool_name",
                    "action",
                )
                if key in payload
            }
            evidence["generated_image_path"] = payload.get("generated_image_path")
        return evidence

    @staticmethod
    def _final_cited_evidence_facts(
        trajectory: List[Dict[str, Any]],
        important_evidence_ids: set[str],
    ) -> List[Dict[str, Any]]:
        if not important_evidence_ids:
            return []
        out_by_id: Dict[str, Dict[str, Any]] = {}
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            env_message = str((step.get("observation") or {}).get("env_message", ""))
            try:
                payload = json.loads(env_message)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            row_sources = [
                payload.get("candidates"),
                payload.get("results"),
                payload.get("answer_evidence"),
                payload.get("recall_candidates"),
                payload.get("case_results"),
                payload.get("advices"),
                payload.get("possible_diseases"),
            ]
            for rows in row_sources:
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    faq_id = str(row.get("faq_id") or "").strip()
                    if not faq_id or faq_id not in important_evidence_ids or faq_id in out_by_id:
                        continue
                    out_by_id[faq_id] = LLMJudge._compact_final_cited_row(row)
        return list(out_by_id.values())

    @staticmethod
    def _compact_final_cited_row(row: Dict[str, Any]) -> Dict[str, Any]:
        if "advice" in row:
            compact = LLMJudge._compact_advice(row)
        else:
            compact = LLMJudge._compact_candidate(row)
        return compact

    @staticmethod
    def _compact_evidence_row_header(row: Dict[str, Any]) -> Dict[str, Any]:
        relevance = row.get("relevance") or {}
        return {
            key: value
            for key, value in {
                "faq_id": row.get("faq_id"),
                "species": row.get("species"),
                "issue_type": row.get("issue_type"),
                "issue": row.get("issue"),
                "scores": {
                    score_key: LLMJudge._round_score(row.get(score_key))
                    for score_key in ("text_score", "image_score", "fused_score", "species_match_score")
                    if score_key in row
                },
                "relevance": LLMJudge._compact_relevance(relevance if isinstance(relevance, dict) else {}),
            }.items()
            if value not in (None, {}, [])
        }

    @staticmethod
    def _short(value: Any, limit: int = 180) -> str:
        text = "" if value is None else str(value)
        return text[:limit]

    @staticmethod
    def _compact_candidate_list(
        rows: Any,
        seen_evidence_ids: set[str] | None = None,
        important_evidence_ids: set[str] | None = None,
        default_keep: int = JUDGE_DEFAULT_EVIDENCE_ROWS,
    ) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        out = []
        seen_evidence_ids = seen_evidence_ids if seen_evidence_ids is not None else set()
        important_evidence_ids = important_evidence_ids or set()
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            compact = LLMJudge._compact_candidate(row)
            identity = LLMJudge._evidence_identity(compact)
            source_id = str(compact.get("faq_id") or "").strip()
            is_important = bool(source_id and source_id in important_evidence_ids)
            if idx >= default_keep and not is_important:
                continue
            if identity and identity in seen_evidence_ids:
                continue
            if identity:
                seen_evidence_ids.add(identity)
            out.append(compact)
        return out

    @staticmethod
    def _evidence_identity(row: Dict[str, Any]) -> str:
        faq_id = row.get("faq_id")
        if faq_id:
            return f"faq:{faq_id}"
        return json.dumps(row, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _compact_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
        scores = {
            key: LLMJudge._round_score(row.get(key))
            for key in (
                "text_score",
                "image_score",
                "fused_score",
                "species_match_score",
                "domain_adjustment",
            )
            if key in row
        }
        facts = []
        for source_key in ("facts", "matched_facts", "all_facts"):
            values = row.get(source_key) or []
            if isinstance(values, list):
                facts.extend([fact for fact in values if isinstance(fact, dict)])
        case_evidence = row.get("case_evidence") or {}
        if isinstance(case_evidence, dict):
            values = case_evidence.get("facts") or []
            if isinstance(values, list):
                facts.extend([fact for fact in values if isinstance(fact, dict)])

        compact_facts = LLMJudge._compact_facts(facts)
        compact = {
            "faq_id": row.get("faq_id"),
            "species": row.get("species"),
            "issue_type": row.get("issue_type"),
            "image_path": row.get("image_path"),
            "query_image_path": row.get("query_image_path"),
            "query_image_index": row.get("query_image_index"),
            "scores": scores,
            "evidence_channels": LLMJudge._drop_heavy_fields(row.get("evidence_channels")),
            "relevance": LLMJudge._compact_relevance(row.get("relevance") or {}),
            "text_evidence": None if compact_facts else LLMJudge._drop_heavy_fields(row.get("text_evidence")),
            "facts": compact_facts,
        }
        return {key: value for key, value in compact.items() if value not in (None, {}, [])}

    @staticmethod
    def _compact_advice(row: Dict[str, Any]) -> Dict[str, Any]:
        facts = []
        for source_key in ("facts", "matched_facts", "all_facts"):
            values = row.get(source_key) or []
            if isinstance(values, list):
                facts.extend([fact for fact in values if isinstance(fact, dict)])
        compact = {
            "faq_id": row.get("faq_id"),
            "species": row.get("species"),
            "issue": row.get("issue"),
            "issue_type": row.get("issue_type"),
            "advice": row.get("advice"),
            "evidence_question": row.get("evidence_question"),
            "relevance": LLMJudge._compact_relevance(row.get("relevance") or {}),
            "facts": LLMJudge._compact_facts(facts),
        }
        return {key: value for key, value in compact.items() if value not in (None, {}, [])}

    @staticmethod
    def _compact_advice_list(
        rows: Any,
        important_evidence_ids: set[str] | None = None,
        default_keep: int = JUDGE_DEFAULT_EVIDENCE_ROWS,
    ) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        important_evidence_ids = important_evidence_ids or set()
        out = []
        seen = set()
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("faq_id") or "").strip()
            is_important = bool(source_id and source_id in important_evidence_ids)
            if idx >= default_keep and not is_important:
                continue
            compact = LLMJudge._compact_advice(row)
            identity = LLMJudge._evidence_identity(compact)
            if identity in seen:
                continue
            seen.add(identity)
            out.append(compact)
        return out

    @staticmethod
    def _compact_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            compact = {
                key: fact.get(key)
                for key in ("faq_id", "species", "field", "question", "answer")
                if key in fact
            }
            dedupe_key = json.dumps(compact, sort_keys=True, ensure_ascii=False)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.append(compact)
        return out

    @staticmethod
    def _compact_relevance(relevance: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(relevance, dict) or not relevance:
            return {}
        out = {}
        for key in (
            "species_match_score",
            "field_match",
            "query_term_match",
            "should_agent_use",
            "mismatch_reason",
            "use_as",
        ):
            if key in relevance:
                out[key] = LLMJudge._drop_heavy_fields(relevance.get(key))
        return out

    @staticmethod
    def _round_score(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        return value

    @staticmethod
    def _compact_params(action: Any, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {}
        compact = {}
        action_name = str(action or "")
        for key, value in params.items():
            if action_name == "submit_answer" and key == "answer":
                continue
            if key in {
                "answer",
                "query",
                "species",
                "field",
                "image_path",
                "image_paths",
                "image_index",
                "bbox",
                "boxes",
                "center",
                "scale",
                "normalized",
                "labels",
                "confidence",
                "remaining_uncertainty",
                "evidence_used",
            }:
                compact[key] = LLMJudge._drop_heavy_fields(value)
        if not compact and action:
            compact = LLMJudge._drop_heavy_fields(params)
        return compact

    @staticmethod
    def _compact_env_state(env_state: Any) -> Dict[str, Any]:
        if not isinstance(env_state, dict):
            return {}
        keys = [
            "step_count",
            "steps_remaining",
            "must_submit_next",
            "last_action",
            "repeated_action",
            "used_tools",
            "visual_artifacts",
        ]
        return {
            key: LLMJudge._drop_heavy_fields(env_state.get(key))
            for key in keys
            if key in env_state
        }

    @staticmethod
    def _compact_decision_metadata(metadata: Any) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        keys = [
            "working_hypothesis",
            "evidence_gap",
            "why_this_tool",
            "expected_contribution",
            "accepted_evidence",
            "rejected_evidence",
        ]
        return {
            key: LLMJudge._drop_heavy_fields(metadata.get(key))
            for key in keys
            if metadata.get(key) not in (None, "", [], {})
        }

    @staticmethod
    def _task_metadata_for_judge(metadata: Any) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        priority_keys = [
            "category",
            "entity_type",
            "entity_name",
            "entity_scientific_name",
            "entity_common_names",
            "meta_data_state",
            "meta_data_county",
            "meta_data_asked_time",
            "source",
            "subset",
            "image_paths",
        ]
        out = {
            key: LLMJudge._drop_heavy_fields(metadata.get(key))
            for key in priority_keys
            if key in metadata
        }
        extras = {
            key: value
            for key, value in metadata.items()
            if key not in out and key not in {"raw", "images", "image_bytes"}
        }
        if extras:
            out["extra_metadata"] = LLMJudge._drop_heavy_fields(extras)
        return out

    @staticmethod
    def _compact_visual_artifact(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        keys = [
            "tool_name",
            "action",
            "operation",
            "source_image_path",
            "generated_image_path",
            "bbox_pixels",
            "bbox_normalized",
        ]
        return {
            key: LLMJudge._drop_heavy_fields(value.get(key))
            for key in keys
            if key in value
        }

    @staticmethod
    def _compact_evidence_ledger(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            if not isinstance(item, dict):
                continue
            compact = {
                key: LLMJudge._drop_heavy_fields(item.get(key))
                for key in (
                    "faq_id",
                    "source_id",
                    "source_type",
                    "species",
                    "issue_type",
                    "reason",
                    "evidence_role",
                    "claim_supported",
                )
                if key in item
            }
            if compact:
                out.append(compact)
        return out

    @staticmethod
    def _compact_evidence_ledger_delta(value: Any, seen_ids: set[str]) -> List[Dict[str, Any]]:
        out = []
        for item in LLMJudge._compact_evidence_ledger(value):
            source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
            identity = source_id or json.dumps(item, sort_keys=True, ensure_ascii=False)
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            out.append(item)
        return out

    @staticmethod
    def _drop_heavy_fields(value: Any) -> Any:
        heavy_keys = {
            "case_evidence",
            "facts_by_group",
            "embedding",
            "embeddings",
            "image_embedding",
            "text_embedding",
            "pixel_values",
            "raw_image",
            "image_bytes",
            "llm_raw",
        }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [LLMJudge._drop_heavy_fields(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): LLMJudge._drop_heavy_fields(item)
                for key, item in value.items()
                if str(key) not in heavy_keys
            }
        return str(value)

    @staticmethod
    def _truncate_obj(value: Any, max_str: int = 180, max_list: int = 4, max_dict: int = 8) -> Any:
        if isinstance(value, str):
            return value[:max_str]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [LLMJudge._truncate_obj(v, max_str=max_str, max_list=max_list, max_dict=max_dict) for v in value[:max_list]]
        if isinstance(value, dict):
            out = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= max_dict:
                    break
                out[str(key)] = LLMJudge._truncate_obj(item, max_str=max_str, max_list=max_list, max_dict=max_dict)
            return out
        return str(value)[:max_str]

    @staticmethod
    def _ultra_compact_trace(compact_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for step in compact_trace:
            evidence = step.get("evidence") or {}
            out.append(
                {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "params": LLMJudge._compact_params(step.get("action"), step.get("params", {})),
                    "evidence_keys": sorted([str(k) for k in evidence.keys()])[:8],
                    "generated_image_path": evidence.get("generated_image_path"),
                    "message_excerpt": LLMJudge._short(step.get("message_excerpt"), 80),
                }
            )
        return out

    @staticmethod
    def _ultra_compact_decisions(compact_decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for step in compact_decisions:
            metadata = step.get("decision_metadata") or {}
            out.append(
                {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "why": LLMJudge._short(metadata.get("why_this_tool"), 120),
                    "hypothesis": LLMJudge._short(metadata.get("working_hypothesis"), 120),
                }
            )
        return out

    @classmethod
    def _context_fallback_judge(
        cls,
        deterministic_flags: List[str],
        compact_trace: List[Dict[str, Any]],
        compact_decisions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        step_scores = cls._fallback_step_scores(compact_trace, compact_decisions)
        parsed = {
            "reason": "Judge VLM context limit exceeded; used deterministic fallback step scoring.",
            "subscores": {
                "answer_correctness": 0.0,
                "evidence_grounding": 0.0,
                "trajectory_quality": min(2.0, 2.0 * cls._summarize_step_scores(step_scores)["avg_step_score"]),
                "tool_efficiency": 0.5,
                "visual_reasoning": 0.5,
            },
            "flags": deterministic_flags + ["judge_context_fallback"],
            "step_scores": step_scores,
            "raw_judge_output": "",
        }
        return cls._normalize_judge_result(parsed, parsed["flags"], compact_trace, compact_decisions)

    @staticmethod
    def _keyword_overlap(a: str, b: str) -> float:
        stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "with", "by", "it"}
        toks_a = {t for t in re.findall(r"[a-zA-Z0-9]+", a.lower()) if len(t) > 2 and t not in stop}
        toks_b = {t for t in re.findall(r"[a-zA-Z0-9]+", b.lower()) if len(t) > 2 and t not in stop}
        if not toks_a or not toks_b:
            return 0.0
        return len(toks_a & toks_b) / max(1, len(toks_b))

    @staticmethod
    def _deterministic_flags(result: Dict[str, Any], compact_trace: List[Dict[str, Any]]) -> List[str]:
        flags = []
        actions = [str(s.get("action", "")) for s in compact_trace]
        if not actions or actions[-1] != "submit_answer":
            flags.append("no_terminal_submit")
        action_param_keys = [
            (str(s.get("action", "")), json.dumps(s.get("params", {}), sort_keys=True))
            for s in compact_trace
        ]
        if len(action_param_keys) != len(set(action_param_keys)):
            flags.append("duplicate_action_params")
        if actions[:3] == ["retrieve_multimodal_evidence", "identify_disease", "get_management_advice"]:
            flags.append("template_retrieve_identify_advice_path")
        tool_actions = [a for a in actions if a and a not in {"submit_answer", "request_info"}]
        if not tool_actions:
            flags.append("no_tool_call")
        used_visual_retrieval = any(a in {"visual_retrieve", "retrieve_multimodal_evidence"} for a in actions)
        used_visual_artifact = any(a in {"crop_image", "zoom_image", "draw_bounding_box"} for a in actions)
        tracked_visual_artifact = any((s.get("evidence") or {}).get("generated_image_path") for s in compact_trace)
        if used_visual_retrieval:
            flags.append("used_visual_retrieval")
        if used_visual_artifact or tracked_visual_artifact:
            flags.append("used_visual_artifact")
        if any("image_path" in json.dumps(s, ensure_ascii=False) for s in compact_trace):
            if not used_visual_retrieval and not used_visual_artifact:
                flags.append("image_task_without_visual_evidence")
        return flags

    @staticmethod
    def _important_evidence_ids(result: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text:
                ids.add(text)

        for step in result.get("trajectory", []) or []:
            if not isinstance(step, dict) or step.get("action") != "submit_answer":
                continue
            params = step.get("params") or {}
            if not isinstance(params, dict):
                continue
            for item in params.get("evidence_used") or []:
                if not isinstance(item, dict):
                    continue
                source_id = item.get("source_id") or item.get("faq_id") or item.get("case_id")
                add(source_id)
        judge_context = result.get("judge_context") if isinstance(result.get("judge_context"), dict) else {}
        final_context = judge_context.get("final") if isinstance(judge_context.get("final"), dict) else {}
        for item in final_context.get("evidence_used") or []:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id") or item.get("faq_id") or item.get("case_id")
            add(source_id)
        return ids

    @staticmethod
    def _evidence_ledger_flags(result: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        accepted: Dict[str, str] = {}
        rejected: set[str] = set()

        judge_context = result.get("judge_context") if isinstance(result.get("judge_context"), dict) else {}
        decision_steps = result.get("decision_trace", []) or judge_context.get("agent_decision_metadata", []) or []
        for step in decision_steps:
            if not isinstance(step, dict):
                continue
            ledger = step.get("tool_evidence_ledger") or {}
            if isinstance(ledger, dict):
                for item in ledger.get("accepted_delta") or ledger.get("accepted") or []:
                    if not isinstance(item, dict):
                        continue
                    source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
                    if source_id:
                        accepted[source_id] = str(item.get("evidence_role") or "accepted")
                for item in ledger.get("rejected_delta") or ledger.get("rejected") or []:
                    if not isinstance(item, dict):
                        continue
                    source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
                    if source_id:
                        rejected.add(source_id)
            for item in step.get("agent_accepted_evidence") or []:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
                if source_id:
                    accepted[source_id] = str(item.get("evidence_role") or "accepted")
            for item in step.get("agent_rejected_evidence") or []:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("faq_id") or item.get("source_id") or "").strip()
                if source_id:
                    rejected.add(source_id)

        submit_params = {}
        compact_steps = result.get("trajectory", []) or judge_context.get("compact_trajectory", []) or []
        for step in compact_steps:
            if isinstance(step, dict) and step.get("action") == "submit_answer":
                submit_params = step.get("params") or {}
        cited_ids = set()
        for item in submit_params.get("evidence_used") or []:
            if isinstance(item, dict):
                source_id = str(item.get("source_id") or item.get("faq_id") or item.get("case_id") or "").strip()
                if source_id:
                    cited_ids.add(source_id)
        final_context = judge_context.get("final") if isinstance(judge_context.get("final"), dict) else {}
        for item in final_context.get("evidence_used") or []:
            if isinstance(item, dict):
                source_id = str(item.get("source_id") or item.get("faq_id") or item.get("case_id") or "").strip()
                if source_id:
                    cited_ids.add(source_id)

        if cited_ids & rejected:
            flags.append("final_cites_rejected_evidence")
        if accepted and cited_ids and not (cited_ids & set(accepted)):
            flags.append("final_cites_untracked_evidence")
        cited_accepted_roles = [accepted[x] for x in cited_ids if x in accepted]
        if cited_accepted_roles and all(role == "analogical" for role in cited_accepted_roles):
            flags.append("final_supported_only_by_analogical_evidence")
        confidence = str(submit_params.get("confidence") or "").lower()
        if confidence == "high" and "final_supported_only_by_analogical_evidence" in flags:
            flags.append("high_confidence_with_only_analogical_evidence")
        return flags

    @classmethod
    def _normalize_judge_result(
        cls,
        parsed: Dict[str, Any],
        deterministic_flags: List[str],
        compact_trace: List[Dict[str, Any]] | None = None,
        compact_decisions: List[Dict[str, Any]] | None = None,
        include_step_rationales: bool = True,
    ) -> Dict[str, Any]:
        subscores = parsed.get("subscores")
        if not isinstance(subscores, dict):
            subscores = {}
        expected = {
            "answer_correctness": 0.0,
            "evidence_grounding": 0.0,
            "trajectory_quality": 0.0,
            "tool_efficiency": 0.0,
            "visual_reasoning": 0.0,
        }
        max_scores = {
            "answer_correctness": 4.0,
            "evidence_grounding": 2.0,
            "trajectory_quality": 2.0,
            "tool_efficiency": 1.0,
            "visual_reasoning": 1.0,
        }
        for key, default in expected.items():
            try:
                value = float(subscores.get(key, default))
            except Exception:
                value = default
            subscores[key] = max(0.0, min(max_scores[key], value))

        if "template_retrieve_identify_advice_path" in deterministic_flags:
            subscores["trajectory_quality"] = min(subscores["trajectory_quality"], 1.0)
        if "no_tool_call" in deterministic_flags:
            subscores["evidence_grounding"] = min(subscores["evidence_grounding"], 0.5)
            subscores["trajectory_quality"] = min(subscores["trajectory_quality"], 0.5)
        if "image_task_without_visual_evidence" in deterministic_flags:
            subscores["visual_reasoning"] = min(subscores["visual_reasoning"], 0.5)
        if "final_cites_rejected_evidence" in deterministic_flags:
            subscores["evidence_grounding"] = min(subscores["evidence_grounding"], 0.5)
            subscores["trajectory_quality"] = min(subscores["trajectory_quality"], 1.0)
        if "final_cites_untracked_evidence" in deterministic_flags:
            subscores["evidence_grounding"] = min(subscores["evidence_grounding"], 1.5)
        if "high_confidence_with_only_analogical_evidence" in deterministic_flags:
            subscores["trajectory_quality"] = min(subscores["trajectory_quality"], 1.5)

        tool_steps = [
            s for s in (compact_trace or [])
            if str(s.get("action") or "") not in {"submit_answer", "request_info", ""}
        ]
        has_duplicate = "duplicate_action_params" in deterministic_flags
        if tool_steps and not has_duplicate:
            efficient_short_path = len(tool_steps) <= 2 and subscores["answer_correctness"] >= 3.0 and subscores["evidence_grounding"] >= 1.5
            if efficient_short_path:
                subscores["tool_efficiency"] = max(subscores["tool_efficiency"], 0.5)

        try:
            parsed["model_raw_total_score"] = float(parsed.get("total_score"))
        except Exception:
            parsed["model_raw_total_score"] = None
        try:
            parsed["model_raw_answer_score"] = float(parsed.get("answer_score"))
        except Exception:
            parsed["model_raw_answer_score"] = None
        try:
            parsed["model_raw_process_score"] = float(parsed.get("process_score"))
        except Exception:
            parsed["model_raw_process_score"] = None
        answer_score = float(subscores.get("answer_correctness", 0.0))
        process_score = (
            float(subscores.get("evidence_grounding", 0.0))
            + float(subscores.get("trajectory_quality", 0.0))
            + float(subscores.get("tool_efficiency", 0.0))
            + float(subscores.get("visual_reasoning", 0.0))
        )
        total = answer_score + process_score
        parsed["subscores"] = subscores
        parsed["answer_score"] = round(max(0.0, min(4.0, answer_score)), 3)
        parsed["process_score"] = round(max(0.0, min(6.0, process_score)), 3)
        parsed["total_score"] = round(max(0.0, min(10.0, total)), 3)
        parsed["score_source"] = "code_from_subscores"
        parsed["verdict"] = "pass" if parsed["total_score"] >= 7.0 else "fail"
        flags = parsed.get("flags")
        if not isinstance(flags, list):
            flags = []
        parsed["flags"] = sorted(set([str(x) for x in flags] + deterministic_flags))

        step_scores = cls._normalize_step_scores(
            parsed.get("step_scores"),
            compact_trace or [],
            compact_decisions or [],
            include_rationales=include_step_rationales,
        )
        step_scores = cls._calibrate_step_scores(
            step_scores,
            subscores,
            compact_trace or [],
            include_rationales=include_step_rationales,
        )
        parsed["step_scores"] = step_scores
        parsed["step_score_summary"] = cls._summarize_step_scores(step_scores)
        return parsed

    @staticmethod
    def _calibrate_step_scores(
        step_scores: List[Dict[str, Any]],
        subscores: Dict[str, float],
        compact_trace: List[Dict[str, Any]],
        include_rationales: bool = True,
    ) -> List[Dict[str, Any]]:
        evidence_grounding = float(subscores.get("evidence_grounding", 0.0))
        trace_by_step = {str(step.get("step")): step for step in compact_trace}
        calibrated = []
        for item in step_scores:
            step = dict(item)
            action = str(step.get("action") or "")
            score = float(step.get("score", 0.0))
            issues = list(step.get("issues") or []) if include_rationales else []
            trace_step = trace_by_step.get(str(step.get("step")), {})
            evidence = trace_step.get("evidence") or {}
            if action == "submit_answer":
                if evidence_grounding <= 0.5:
                    score = min(score, 0.5)
                    if include_rationales:
                        issues.append("final answer lacks sufficient evidence grounding")
                elif evidence_grounding >= 1.2:
                    score = max(score, 0.75)
            elif action in {"visual_retrieve", "retrieve_multimodal_evidence", "retrieve_by_embedding", "search_knowledge_base", "get_management_advice"}:
                unusable = LLMJudge._retrieval_evidence_is_unusable(evidence)
                analogical = LLMJudge._retrieval_evidence_is_analogical(evidence)
                if unusable:
                    score = min(score, 0.35)
                    if include_rationales:
                        issues.append("top retrieval evidence is recall_only or marked unusable")
                elif analogical and score > 0.85:
                    score = 0.85
                    if include_rationales:
                        issues.append("retrieval evidence is analogical; judge transfer reasoning")
            elif action in {"crop_image", "zoom_image", "draw_bounding_box"}:
                if evidence_grounding <= 1.0 and score > 0.75:
                    score = 0.75
                    if include_rationales:
                        issues.append("visual artifact was not translated into correct evidence")
            step["score"] = round(max(0.0, min(1.0, score)), 3)
            if include_rationales:
                step["issues"] = sorted(set(str(x) for x in issues))[:8]
            else:
                step = {"step": step.get("step"), "action": step.get("action"), "score": step["score"]}
            calibrated.append(step)
        return calibrated

    @staticmethod
    def _retrieval_evidence_is_unusable(evidence: Dict[str, Any]) -> bool:
        candidates = evidence.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            summary = evidence.get("evidence_summary") or {}
            if isinstance(summary, dict):
                candidates = summary.get("top_rows")
        if not isinstance(candidates, list) or not candidates:
            return False
        summary = evidence.get("evidence_summary") or {}
        if isinstance(summary, dict) and summary.get("recall_only_or_unusable_count") and not summary.get("direct_like_count") and not summary.get("analogical_like_count"):
            return True
        checked = 0
        unusable = 0
        for candidate in candidates[:3]:
            if not isinstance(candidate, dict):
                continue
            relevance = candidate.get("relevance") or {}
            if not isinstance(relevance, dict) or not relevance:
                continue
            checked += 1
            should_use = relevance.get("should_agent_use")
            use_as = str(relevance.get("use_as") or "")
            if should_use is False or use_as == "recall_only":
                unusable += 1
        return checked > 0 and unusable == checked

    @staticmethod
    def _retrieval_evidence_is_analogical(evidence: Dict[str, Any]) -> bool:
        candidates = evidence.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            summary = evidence.get("evidence_summary") or {}
            if isinstance(summary, dict):
                if summary.get("analogical_like_count"):
                    return True
                candidates = summary.get("top_rows")
        if not isinstance(candidates, list) or not candidates:
            return False
        for candidate in candidates[:3]:
            if not isinstance(candidate, dict):
                continue
            relevance = candidate.get("relevance") or {}
            if not isinstance(relevance, dict):
                continue
            mismatch_reason = str(relevance.get("mismatch_reason") or "")
            use_as = str(relevance.get("use_as") or "")
            should_use = relevance.get("should_agent_use")
            if should_use is not False and use_as != "recall_only" and mismatch_reason:
                return True
        return False

    @classmethod
    def _normalize_step_scores(
        cls,
        raw_step_scores: Any,
        compact_trace: List[Dict[str, Any]],
        compact_decisions: List[Dict[str, Any]],
        include_rationales: bool = True,
    ) -> List[Dict[str, Any]]:
        fallback = cls._fallback_step_scores(compact_trace, compact_decisions)
        fallback_by_step = {str(item.get("step")): item for item in fallback}
        if not isinstance(raw_step_scores, list):
            return fallback

        normalized = []
        for idx, trace_step in enumerate(compact_trace):
            step_id = trace_step.get("step")
            action = trace_step.get("action")
            raw = None
            if idx < len(raw_step_scores) and isinstance(raw_step_scores[idx], (int, float)):
                raw = {"step": step_id, "score": raw_step_scores[idx]}
            if raw is None:
                raw = next(
                    (
                        item for item in raw_step_scores
                        if isinstance(item, dict) and str(item.get("step")) == str(step_id)
                    ),
                    None,
                )
            if not raw:
                normalized.append(fallback_by_step.get(str(step_id), cls._fallback_step_score(trace_step, {})))
                continue
            try:
                score = float(raw.get("score", 0.0))
            except Exception:
                score = float(fallback_by_step.get(str(step_id), {}).get("score", 0.0))
            step = {
                "step": step_id,
                "action": action,
                "score": round(max(0.0, min(1.0, score)), 3),
            }
            if include_rationales:
                step.update(
                    {
                        "reason": str(raw.get("reason") or fallback_by_step.get(str(step_id), {}).get("reason", ""))[:500],
                        "positive_evidence": cls._normalize_string_list(raw.get("positive_evidence")),
                        "issues": cls._normalize_string_list(raw.get("issues")),
                    }
                )
            normalized.append(step)
        return normalized

    @classmethod
    def _fallback_step_scores(
        cls,
        compact_trace: List[Dict[str, Any]],
        compact_decisions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        decisions_by_step = {str(item.get("step")): item for item in compact_decisions}
        return [
            cls._fallback_step_score(step, decisions_by_step.get(str(step.get("step")), {}))
            for step in compact_trace
        ]

    @staticmethod
    def _fallback_step_score(step: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        action = str(step.get("action") or "")
        evidence = step.get("evidence") or {}
        metadata = decision.get("decision_metadata") or {}
        issues = []
        positives = []
        score = 0.25

        if action == "submit_answer":
            answer = (step.get("params") or {}).get("answer")
            score = 0.75 if str(answer or "").strip() else 0.1
            positives.append("submitted final answer" if score > 0.1 else "")
        elif action:
            score = 0.45
            positives.append(f"called {action}")

        if evidence:
            score += 0.15
            positives.append("received structured evidence")
        if evidence.get("generated_image_path"):
            score += 0.1
            positives.append("created visual artifact")
        if metadata.get("why_this_tool") or metadata.get("expected_contribution"):
            score += 0.15
            positives.append("stated tool purpose")
        if metadata.get("accepted_evidence") or metadata.get("rejected_evidence"):
            score += 0.1
            positives.append("updated accepted/rejected evidence")
        if action in {"crop_image", "zoom_image", "draw_bounding_box", "inspect_image_metadata", "visual_retrieve", "retrieve_multimodal_evidence"}:
            score += 0.05
            positives.append("used visual pathway")

        env_state = step.get("env_state") or {}
        if action and str(action).startswith("unknown"):
            score = min(score, 0.2)
            issues.append("unknown action")
        if env_state.get("must_submit_next") and action != "submit_answer":
            score = min(score, 0.4)
            issues.append("did not submit when budget required it")
        if not metadata and action != "submit_answer":
            issues.append("missing decision metadata")
        if action in {"crop_image", "zoom_image", "draw_bounding_box"} and not evidence.get("generated_image_path"):
            issues.append("visual action did not yield a tracked artifact")

        positives = [p for p in positives if p]
        return {
            "step": step.get("step"),
            "action": action,
            "score": round(max(0.0, min(1.0, score)), 3),
            "reason": "Heuristic fallback score based on action, evidence, metadata, and visual artifact tracking.",
            "positive_evidence": positives[:5],
            "issues": issues[:5],
        }

    @staticmethod
    def _summarize_step_scores(step_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not step_scores:
            return {
                "avg_step_score": 0.0,
                "min_step_score": 0.0,
                "max_step_score": 0.0,
                "total_steps": 0,
                "low_score_steps": [],
                "trajectory_notes": "No trajectory steps were available for step-level judging.",
            }
        scores = [float(item.get("score", 0.0)) for item in step_scores]
        low_steps = [
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "score": item.get("score"),
                "issues": item.get("issues", []),
            }
            for item in step_scores
            if float(item.get("score", 0.0)) < 0.5
        ]
        return {
            "avg_step_score": round(sum(scores) / len(scores), 3),
            "min_step_score": round(min(scores), 3),
            "max_step_score": round(max(scores), 3),
            "total_steps": len(step_scores),
            "low_score_steps": low_steps[:5],
            "trajectory_notes": "Step scores are diagnostic and justify process_score; total_score = answer_score + process_score.",
        }

    @staticmethod
    def _normalize_string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:300] for item in value[:8]]

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any] | None:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None
