from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import PipelineConfig


class RoutingDecision(BaseModel):
    selected_model: str = Field(
        description=(
            "The exact name of the selected model. MUST be exactly one of the "
            "names from the Candidate Model Score Table."
        )
    )
    target_organ: str = Field(description="The target organ handled by this organ agent.")
    selected_score: float = Field(description="The final routing_score of the selected model.")
    reason: str = Field(
        description=(
            "A brief explanation of why this model's prior score and mask-overlap "
            "evidence make it the best available choice."
        )
    )
    primary_decision_factor: str = Field(
        default="routing_score",
        description=(
            "The main factor behind the choice, such as organ_modality_fit, "
            "prior_routing_score, overlap_score, consensus_iou, mask_area_fraction, "
            "mask_quality_score, execution_safety, or routing_score."
        ),
    )
    evidence_used: list[str] = Field(
        default_factory=list,
        description="Evidence fields used to make the decision.",
    )
    why_not_highest_score_model: str = Field(
        default="",
        description=(
            "If the selected model is not the highest routing_score candidate, explain why. "
            "Otherwise leave this empty."
        ),
    )
    confidence: str = Field(
        default="medium",
        description="Decision confidence: low, medium, or high.",
    )


class LLMRouter:
    """Organ-aware LLM orchestrator with strictly enforced JSON output."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # LLM은 원본 의료영상 픽셀이 아니라 텍스트 scorecard를 입력으로 받는다.
        # temperature=0으로 설정하여 실험을 반복해도 라우팅 결과가 최대한 흔들리지 않게 한다.
        self.llm = ChatOllama(
            model=config.llm_model,
            temperature=0.0,
            format="json",
        )
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)
        self.prompt = PromptTemplate(
            template="""You are an expert medical AI orchestration agent for {target_organ} segmentation.

{format_instructions}

[Context]
- User Query: {query}
- Target Organ Agent: {target_organ}
- Image Metadata: {metadata}

[Candidate Model Score Table]
{candidates}

[Instructions]
The candidate models have already produced masks when execution_status=success.
Their masks were compared by overlap against an inter-model consensus mask.
mask_quality_score is a no-GT morphology stability score. It penalizes suspicious area, excessive fragmentation, rough boundaries, weak dominant components, or many internal holes.
Analyze the candidate score table. Your ONLY task is to return a JSON object with the fields specified above.
The selected_model field MUST exactly match one of the model names from the Candidate Model Score Table.
Select the model that is most appropriate for the user request, target organ, modality, image metadata, model description, prior validation evidence, and no-GT mask agreement evidence.
Use routing_score as an important reference, but do not treat it as the only decision criterion.
You may choose a model that does not have the highest routing_score if the candidate has stronger organ/modality fit, more reliable mask agreement, safer execution status, a less suspicious mask_area_fraction, stronger mask_quality_score, or a better explanation for the current request.
When explaining the choice, mention the selected candidate's execution_status, prior evidence, overlap evidence, consensus_iou, avg_pairwise_iou, mask_area_fraction, mask_quality_score, and quality_flags when available.
Do not over-trust consensus_iou when mask_quality_score is low or quality_flags indicate rough_boundary, suspicious_lung_area, fragmented_mask, or implausible_lung_area.
If you do not choose the highest routing_score candidate, explain why in why_not_highest_score_model.
Do not claim clinical accuracy. GT is unknown at inference time.
DO NOT select any model with execution_status other than success, zero routing_score, selection_enabled=false, or an error.
""",
            input_variables=["query", "target_organ", "metadata", "candidates"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser

    def select_model(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        sample_metadata: dict[str, str] | None = None,
        target_organ: str | None = None,
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidate models were provided.")

        agent_target = target_organ or self.config.target_organ
        # 예전 형식의 후보 정보도 prompt에 넣기 전에 routing_score를 갖도록 정규화한다.
        scored_candidates = [_with_routing_score(candidate) for candidate in candidates]
        valid_names = {candidate["model_name"] for candidate in scored_candidates}

        candidates_str = "\n".join(
            [
                "- Model Name: {model_name}, Target Organs: {target_organs}, "
                "final routing_score: {routing_score:.4f}, prior_routing_score: {prior_routing_score:.4f}, "
                "DSC prior: {dsc:.4f}, IoU prior: {iou:.4f}, "
                "overlap_score: {overlap_score:.4f}, consensus_iou: {consensus_iou:.4f}, "
                "avg_pairwise_iou: {avg_pairwise_iou:.4f}, mask_area_fraction: {mask_area_fraction:.4f}, "
                "mask_quality_score: {mask_quality_score:.4f}, quality_flags: {quality_flags}, "
                "component_count: {component_count}, boundary_roughness: {boundary_roughness:.4f}, "
                "Eval Count: {eval_count}, Selection Enabled: {selection_enabled}, "
                "Execution Status: {execution_status}, Wrapper Status: {wrapper_status}, "
                "Modality: {modality}, Architecture: {architecture}, "
                "Weight Status: {weight_status}, Weight Action: {weight_action}, "
                "Error: {error}, Source: {source_url}, Description: {description}".format(
                    model_name=candidate["model_name"],
                    target_organs=candidate.get("target_organs", ""),
                    routing_score=float(candidate.get("routing_score", 0.0) or 0.0),
                    prior_routing_score=float(candidate.get("prior_routing_score", candidate.get("routing_score", 0.0)) or 0.0),
                    dsc=float(candidate.get("dsc", 0.0) or 0.0),
                    iou=float(candidate.get("iou", 0.0) or 0.0),
                    overlap_score=float(candidate.get("overlap_score", 0.0) or 0.0),
                    consensus_iou=float(candidate.get("consensus_iou", 0.0) or 0.0),
                    avg_pairwise_iou=float(candidate.get("avg_pairwise_iou", 0.0) or 0.0),
                    mask_area_fraction=float(candidate.get("mask_area_fraction", 0.0) or 0.0),
                    mask_quality_score=float(candidate.get("mask_quality_score", 0.0) or 0.0),
                    quality_flags=candidate.get("quality_flags", []),
                    component_count=int(candidate.get("component_count", 0) or 0),
                    boundary_roughness=float(candidate.get("boundary_roughness", 0.0) or 0.0),
                    eval_count=int(candidate.get("eval_count", 0) or 0),
                    selection_enabled=bool(candidate.get("selection_enabled", True)),
                    execution_status=candidate.get("execution_status", ""),
                    wrapper_status=candidate.get("wrapper_status", ""),
                    modality=candidate.get("modality", ""),
                    architecture=candidate.get("architecture", ""),
                    weight_status=candidate.get("weight_status", ""),
                    weight_action=candidate.get("weight_action", ""),
                    error=candidate.get("error", ""),
                    source_url=candidate.get("source_url", ""),
                    description=candidate.get("description", ""),
                )
                for candidate in scored_candidates
            ]
        )
        metadata_json = json.dumps(sample_metadata or {}, ensure_ascii=False)

        try:
            # prompt는 구조화된 JSON 출력을 요구한다.
            # LLM은 선택 이유를 설명할 수 있지만, selected_model은 반드시 scorecard 안의 모델명이어야 한다.
            decision = self.chain.invoke(
                {
                    "query": query,
                    "target_organ": agent_target,
                    "metadata": metadata_json,
                    "candidates": candidates_str,
                }
            )

            selected = decision.get("selected_model")
            if selected in valid_names:
                selected_candidate = next(
                    candidate for candidate in scored_candidates if candidate["model_name"] == selected
                )
                # guardrail은 결정론적으로 동작한다.
                # LLM이 invalid 후보를 선택해도 최종 시스템 선택은 유효한 scorecard 후보로 fallback된다.
                if not _passes_hard_constraints(selected_candidate):
                    print(
                        "[Warning] LLM selected a candidate that violates hard constraints "
                        f"('{selected}'). Using fallback."
                    )
                    best_candidate = _best_candidate(scored_candidates)
                    return _fallback_decision(best_candidate, agent_target, reason_prefix="Hard-constraint fallback")

                decision["target_organ"] = agent_target
                decision["selected_score"] = float(selected_candidate["routing_score"])
                if not decision.get("reason"):
                    decision["reason"] = _selection_reason(selected_candidate, agent_target)
                if not decision.get("primary_decision_factor"):
                    decision["primary_decision_factor"] = "llm_bounded_autonomy"
                if not decision.get("evidence_used"):
                    decision["evidence_used"] = _default_evidence_used()
                if not decision.get("why_not_highest_score_model"):
                    decision["why_not_highest_score_model"] = _why_not_highest_score_model(
                        selected_candidate,
                        _best_candidate(scored_candidates),
                    )
                if not decision.get("confidence"):
                    decision["confidence"] = "medium"
                return decision

            print(f"[Warning] LLM output invalid model name: '{selected}'. Expected one of: {valid_names}")
        except Exception as exc:
            print(f"[Router Error] Parsing failed: {exc}")

        # JSON 파싱 오류, Ollama 연결 오류, 알 수 없는 모델명은 파이프라인을 중단하지 않는다.
        # 이 경우 결정론적 scorecard 기반 fallback 라우팅으로 내려간다.
        best_candidate = _best_candidate(scored_candidates)
        print(f"[Fallback] Selected {best_candidate['model_name']} automatically.")
        return _fallback_decision(best_candidate, agent_target)


def _with_routing_score(candidate: dict[str, Any]) -> dict[str, Any]:
    """prompt 생성 전에 모든 후보가 사용할 수 있는 routing_score를 갖도록 맞춘다."""
    scored = dict(candidate)
    score = scored.get("routing_score", scored.get("score"))
    if score is None:
        dsc = float(scored.get("dsc", 0.0) or 0.0)
        iou = float(scored.get("iou", 0.0) or 0.0)
        score = (0.7 * dsc) + (0.3 * iou)
    if not bool(scored.get("selection_enabled", True)):
        score = 0.0
    if scored.get("error") or scored.get("execution_status") in {"error", "not_run"}:
        score = 0.0
    scored["routing_score"] = float(score or 0.0)
    return scored


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda candidate: (
            int(candidate.get("execution_status") == "success"),
            int(bool(candidate.get("selection_enabled", True))),
            float(candidate.get("routing_score", candidate.get("score", 0.0)) or 0.0),
            float(candidate.get("dsc", 0.0) or 0.0),
            float(candidate.get("iou", 0.0) or 0.0),
        ),
    )


def _passes_hard_constraints(candidate: dict[str, Any]) -> bool:
    """LLM이 선택한 후보가 최종 선택 가능한지 검사한다."""
    if candidate.get("execution_status") != "success":
        return False
    if not bool(candidate.get("selection_enabled", True)):
        return False
    if candidate.get("error"):
        return False
    if float(candidate.get("routing_score", candidate.get("score", 0.0)) or 0.0) <= 0.0:
        return False
    return True


def _fallback_decision(
    best_candidate: dict[str, Any],
    target_organ: str,
    reason_prefix: str = "Fallback",
) -> dict[str, Any]:
    """LLM 출력이 신뢰 불가능할 때 router 형식에 맞는 fallback 결정을 반환한다."""
    score = float(best_candidate.get("routing_score", best_candidate.get("score", 0.0)) or 0.0)
    return {
        "selected_model": best_candidate["model_name"],
        "target_organ": target_organ,
        "selected_score": score,
        "reason": f"{reason_prefix}: {_selection_reason(best_candidate, target_organ)}",
        "primary_decision_factor": "hard_constraint_fallback",
        "evidence_used": _default_evidence_used(),
        "why_not_highest_score_model": "",
        "confidence": "medium",
    }


def _default_evidence_used() -> list[str]:
    return [
        "routing_score",
        "prior_routing_score",
        "overlap_score",
        "consensus_iou",
        "avg_pairwise_iou",
        "mask_area_fraction",
        "mask_quality_score",
        "quality_flags",
        "execution_status",
        "selection_enabled",
        "modality",
        "architecture",
    ]


def _why_not_highest_score_model(
    selected_candidate: dict[str, Any],
    best_candidate: dict[str, Any],
) -> str:
    if selected_candidate["model_name"] == best_candidate["model_name"]:
        return ""
    selected_score = float(selected_candidate.get("routing_score", 0.0) or 0.0)
    best_score = float(best_candidate.get("routing_score", 0.0) or 0.0)
    return (
        f"LLM selected {selected_candidate['model_name']} despite lower routing_score "
        f"({selected_score:.4f} vs {best_score:.4f}) based on bounded-autonomy evidence "
        "from the candidate scorecard."
    )


def _selection_reason(candidate: dict[str, Any], target_organ: str) -> str:
    final_score = float(candidate.get("routing_score", candidate.get("score", 0.0)) or 0.0)
    prior_score = float(candidate.get("prior_routing_score", final_score) or 0.0)
    overlap_score = float(candidate.get("overlap_score", 0.0) or 0.0)
    consensus_iou = float(candidate.get("consensus_iou", 0.0) or 0.0)
    avg_pairwise_iou = float(candidate.get("avg_pairwise_iou", 0.0) or 0.0)
    mask_area_fraction = float(candidate.get("mask_area_fraction", 0.0) or 0.0)
    mask_quality_score = float(candidate.get("mask_quality_score", 0.0) or 0.0)
    quality_flags = candidate.get("quality_flags", [])
    execution_status = str(candidate.get("execution_status", ""))
    mask_empty = bool(candidate.get("mask_empty", False))
    return (
        f"selected {candidate['model_name']} for the {target_organ} organ agent because it had "
        f"a suitable bounded-autonomy evidence profile with routing_score={final_score:.4f}. "
        f"The candidate completed with execution_status={execution_status}, mask_empty={mask_empty}, "
        f"prior_routing_score={prior_score:.4f}, and no-GT overlap evidence "
        f"(overlap_score={overlap_score:.4f}, consensus_iou={consensus_iou:.4f}, "
        f"avg_pairwise_iou={avg_pairwise_iou:.4f}, mask_area_fraction={mask_area_fraction:.4f}, "
        f"mask_quality_score={mask_quality_score:.4f}, quality_flags={quality_flags}). "
        "This is a scorecard-based routing decision, not a claim of clinical accuracy."
    )
