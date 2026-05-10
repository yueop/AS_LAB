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


class LLMRouter:
    """Organ-aware LLM orchestrator with strictly enforced JSON output."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

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
Analyze the candidate score table. Your ONLY task is to return a JSON object with the fields specified above.
The selected_model field MUST exactly match one of the model names from the Candidate Model Score Table.
Select the model that has the highest final routing_score for this target organ agent.
Use prior_routing_score as prior validation evidence and overlap_score/consensus_iou as no-GT mask agreement evidence.
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
        scored_candidates = [_with_routing_score(candidate) for candidate in candidates]
        valid_names = {candidate["model_name"] for candidate in scored_candidates}

        candidates_str = "\n".join(
            [
                "- Model Name: {model_name}, Target Organs: {target_organs}, "
                "final routing_score: {routing_score:.4f}, prior_routing_score: {prior_routing_score:.4f}, "
                "DSC prior: {dsc:.4f}, IoU prior: {iou:.4f}, "
                "overlap_score: {overlap_score:.4f}, consensus_iou: {consensus_iou:.4f}, "
                "avg_pairwise_iou: {avg_pairwise_iou:.4f}, mask_area_fraction: {mask_area_fraction:.4f}, "
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
                expected = _best_candidate(scored_candidates)
                if selected != expected["model_name"]:
                    print(
                        "[Warning] LLM did not choose the highest-score model "
                        f"('{selected}' vs '{expected['model_name']}'). Using highest-score model."
                    )
                    return _fallback_decision(expected, agent_target, reason_prefix="Score guardrail")

                decision["target_organ"] = agent_target
                decision["selected_score"] = float(selected_candidate["routing_score"])
                decision["reason"] = _selection_reason(selected_candidate, agent_target)
                return decision

            print(f"[Warning] LLM output invalid model name: '{selected}'. Expected one of: {valid_names}")
        except Exception as exc:
            print(f"[Router Error] Parsing failed: {exc}")

        best_candidate = _best_candidate(scored_candidates)
        print(f"[Fallback] Selected {best_candidate['model_name']} automatically.")
        return _fallback_decision(best_candidate, agent_target)


def _with_routing_score(candidate: dict[str, Any]) -> dict[str, Any]:
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
            float(candidate.get("routing_score", candidate.get("score", 0.0)) or 0.0),
            float(candidate.get("dsc", 0.0) or 0.0),
            float(candidate.get("iou", 0.0) or 0.0),
        ),
    )


def _fallback_decision(
    best_candidate: dict[str, Any],
    target_organ: str,
    reason_prefix: str = "Fallback",
) -> dict[str, Any]:
    score = float(best_candidate.get("routing_score", best_candidate.get("score", 0.0)) or 0.0)
    return {
        "selected_model": best_candidate["model_name"],
        "target_organ": target_organ,
        "selected_score": score,
        "reason": f"{reason_prefix}: {_selection_reason(best_candidate, target_organ)}",
    }


def _selection_reason(candidate: dict[str, Any], target_organ: str) -> str:
    final_score = float(candidate.get("routing_score", candidate.get("score", 0.0)) or 0.0)
    prior_score = float(candidate.get("prior_routing_score", final_score) or 0.0)
    overlap_score = float(candidate.get("overlap_score", 0.0) or 0.0)
    consensus_iou = float(candidate.get("consensus_iou", 0.0) or 0.0)
    return (
        f"selected {candidate['model_name']} for the {target_organ} organ agent because it had "
        f"the highest final routing_score ({final_score:.4f}), combining prior score "
        f"({prior_score:.4f}) with mask-overlap evidence "
        f"(overlap_score={overlap_score:.4f}, consensus_iou={consensus_iou:.4f})."
    )
