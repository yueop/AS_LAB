from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import PipelineConfig


# 1. Pydantic 모델을 조금 더 명확하게 정의 (설명을 추가하여 LLM 이해도 향상)
class RoutingDecision(BaseModel):
    selected_model: str = Field(description="The exact name of the selected model. MUST be exactly one of the names from the Candidate Models list (e.g., 'unet_lung', 'medsam').")
    target_organ: str = Field(description="The target organ requested by the user.")
    reason: str = Field(description="A brief explanation of why this model was chosen based on the DSC and IoU metrics.")

class LLMRouter:
    """Routes each request using LangChain and ChatOllama with strictly enforced JSON output."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        
        self.llm = ChatOllama(
            model=config.llm_model, 
            temperature=0.0,
            format="json" 
        )
        
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)
        
        # 2. 프롬프트 구조를 간결하고 강력하게 변경
        self.prompt = PromptTemplate(
            template="""You are an expert medical AI routing agent.

{format_instructions}

[Context]
- User Query: {query}
- Default Target Organ: {target_organ}
- Image Metadata: {metadata}

[Candidate Models & Metrics]
{candidates}

[Instructions]
Analyze the candidates. Your ONLY task is to return a JSON object with the fields specified above.
The 'selected_model' field MUST exactly match one of the model names from the Candidate list.
Select the model that has the HIGHEST DSC and IoU metrics.
DO NOT select any model that has null metrics or explicitly lists an "error".
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
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No candidate models were provided.")

        valid_names = {c["model_name"] for c in candidates}
        
        # 리스트 형태의 candidates를 LLM이 읽기 편한 문자열로 변환
        candidates_str = "\n".join([
            f"- Model Name: {c['model_name']}, Metrics: {json.dumps(c.get('metrics', {}))}" 
            for c in candidates
        ])
        
        metadata_json = json.dumps(sample_metadata or {}, ensure_ascii=False)

        try:
            decision = self.chain.invoke({
                "query": query,
                "target_organ": self.config.target_organ,
                "metadata": metadata_json,
                "candidates": candidates_str
            })

            # LLM이 뱉은 결과 검증
            selected = decision.get("selected_model")
            if selected in valid_names:
                return decision
            else:
                print(f"[Warning] LLM output invalid model name: '{selected}'. Expected one of: {valid_names}")

        except Exception as e:
            print(f"[Router Error] Parsing failed: {str(e)}")

        # Fallback 로직 (에러 시 무조건 점수 높은 거 선택)
        best_candidate = max(candidates, key=lambda c: (
            c.get("metrics", {}).get("dsc", 0.0) if c.get("metrics") else 0.0
        ))
        
        print(f"[Fallback] Selected {best_candidate['model_name']} automatically.")
        return {
            "selected_model": best_candidate["model_name"],
            "target_organ": self.config.target_organ,
            "reason": "Fallback: Selected best candidate by DSC score due to LLM routing failure.",
        }