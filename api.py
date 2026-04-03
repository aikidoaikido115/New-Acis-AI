from math import exp
import os
import json
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import uvicorn

load_dotenv()

app = FastAPI(title="Food Allergen Filter API")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"logprobs": True}
)


class AllergyCheckRequest(BaseModel):
    menu_data: dict
    allergy_details: List[dict]


class AllergyCheckResponse(BaseModel):
    menu_name: str
    menu_description: str
    matched_allergies: List[dict]
    confidence: str
    status: str
    raw_llm: str
    reason: str


def check_allergy_system(menu_data: dict, allergy_details: List[dict]) -> dict:
    menu_name = menu_data.get("menu_name", "")
    menu_description = menu_data.get("menu_description", "")

    allergy_names = [
        item.get("allergy_name", "").strip()
        for item in allergy_details
        if item.get("allergy_name")
    ]

    prompt = f"""ตรวจสอบเมนูอาหารว่ามีความเสี่ยงต่อสารก่อภูมิแพ้หรือไม่

menu_name: {menu_name}
menu_description: {menu_description}
allergy_names: {json.dumps(allergy_names, ensure_ascii=False)}

ให้ตอบเป็น JSON เท่านั้นในรูปแบบนี้:
{{
  "is_risky": true หรือ false,
  "matched_allergy_names": ["ชื่อสารก่อภูมิแพ้ที่พบ"],
  "self_check": {{
    "is_consistent": true หรือ false,
    "reason": "เหตุผลสั้นๆ กระชับ"
  }}
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {
            "is_risky": False,
            "matched_allergy_names": [],
            "self_check": {"is_consistent": False, "reason": "parse JSON ไม่สำเร็จ"},
        }

    try:
        logprob_data = response.response_metadata.get("logprobs", {}).get("content", [])
        if logprob_data:
            chosen_logprob = logprob_data[0]["logprob"]
            confidence = exp(chosen_logprob)
        else:
            confidence = -1.0
    except (KeyError, IndexError, TypeError):
        confidence = -1.0

    matched_names = [name.strip() for name in parsed.get("matched_allergy_names", []) if isinstance(name, str)]

    matched_allergies = [
        {
            "allergy_id": item.get("allergy_id"),
            "allergy_name": item.get("allergy_name"),
            "count": item.get("count", 1),
        }
        for item in allergy_details
        if item.get("allergy_name") in matched_names
    ]

    if 0.1 <= confidence <= 0.9:
        status = "MANUAL_REVIEW"
    elif bool(parsed.get("is_risky")):
        status = "ALLERGY_WARN"
    else:
        status = "SAFE"

    self_check = parsed.get("self_check", {}) if isinstance(parsed, dict) else {}
    reason = ""
    if isinstance(self_check, dict):
        reason = str(self_check.get("reason", "")).strip()

    return {
        "menu_name": menu_name,
        "menu_description": menu_description,
        "matched_allergies": matched_allergies,
        "confidence": f"{confidence:.2%}" if confidence >= 0 else "N/A",
        "status": status,
        "raw_llm": response.content,
        "reason": reason,
    }


@app.post("/check-allergy", response_model=AllergyCheckResponse)
def check_allergy(payload: AllergyCheckRequest) -> AllergyCheckResponse:
    try:
        result = check_allergy_system(payload.menu_data, payload.allergy_details)
        return AllergyCheckResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check allergy: {exc}") from exc

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    # Heroku injects PORT at runtime; keep API_PORT as local fallback.
    port = int(os.getenv("PORT", os.getenv("API_PORT", "7111")))
    uvicorn.run("api:app", host=host, port=port, reload=False)