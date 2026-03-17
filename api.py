from math import exp

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
    menu_item: str
    allergy_list: str


class AllergyCheckResponse(BaseModel):
    answer: str
    confidence: str
    status: str


def check_allergy_system(menu_item: str, allergy_list: str) -> dict:
    prompt = f"""ตรวจสอบว่าเมนู \"{menu_item}\" มีส่วนผสมที่คนแพ้ \"{allergy_list}\" หรือไม่
กฎการตอบ:
- ถ้ามี/เสี่ยง ให้ตอบว่า \"True\"
- ถ้าไม่มี/ปลอดภัย ให้ตอบว่า \"False\"
คำตอบ:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("Metadata:", response.response_metadata)

    try:
        logprob_data = response.response_metadata.get("logprobs", {}).get("content", [])
        if logprob_data:
            chosen_logprob = logprob_data[0]["logprob"]
            confidence = exp(chosen_logprob)
        else:
            confidence = -1.0
    except (KeyError, IndexError, TypeError):
        confidence = -1.0
        print("Warning: Unable to calculate confidence score, defaulting to -1.0")

    answer = response.content.strip().lower()

    if 0.1 <= confidence <= 0.9:
        status = "MANUAL_REVIEW"
    elif answer == "true":
        status = "ALLERGY_WARN"
    else:
        status = "SAFE"

    return {
        "answer": answer,
        "confidence": f"{confidence:.2%}",
        "status": status,
    }


@app.post("/check-allergy", response_model=AllergyCheckResponse)
def check_allergy(payload: AllergyCheckRequest) -> AllergyCheckResponse:
    try:
        result = check_allergy_system(payload.menu_item, payload.allergy_list)
        return AllergyCheckResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check allergy: {exc}") from exc

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=7111, reload=True)