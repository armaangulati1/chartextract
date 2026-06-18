from typing import Optional

from fastapi import FastAPI
from langfuse import get_client
from pydantic import BaseModel, Field

from db import recent, save_extraction
from schema import ExtractionOutput
from extractor import extract


app = FastAPI(title="Clinical Text Extractor API")

class ExtractRequest(BaseModel):
    text: str
    review_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="confidence cutoff for needs_review; defaults to REVIEW_CONFIDENCE_THRESHOLD env",
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=ExtractionOutput)
def extract_endpoint(req: ExtractRequest):
    result = extract(req.text, review_threshold=req.review_threshold)
    save_extraction(req.text, result.model_dump(mode="json"))
    get_client().flush()
    return result

@app.get("/history")
def history():
    return recent()