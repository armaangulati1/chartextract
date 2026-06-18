from fastapi import FastAPI
from langfuse import get_client
from pydantic import BaseModel

from db import recent, save_extraction
from schema import OncologyExtract
from extractor import extract


app = FastAPI(title="Clinical Text Extractor API")

class ExtractRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=OncologyExtract)
def extract_endpoint(req: ExtractRequest):
    result = extract(req.text)
    save_extraction(req.text, result.model_dump(mode="json"))
    get_client().flush()
    return result

@app.get("/history")
def history():
    return recent()