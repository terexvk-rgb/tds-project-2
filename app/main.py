# app/main.py
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import asyncio
import pandas as pd
from app.worker import execute_task

load_dotenv()

app = FastAPI()

def make_serializable(obj):
    """Recursively convert Pandas objects to JSON-serializable types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

@app.post("/")
async def analyze(files: List[UploadFile] = File(...)):
    uploaded = {}
    question_text = None

    for f in files:
        content = await f.read()
        if f.filename.lower().startswith("questions") or f.filename.lower().endswith("questions.txt"):
            question_text = content.decode("utf-8")
        else:
            uploaded[f.filename] = content

    if not question_text:
        raise HTTPException(status_code=400, detail="questions.txt is required")

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, execute_task, question_text, uploaded, 170),
            timeout=170
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out (~170s limit)")

    # Ensure all outputs are JSON serializable
    result = make_serializable(result)

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
