# app/llm_client.py
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env from same directory as this file
load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or app/.env")

genai.configure(api_key=api_key)
MODEL = "gemini-1.5-flash"

# The planner prompt instructs the model to return strict JSON.
PLANNER_PROMPT = """
You are a data-analyst planner. Given the TASK (natural language) below, return a strict JSON object with:
{
  "steps": [ ... ],           # ordered steps
  "final_output": {           # how to present final answers (json/object/array)
    "type": "json"|"text",
    "schema": "explain briefly desired output structure in one line"
  }
}

Each step must be an object with keys:
- "tool": one of ["duckdb","web_scrape","read_file","python_exec","plot"]
- "id": string unique id for the step (e.g., "step1")
- "action": "query" | "fetch" | "load" | "exec" | "plot"
- plus tool-specific fields:
  - duckdb + action=query -> {"query": "..."} (use read_parquet('s3://...') if S3)
  - web_scrape -> {"url": "..."}
  - read_file -> {"filename": "..."} (refer to uploaded files)
  - python_exec -> {"code": "..."} (must set `result = ...` to return)
  - plot -> {"x": "...", "y": "...", "params": {...}} (or provide code)

Important:
- Return only valid JSON and nothing else.
- Be conservative: prefer explicit actions (e.g., "query this parquet path") rather than vague steps.
- When referencing uploaded files, use the filename exactly as uploaded.
- For DuckDB queries referencing S3 parquet, assume HTTPFS available.
"""

def extract_json_from_text(text: str):
    """
    Extracts the first valid JSON object or array from any text.
    Handles Markdown fences, extra commentary, and multiple JSON blocks.
    """
    # Remove Markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

    # Try direct load
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting object
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try extracting array
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model output:\n{text}")

def plan_with_llm(task_text: str) -> dict:
    prompt = PLANNER_PROMPT + "\n\nTASK:\n" + task_text + "\n\nRespond with JSON only. Do not return a bare array; always return an object with keys 'steps' and 'final_output'."

    resp = genai.GenerativeModel(MODEL).generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=1200
        )
    )
    text = resp.text.strip()

    # Try parsing as-is first
    try:
        return _ensure_plan_shape(json.loads(text))
    except json.JSONDecodeError:
        pass  # We'll try fixing it below

    # 1. Try to extract JSON portion only
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)
    else:
        json_str = text

    # 2. Escape bad backslashes (e.g., \' -> ', unescaped \n, etc.)
    json_str = json_str.replace("\\'", "'")  # remove bad \' escapes
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)  # escape stray backslashes

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Planner returned non-JSON after repair. Error: {e}\nModel output:\n{text}")

    return _ensure_plan_shape(parsed)


def _ensure_plan_shape(parsed):
    """Ensure output always has steps + final_output keys."""
    if isinstance(parsed, list):
        parsed = {
            "steps": parsed,
            "final_output": {"type": "json", "schema": "unspecified array"}
        }
    elif not isinstance(parsed, dict):
        raise ValueError(f"Planner returned unexpected type: {type(parsed)}\n{parsed}")

    if "steps" not in parsed:
        parsed["steps"] = []
    if "final_output" not in parsed:
        parsed["final_output"] = {"type": "json", "schema": "unspecified"}

    return parsed

def repair_step_with_llm(step: dict, error_msg: str, context: str = "") -> dict:
    """
    Ask the LLM to produce a fixed version of a single step, given the error message and optional context.
    Returns a replacement step dict.
    """
    repair_prompt = f"""
A single step (as JSON) failed during execution.

STEP:
{json.dumps(step, indent=2)}

ERROR:
{error_msg}

CONTEXT:
{context}

Return a corrected step JSON object (not wrapped in an array). Only return the JSON object.
"""
    resp = genai.GenerativeModel(MODEL).generate_content(
        repair_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=400
        )
    )
    try:
        return extract_json_from_text(resp.text.strip())
    except Exception as e:
        raise ValueError(f"Repair step LLM returned non-JSON. Error: {e}\nOutput:\n{resp.text}")
