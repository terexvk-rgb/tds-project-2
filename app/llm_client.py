# app/llm_client.py
import os
import re
import json
import base64
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
  "steps": [ ... ],
  "final_output": {
    "type": "json",
    "schema": "JSON object with exactly these keys: edge_count (number), highest_degree_node (string), average_degree (number), density (number), shortest_path_alice_eve (number), network_graph (base64 PNG string without any data:image prefix), degree_histogram (base64 PNG string without any data:image prefix)"
  }
}

Each step must have:
- "tool": one of ["duckdb","web_scrape","read_file","python_exec","plot"]
- "id": unique string (e.g., "step1")
- "action": "query" | "fetch" | "load" | "exec" | "plot"
- Tool-specific fields:
  - duckdb: { "query": "..." }
  - web_scrape: { "url": "..." }
  - read_file: { "filename": "..." }
  - python_exec: { "code": "..." }
  - plot: { "x": "...", "y": "...", "params": {...} } or { "code": "..." }

Important:
- Return ONLY valid JSON.
- No markdown, no explanations, no extra text.
- Always include both 'steps' and 'final_output'.
- Images must be valid base64 PNG strings without any prefix like data:image/png;base64,
"""

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown fences and whitespace."""
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()

def _repair_json_common_errors(text: str) -> str:
    """Repair common JSON mistakes from LLM output."""
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # remove trailing commas
    text = text.replace("\\'", "'")  # fix bad escaped single quotes
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)  # escape stray backslashes
    return text

def _strip_data_url_prefix(b64_str: str) -> str:
    """Remove data:image/png;base64, prefix if present."""
    if isinstance(b64_str, str) and b64_str.startswith("data:image"):
        return b64_str.split(",", 1)[-1]
    return b64_str

def extract_json_from_text(text: str):
    """Extract the first valid JSON object or array from any text."""
    cleaned = _strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(_repair_json_common_errors(obj_match.group(0)))
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if arr_match:
        try:
            return json.loads(_repair_json_common_errors(arr_match.group(0)))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model output:\n{text}")

def _ensure_plan_shape(parsed):
    """Ensure parsed plan always has steps + final_output keys and required schema."""
    if isinstance(parsed, list):
        parsed = {
            "steps": parsed,
            "final_output": {"type": "json", "schema": "unspecified array"}
        }
    elif not isinstance(parsed, dict):
        raise ValueError(f"Planner returned unexpected type: {type(parsed)}\n{parsed}")

    if "steps" not in parsed or not isinstance(parsed["steps"], list):
        parsed["steps"] = []
    if "final_output" not in parsed or not isinstance(parsed["final_output"], dict):
        parsed["final_output"] = {"type": "json", "schema": "unspecified"}

    return parsed

def _inject_required_keys(result: dict):
    """Ensure result contains all required keys with safe defaults."""
    required_keys = [
        "edge_count",
        "highest_degree_node",
        "average_degree",
        "density",
        "shortest_path_alice_eve",
        "network_graph",
        "degree_histogram",
    ]
    for key in required_keys:
        if key not in result:
            result[key] = "" if "graph" in key or "histogram" in key else None
        if "graph" in key or "histogram" in key:
            result[key] = _strip_data_url_prefix(result[key])
    return result

def plan_with_llm(task_text: str) -> dict:
    """Call LLM and return a safe, repaired plan JSON."""
    prompt = f"{PLANNER_PROMPT}\n\nTASK:\n{task_text}\n\nRespond with JSON only."

    resp = genai.GenerativeModel(MODEL).generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=1500
        )
    )

    raw_text = resp.text or ""
    raw_text = raw_text.strip()

    last_err = None
    for _ in range(3):
        try:
            parsed = extract_json_from_text(raw_text)
            return _ensure_plan_shape(parsed)
        except Exception as e:
            last_err = e
            raw_text = _repair_json_common_errors(raw_text)

    raise ValueError(f"Planner returned non-JSON after repairs. Last error: {last_err}\nModel output:\n{resp.text}")

def repair_step_with_llm(step: dict, error_msg: str, context: str = "") -> dict:
    """Ask the LLM to produce a fixed version of a single step."""
    repair_prompt = f"""
A single step (as JSON) failed during execution.

STEP:
{json.dumps(step, indent=2)}

ERROR:
{error_msg}

CONTEXT:
{context}

Return a corrected step JSON object only.
"""
    resp = genai.GenerativeModel(MODEL).generate_content(
        repair_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=400
        )
    )
    return extract_json_from_text(resp.text or "")

