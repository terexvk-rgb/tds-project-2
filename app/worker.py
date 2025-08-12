# app/worker.py
from .llm_client import plan_with_llm
from .executor import execute_plan
from typing import Dict
import json, time

def execute_task(question_text: str, uploaded_files: Dict[str, bytes], timeout_seconds=160):
    """
    High-level worker:
    - ask LLM to produce plan (JSON)
    - execute plan
    - assemble final output as specified by plan["final_output"]
    """
    start = time.time()
    plan = plan_with_llm(question_text)
    # plan is expected to be a dict with keys "steps" and "final_output"
    exec_result = execute_plan(plan, uploaded_files, timeout_seconds=timeout_seconds)

    # assemble final output according to plan
    final_spec = plan.get("final_output", {"type": "json", "schema": "raw"})
    if final_spec.get("type") == "json":
        # If plan included step ids for outputs, allow plan to list them
        # We will try to return a JSON object mapping step ids to results unless plan asks otherwise
        output = {}
        # If plan has "assemble" instructions, let LLM do assembly:
        assemble = plan.get("assemble")
        if assemble:
            # give LLM the raw results and ask it to assemble according to assemble instruction
            # to keep simple, return results and let caller interpret
            output = {"assembled_by_plan": assemble, "raw_results": exec_result["results"]}
        else:
            output = exec_result["results"]
        return output
    else:
        # other output types: just return raw
        return exec_result
