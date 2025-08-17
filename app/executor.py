# app/executor.py
import traceback, time
from typing import Dict, Any
# ---- MODIFICATION START: Import pandas ----
import pandas as pd
# ---- MODIFICATION END ----
from .tools import duckdb_query, web_scrape, read_uploaded_file, plot_scatter_with_regression
from .llm_client import repair_step_with_llm

MAX_REPAIR_ATTEMPTS = 2

def execute_plan(plan: Dict[str, Any], uploaded_files: Dict[str, bytes], timeout_seconds: int = 150):
    """
    Execute a plan (dict produced by planner). Steps are run sequentially and their
    results are stored in results dict keyed by step id.
    """
    start = time.time()
    results = {}
    errors = {}

    for step in plan.get("steps", []):
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Plan execution timed out")

        step_id = step.get("id") or f"step_{len(results)}"
        tool = step.get("tool")
        action = step.get("action")
        attempt = 0

        while attempt <= MAX_REPAIR_ATTEMPTS:
            attempt += 1
            try:
                if tool == "web_scrape":
                    url = step["url"]
                    res = web_scrape(url)

                # ---- MODIFICATION START: Handle missing files gracefully ----
                elif tool == "read_file":
                    fname = step["filename"]
                    if fname not in uploaded_files:
                        # If a required file is not uploaded, return an empty DataFrame
                        # to allow the plan to continue gracefully instead of crashing.
                        print(f"Warning: File '{fname}' not found in uploads. Continuing with an empty DataFrame.")
                        res = pd.DataFrame()
                    else:
                        res = read_uploaded_file(uploaded_files[fname], fname)
                # ---- MODIFICATION END ----

                elif tool == "duckdb":
                    sql = step["query"]
                    # To allow duckdb to query results from previous steps,
                    # we pass the results dict to the function.
                    # The function can then register them as temporary tables.
                    # For this simple implementation, we assume queries are self-contained
                    # or operate on files. A more advanced version would handle this.
                    res = duckdb_query(sql)

                elif tool == "python_exec":
                    # For python_exec we provide previous results as variables result_step_<id>
                    code = step["code"]
                    # prepare globals
                    globals_dict = {}
                    for k,v in results.items():
                        globals_dict[f"result_{k}"] = v
                    # execute (dangerous in prod; sandbox recommended)
                    exec(code, globals_dict)
                    # result should be present as `result` variable
                    res = globals_dict.get("result", None)

                elif tool == "plot":
                    # a small helper expecting x and y to be column names and df reference id
                    df_ref = step.get("df_ref")
                    x = step.get("x")
                    y = step.get("y")
                    if df_ref not in results:
                        raise KeyError(f"df_ref '{df_ref}' not present in prior results")
                    df = results[df_ref]
                    res = plot_scatter_with_regression(df, x, y, dotted=step.get("dotted", True))
                else:
                    raise ValueError(f"Unknown tool: {tool}")

                # success
                results[step_id] = res
                break

            except Exception as e:
                tb = traceback.format_exc()
                errors[step_id] = {"error": str(e), "traceback": tb}
                # ask LLM to repair this step (give the failing step + error + small context)
                try:
                    repaired = repair_step_with_llm(step, str(e), context="")
                    # replace step with repaired and retry
                    step = repaired
                    continue
                except Exception as repair_e:
                    # cannot repair — break and bubble up
                    errors[step_id]["repair_error"] = str(repair_e)
                    raise RuntimeError(f"Step {step_id} failed and repair failed: {e}; repair_err: {repair_e}")

    return {"results": results, "errors": errors}