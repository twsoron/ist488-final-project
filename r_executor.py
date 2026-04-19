import subprocess
import json

# --------------------------------------------------
# R Code Executor Tool
# --------------------------------------------------
# This file contains:
#   1. The Python function that actually runs R code (run_r_code)
#   2. The tool definition to register it with the OpenAI Responses API
#   3. A helper to handle the tool call when the model invokes it
# --------------------------------------------------

# Blocklist of dangerous R functions that should never be executed
BLOCKED_FUNCTIONS = [
    "system(", "shell(", "file.remove(", "unlink(",
    "writeLines(", "write.csv(", "write.table(", "sink(",
    "Sys.setenv(", "install.packages(", "source(",
]

TIMEOUT_SECONDS = 5


def run_r_code(code: str) -> dict:
    """
    Executes a string of R code in a subprocess with a timeout and blocklist.
    Returns a dict with 'output' or 'error'.
    """
    # Check for blocked functions
    for blocked in BLOCKED_FUNCTIONS:
        if blocked in code:
            return {
                "error": f"Execution blocked: '{blocked}' is not permitted for security reasons."
            }

    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return {"output": result.stdout.strip() or "(no output)"}
        else:
            return {"error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {TIMEOUT_SECONDS} seconds."}
    except FileNotFoundError:
        return {"error": "R is not installed or not found on this system."}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------
# OpenAI Responses API tool definition
# Pass this in the 'tools' list when calling the API
# --------------------------------------------------

R_EXECUTOR_TOOL = {
    "type": "function",
    "name": "run_r_code",
    "description": (
        "Executes a snippet of R code and returns the output or error message. "
        "Use this when a student shares R code that needs to be run to diagnose a bug "
        "or verify that a fix works. Do not use for assignment solutions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The R code to execute. Must be a complete, runnable snippet.",
            }
        },
        "required": ["code"],
    },
}


# --------------------------------------------------
# Tool call handler
# Call this when the model returns a tool_use block
# --------------------------------------------------

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Routes tool calls from the model to the appropriate Python function.
    Returns the result as a JSON string to pass back to the model.
    """
    if tool_name == "run_r_code":
        result = run_r_code(tool_input["code"])
        return json.dumps(result)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
