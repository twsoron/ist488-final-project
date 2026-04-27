"""Headless smoke test of the run_r_code tool-call loop.

Loads OPENAI_API_KEY from .streamlit/secrets.toml, sends a debug-style
prompt that should make the model call run_r_code, executes the tool
locally via r_executor.handle_tool_call, feeds the result back, and
prints the final assistant text. Verifies the streaming + tool-call
plumbing without needing a browser.

Run: py -3.12 test_tool_loop.py
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

from openai import OpenAI

from r_executor import R_EXECUTOR_TOOL, handle_tool_call


def load_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    secrets_path = Path(".streamlit/secrets.toml")
    with secrets_path.open("rb") as f:
        secrets = tomllib.load(f)
    return secrets["OPENAI_API_KEY"]


def main() -> int:
    client = OpenAI(api_key=load_api_key())

    instructions = (
        "You are a teaching assistant. When asked to verify or check R code, "
        "use the run_r_code tool to execute it and report the result."
    )
    user_prompt = (
        "Run this R code for me and tell me the result: mean(c(1, 2, 3, 4, 5))"
    )

    print(f"Prompt: {user_prompt}\n")

    kwargs = dict(
        model="gpt-4o",
        instructions=instructions,
        input=user_prompt,
        tools=[R_EXECUTOR_TOOL],
        stream=True,
        store=True,
    )

    final_text = ""
    function_calls_made: list[dict] = []
    final_response = None

    for round_num in range(1, 4):
        print(f"--- Round {round_num} ---")
        stream = client.responses.create(**kwargs)
        round_calls = []
        for event in stream:
            if event.type == "response.output_text.delta":
                final_text += event.delta
                sys.stdout.write(event.delta)
                sys.stdout.flush()
            elif event.type == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    print(f"\n[function_call] name={item.name} args={item.arguments}")
                    round_calls.append(item)
            elif event.type == "response.completed":
                final_response = event.response
        print()

        if not round_calls:
            break

        tool_outputs = []
        for fc in round_calls:
            args = json.loads(fc.arguments)
            output = handle_tool_call(fc.name, args)
            print(f"[tool_output] call_id={fc.call_id} -> {output}")
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": output,
            })
            function_calls_made.append({"name": fc.name, "args": args, "output": output})

        kwargs = dict(
            model="gpt-4o",
            previous_response_id=final_response.id,
            input=tool_outputs,
            tools=[R_EXECUTOR_TOOL],
            stream=True,
            store=True,
        )

    print("\n========== RESULT ==========")
    print(f"Final assistant text:\n{final_text}\n")
    print(f"Function calls made: {len(function_calls_made)}")

    passed = (
        len(function_calls_made) > 0
        and function_calls_made[0]["name"] == "run_r_code"
        and "3" in final_text
    )

    if passed:
        print("\nPASS — tool-call loop works end-to-end.")
        return 0
    else:
        print("\nFAIL — see output above.")
        if not function_calls_made:
            print("  (model did not call any tool)")
        elif "3" not in final_text:
            print("  (model did not include the result '3' in final text)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
