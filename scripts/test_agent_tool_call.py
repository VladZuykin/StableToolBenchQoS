"""Smoke-test an OpenAI-compatible agent through ToolBench's LLM adapter."""

import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction


def main() -> int:
    api_key = os.getenv("AGENT_API_KEY")
    base_url = os.getenv("AGENT_API_BASE")
    model = os.getenv("AGENT_MODEL")
    missing = [name for name, value in (
        ("AGENT_API_KEY", api_key),
        ("AGENT_API_BASE", base_url),
        ("AGENT_MODEL", model),
    ) if not value]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    llm = ChatGPTFunction(model=model, openai_key=api_key, base_url=base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    llm.change_messages(
        [
            {
                "role": "user",
                "content": "What is the weather in Moscow? Use the available tool.",
            }
        ]
    )
    message, error_code, _token_usage = llm.parse(
        tools=tools,
        process_id=0,
        temperature=0,
        max_tokens=256,
    )
    if error_code != 0:
        print(f"ToolBench adapter returned error code {error_code}: {message!r}")
        return 1

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        print(f"Agent responded without a tool call: {message.get('content')!r}")
        return 1

    function = tool_calls[0]["function"]
    arguments = json.loads(function["arguments"])
    print(f"ToolBench agent adapter: OK ({model})")
    print(f"Tool call: {function['name']}({arguments})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
