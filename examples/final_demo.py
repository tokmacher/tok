import json

import tok
from agent import MODEL, TOK_SYSTEM_PROMPT, client


def live_llm(tok_input):
    """Actual LLM call to OpenRouter."""
    print(f"\n[OpenRouter] Calling {MODEL}...")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TOK_SYSTEM_PROMPT},
            {"role": "user", "content": tok_input},
        ],
        stream=True,
    )
    full_response = ""
    print("\n[OpenRouter] Response: ", end="", flush=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_response += delta
    print("\n")
    return full_response


def main():
    print("--- TOK FINAL DEMONSTRATION: LIVE OPENROUTER AGENT ---")

    user_request = {
        "task": "Explain the difference between weather in Tokyo and London in Tok format",
        "verbose": False,
    }

    print("\n[Developer] Sending JSON payload:", user_request)

    # The Developer experience: Clean JSON in, Clean JSON out
    result_json = tok.bridge.execute(live_llm, user_request)

    print("\n--- DEVELOPER RESULT (Re-hydrated JSON) ---")
    print(json.dumps(result_json, indent=2))

    print(
        "\n✅ LIVE PROJECT COMPLETE: Tok v1.2 is a fully bidirectional, invisible agentic bridge."
    )


if __name__ == "__main__":
    main()
