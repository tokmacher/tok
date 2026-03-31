import re
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = os.environ.get("TOK_VIS_LOG", SCRIPT_DIR.parent / "tokviz.txt")
OUTPUT_FILE = os.environ.get(
    "TOK_VIS_OUTPUT", SCRIPT_DIR.parent / "viz" / "tok_metrics.json"
)


def parse_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found.")
        return

    metrics = {
        "costs": [],
        "token_savings": {
            "semantic_dedup": 0,
            "recent_file": 0,
            "file_diff": 0,
            "grep_diff": 0,
            "raw_cached": 0,
            "others": 0,
        },
        "prompt_optimization": [],
        "requests": [],
    }

    # Regex patterns
    cost_pattern = re.compile(
        r"cost: baseline=\$([\d.]+) actual=\$([\d.]+) saved=\$([\d.]+) \(([\d.]+)%\) \[(.*)\]"
    )
    token_saved_pattern = re.compile(
        r"Tool results: ~(\d+) tokens saved (\{.*\})"
    )
    prompt_optimized_pattern = re.compile(
        r"tok_prompt_optimized: system prompt reduced from (\d+) to (\d+) chars"
    )
    request_pattern = re.compile(
        r"HTTP Request: (POST|GET) (.*) \"HTTP/1.1 (.*)\""
    )

    with open(LOG_FILE) as f:
        for line in f:
            # Parse Costs
            cost_match = cost_pattern.search(line)
            if cost_match:
                metrics["costs"].append(
                    {
                        "baseline": float(cost_match.group(1)),
                        "actual": float(cost_match.group(2)),
                        "saved": float(cost_match.group(3)),
                        "percentage": float(cost_match.group(4)),
                        "model": cost_match.group(5),
                    }
                )

            # Parse Token Savings
            token_match = token_saved_pattern.search(line)
            if token_match:
                try:
                    details = json.loads(
                        token_match.group(2).replace("'", '"')
                    )
                    for key, val in details.items():
                        if key in metrics["token_savings"]:
                            metrics["token_savings"][key] += val
                        else:
                            metrics["token_savings"]["others"] += val
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass

            # Parse Prompt Optimization
            prompt_match = prompt_optimized_pattern.search(line)
            if prompt_match:
                metrics["prompt_optimization"].append(
                    {
                        "original": int(prompt_match.group(1)),
                        "reduced": int(prompt_match.group(2)),
                    }
                )

            # Parse Requests
            request_match = request_pattern.search(line)
            if request_match:
                metrics["requests"].append(
                    {
                        "method": request_match.group(1),
                        "url": request_match.group(2),
                        "status": request_match.group(3),
                    }
                )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Parsed {len(metrics['costs'])} cost entries, {len(metrics['prompt_optimization'])} prompt optimizations."
    )
    print(f"Metrics saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_logs()
