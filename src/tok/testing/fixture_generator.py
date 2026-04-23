"""Tool for generating replay fixtures for testing."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, cast


class FixtureGenerator:
    """Generate replay fixtures with different characteristics."""

    def __init__(
        self,
        templates_file: str = "tests/fixtures/replay/metadata_templates.json",
    ) -> None:
        self.templates_file = Path(templates_file)
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, Any]:
        """Load metadata templates."""
        if not self.templates_file.exists():
            return {}
        try:
            with open(self.templates_file) as f:
                data = json.load(f)
                return cast("dict[str, Any]", data.get("templates", {}))
        except Exception:
            return {}

    def generate_coding_session(
        self,
        name: str,
        turns: int = 5,
        template: str = "standard_claude",
        complexity: str = "medium",
    ) -> tuple[str, str]:
        """Generate a coding session fixture."""
        # Generate session messages
        messages = []

        for i in range(turns):
            # User message
            if complexity == "simple":
                user_content = f"Fix the bug in file{i}.py"
            elif complexity == "medium":
                user_content = f"Refactor the {['authentication', 'database', 'api', 'ui', 'utils'][i % 5]} module"
            else:  # complex
                user_content = f"Implement a comprehensive {['microservices', 'event sourcing', 'cqrs', 'graphql', 'reactive'][i % 5]} architecture"

            # Assistant response with tools
            tools: list[dict[str, Any]] = []
            if i == 0:
                tools.extend(
                    [
                        {
                            "type": "tool_use",
                            "id": f"t{i}1",
                            "name": "view_file",
                            "input": {"path": f"src/module{i}.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": f"t{i}2",
                            "name": "grep",
                            "input": {"query": "bug"},
                        },
                    ]
                )
            elif i < turns - 1:
                tools.extend(
                    [
                        {
                            "type": "tool_use",
                            "id": f"t{i}1",
                            "name": "edit",
                            "input": {
                                "path": f"src/module{i}.py",
                                "old_string": "old_code",
                                "new_string": "new_code",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": f"t{i}2",
                            "name": "bash",
                            "input": {"command": "python -m pytest tests/test_module{i}.py"},
                        },
                    ]
                )
            else:
                tools.append(
                    {
                        "type": "tool_use",
                        "id": f"t{i}1",
                        "name": "write_to_file",
                        "input": {
                            "path": "docs/module{i}.md",
                            "content": "# Documentation",
                        },
                    }
                )

            messages.append({"role": "user", "content": user_content})

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": tools,
            }
            messages.append(assistant_msg)

            # Add tool results
            for tool in tools:
                messages.append(
                    {
                        "role": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": f"Result from {tool['name']}",
                    }
                )

        # Create fixture JSONL content
        fixture_lines = []
        for i in range(0, len(messages), 3):  # Group into turns
            if i + 2 < len(messages):
                turn_messages = [messages[i], messages[i + 1], messages[i + 2]]
                fixture_lines.append(json.dumps({"messages": turn_messages}))

        fixture_content = "\n".join(fixture_lines)

        # Get metadata
        metadata = self.templates.get(template, self.templates.get("standard_claude", {}))
        metadata = metadata.copy()
        metadata["name"] = name
        metadata["turns"] = turns
        metadata["complexity"] = complexity

        return fixture_content, json.dumps(metadata, indent=2)

    def generate_search_session(
        self, name: str, searches: int = 8, template: str = "standard_claude"
    ) -> tuple[str, str]:
        """Generate a search-intensive session fixture."""
        search_terms = [
            "compression",
            "memory",
            "cache",
            "error",
            "tool",
            "runtime",
            "bridge",
            "config",
        ]
        messages = []

        for i in range(searches):
            # User message
            messages.append(
                {
                    "role": "user",
                    "content": f"Search for {search_terms[i % len(search_terms)]} in the codebase",
                }
            )

            # Assistant response with multiple searches
            tools: list[dict[str, Any]] = []
            for j in range(3):  # Multiple searches per turn
                tools.append(
                    {
                        "type": "tool_use",
                        "id": f"s{i}{j}",
                        "name": "grep_search",
                        "input": {
                            "search_path": "src",
                            "query": search_terms[(i + j) % len(search_terms)],
                        },
                    }
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": tools,
            }
            messages.append(assistant_msg)

            # Add tool results
            for tool in tools:
                messages.append(
                    {
                        "role": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": f"Found {random.randint(5, 20)} matches",
                    }
                )

        # Create fixture JSONL content
        fixture_lines = []
        for i in range(0, len(messages), 3):
            if i + 2 < len(messages):
                turn_messages = [messages[i], messages[i + 1], messages[i + 2]]
                fixture_lines.append(json.dumps({"messages": turn_messages}))

        fixture_content = "\n".join(fixture_lines)

        # Get metadata
        metadata = self.templates.get(template, self.templates.get("standard_claude", {}))
        metadata = metadata.copy()
        metadata["name"] = name
        metadata["searches"] = searches
        metadata["type"] = "search_intensive"

        return fixture_content, json.dumps(metadata, indent=2)

    def generate_high_pressure_session(
        self, name: str, repeats: int = 6, template: str = "high_pressure"
    ) -> tuple[str, str]:
        """Generate a high-pressure session with repeats and errors."""
        messages = []

        for i in range(repeats):
            # User message - repeat same request
            messages.append({"role": "user", "content": "Read the main configuration file"})

            # Assistant response with repeated file reads
            tools: list[dict[str, Any]] = [
                {
                    "type": "tool_use",
                    "id": f"r{i}1",
                    "name": "view_file",
                    "input": {"path": "src/config.py"},
                },
                {
                    "type": "tool_use",
                    "id": f"r{i}2",
                    "name": "view_file",
                    "input": {"path": "src/config.py"},
                },  # Repeat
            ]

            # Add some error cases
            if i % 3 == 0:
                tools.append(
                    {
                        "type": "tool_use",
                        "id": f"r{i}3",
                        "name": "view_file",
                        "input": {"path": "nonexistent.py"},
                    }
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": tools,
            }
            messages.append(assistant_msg)

            # Add tool results
            for tool in tools:
                if "nonexistent" in tool["input"]["path"]:
                    content = "File not found"
                else:
                    content = "Configuration file content"
                messages.append(
                    {
                        "role": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": content,
                    }
                )

        # Create fixture JSONL content
        fixture_lines = []
        for i in range(0, len(messages), 3):
            if i + 2 < len(messages):
                turn_messages = [messages[i], messages[i + 1], messages[i + 2]]
                fixture_lines.append(json.dumps({"messages": turn_messages}))

        fixture_content = "\n".join(fixture_lines)

        # Get metadata
        metadata = self.templates.get(template, self.templates.get("high_pressure", {}))
        metadata = metadata.copy()
        metadata["name"] = name
        metadata["repeats"] = repeats
        metadata["type"] = "high_pressure"

        return fixture_content, json.dumps(metadata, indent=2)

    def save_fixture(
        self,
        name: str,
        fixture_content: str,
        metadata: str,
        output_dir: str = "tests/fixtures/replay",
    ) -> None:
        """Save fixture and metadata to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save fixture
        fixture_file = output_path / f"{name}.jsonl"
        fixture_file.write_text(fixture_content)

        # Save metadata
        meta_file = output_path / f"{name}.jsonl.meta.json"
        meta_file.write_text(metadata)


def main() -> None:
    """CLI for fixture generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate replay fixtures")
    parser.add_argument("--type", choices=["coding", "search", "pressure"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--template", default="standard_claude")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--searches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--complexity",
        choices=["simple", "medium", "complex"],
        default="medium",
    )
    parser.add_argument("--output", default="tests/fixtures/replay")

    args = parser.parse_args()

    generator = FixtureGenerator()

    if args.type == "coding":
        fixture, metadata = generator.generate_coding_session(args.name, args.turns, args.template, args.complexity)
    elif args.type == "search":
        fixture, metadata = generator.generate_search_session(args.name, args.searches, args.template)
    elif args.type == "pressure":
        fixture, metadata = generator.generate_high_pressure_session(args.name, args.repeats, args.template)

    generator.save_fixture(args.name, fixture, metadata, args.output)


if __name__ == "__main__":
    main()
