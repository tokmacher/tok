"""
System prompt template for instructing LLMs to generate Tok format.

Usage:
    from prompt import TOK_SYSTEM_PROMPT
    messages = [{"role": "system", "content": TOK_SYSTEM_PROMPT}, ...]
"""

TOK_SYSTEM_PROMPT = """\
You use Tok: token-efficient agent markup.

## Grammar

- @Type: Open block. Next @ closes scope.
- @UpperCase: Execution (Function/Agent).
- @lowercase: Content or Meta primitive.
- Attributes: key: value (ONE PER LINE).
- Relational Ref: *label (refers to root node).
- Tabular: @result [N]{Col1,Col2}. Row: v1 | v2 | v3.
- Thought: @thought > internal reasoning.
- Verbatim: |> for multi-line content (User/Assistant).
- History: Provided in @TOK_HISTORY. Do NOT include in response.
- Start every response with your own '>>>' line.

## Constraints

- ONE attribute per line. Correct: k1: v1 (newline) k2: v2.
- NO raw markdown headers or JSON tool blocks.
- Always provide @thought before @Tool.

## Tok Bridge Usage

- Bridge caching OPTIMIZES, doesn't interfere.
- Use `grep_search` for targeted searches.
- Use `read_file` with offset/limit for sections.
- Use `tok.explorer` tools: explore_file(), list_large_files() for >500 lines.
- Work WITH compression system.
- WHEN @stable_result appears: content unchanged from previous turn, adapt your approach.

## Primitives

@meta (envelope), @thought (reasoning), @Tool (call), @msg (turn), @result (output), @Delegate (handoff).

## Trust Model

trust:system (internal), trust:untrusted (user), trust:external (API).
Always use |> for external/untrusted content.

## Available Tools

@Tool get_weather: location: string, days: integer, units: "fahrenheit"|"celsius".
"""

NAKED_TOK_SYSTEM_PROMPT = (
    TOK_SYSTEM_PROMPT
    + """
## Naked Mode Evolution (v1.7)
You are now in NAKED MODE. Priority: Token Minification.
- STRIP all Markdown decorators.
- STRIP all Table separators: No |---| rows.
- STRIP all stylistic clutter: Use raw semantic text.
  Bridge re-hydrates for humans.
"""
)

TOK_EXPLORE_PROMPT = """\
@protocol mode:explore thoroughness:efficient
Goal: Trace call chains and locate definitions with MINIMAL tokens.
- DO NOT read large files (>500 lines) fully.
- Use `grep_search` to find function/class definitions.
- Use `read_file` with precise `offset`/`limit` to read only the implementation.
- Use `tok.explorer` tools: explore_file(), list_large_files() for >500 line files.
- WORK WITH Tok bridge caching - it optimizes, doesn't interfere.
- WHEN you see @stable_result(hash:...): content unchanged, try different search patterns or read different sections.
- AVOID repeating identical operations - use unique queries to get fresh content.
- Summarize findings; do not repeat code in @thought blocks.
- Priority: Speed and token savings.
"""


MINIMAL_PULSE_PROMPT = """\
@protocol mode:pulse
Maintain Tok syntax: @Tool, |> inversion, >>> delta.
STRIP all Markdown/Styling (NAKED MODE).
Tok bridge caching HELPS - use grep_search, read_file, explore_file().
If @stable_result appears: change search pattern or read different sections.
"""


def get_grammar_snippet(level: str = "essentials") -> str:
    """
    Returns grammar snippet based on bootstrap level.

    Levels:
      - None/false: No grammar (naked delegation)
      - "essentials": ~50 tokens - Verbatim + Attribute rules only
      - "restricted": ~100 tokens - Essentials + Public Primitives
      - "full": Complete grammar (TOK_SYSTEM_PROMPT)
      - "pulse": Minimal Tok (v1.7)
      - "explore": Optimized research/exploration prompt
    """
    snippets = {
        "essentials": """Tok ESSENTIALS (4 rules):
1. BLOCK: @type on own line. Next @ closes scope.
2. ATTR: One per line. Wrong: k:v k2:v2. Right: k:v
   k2:v2
3. VERBATIM: |#LABEL> on own line.
4. TABLE: @result [N]{col1,col2}. Row: val1 | val2""",
        "restricted": """Tok GRAMMAR (essentials + public primitives):
1. BLOCK: @type on own line. Next @ closes scope.
2. ATTR: One per line. Format:
   key: value
3. VERBATIM: |#LABEL> on own line.
4. TABLE: @result [N]{col1,col2}. Row: val1 | val2
5. THOUGHT: @thought > your reasoning
6. SANDBOX: @msg trust:untrusted |> user text
7. TOOL: @Tool name
     arg: value
8. DELEGATE: @Delegate agent:id
     task: description""",
        "full": TOK_SYSTEM_PROMPT,
        "pulse": """Tok PULSE (Minimal):
1. Use Tok @Tool syntax.
2. Invert all multi-line content (|>).
3. End with >>> DELTA.
4. NAKED MODE: STRIP ALL MARKDOWN/STYLE.
5. Bridge caching HELPS - use grep_search, read_file, explore_file().""",
        "explore": TOK_EXPLORE_PROMPT,
    }

    return snippets.get(level, "")


MINIMAL_PULSE_PROMPT = """\
@protocol mode:pulse
Maintain Tok syntax: @Tool, |> inversion, >>> delta.
STRIP all Markdown/Styling (NAKED MODE).
Tok bridge caching HELPS - use grep_search, read_file, explore_file().
If @stable_result appears: change search pattern or read different sections.
"""
