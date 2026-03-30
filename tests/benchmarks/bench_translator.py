"""Benchmark: character/token reduction on representative responses."""

import time

import pytest
from tok.translator import postprocess_response

_TOK_RESPONSE = """\
>>> t:15|usr:explain_code|agt:analyzed|state:report_ready
@thought
  |> User wants explanation of the auth module.
  |> Key areas: JWT validation, session management, middleware chain.
  |> Need to cover error handling paths too.
@msg role:assistant
  |> The auth module has three layers:
  |> 1. JWT validation middleware checks token signature and expiry
  |> 2. Session manager maintains active sessions in Redis
  |> 3. Permission resolver maps roles to allowed endpoints
  |> Error flow: invalid token returns 401, expired token triggers refresh,
  |> missing session returns 403 with re-auth prompt.
"""

_MD_RESPONSE = """\
## Auth Module Explanation

The auth module has **three layers**:

1. **JWT validation middleware** checks token signature and expiry
2. **Session manager** maintains active sessions in Redis
3. **Permission resolver** maps roles to allowed endpoints

---

### Error Flow

- Invalid token → returns `401`
- Expired token → triggers refresh
- Missing session → returns `403` with re-auth prompt
"""


@pytest.mark.benchmark
class TestTranslatorBenchmarks:
    def test_tok_reduction(self):
        start = time.perf_counter()
        result, mode = postprocess_response(_TOK_RESPONSE)
        elapsed = time.perf_counter() - start

        ratio = (1 - len(result) / len(_TOK_RESPONSE)) * 100
        print(
            f"\n  Tok: {len(_TOK_RESPONSE)} -> {len(result)} chars | "
            f"Reduction: {ratio:.1f}% | Mode: {mode} | Time: {elapsed * 1000:.2f}ms"
        )
        assert mode == "tok-native"
        assert ratio > 20

    def test_markdown_reduction(self):
        start = time.perf_counter()
        result, mode = postprocess_response(_MD_RESPONSE)
        elapsed = time.perf_counter() - start

        ratio = (1 - len(result) / len(_MD_RESPONSE)) * 100
        print(
            f"\n  MD: {len(_MD_RESPONSE)} -> {len(result)} chars | "
            f"Reduction: {ratio:.1f}% | Mode: {mode} | Time: {elapsed * 1000:.2f}ms"
        )
        assert mode == "markdown"
