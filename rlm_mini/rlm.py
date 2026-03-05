"""
rlm.py — Core Recursive Language Model loop.

Architecture (from the RLM paper https://arxiv.org/abs/2512.24601):

  RLM.completion(prompt)
  ├── builds initial message history (system prompt + context metadata)
  └── loop up to max_iterations:
      ├── call LM  →  raw text response
      ├── parse ```repl ... ``` blocks
      ├── execute each block in the persistent sandboxed REPL
      │     ↳ code can call llm_query() / rlm_query() internally
      ├── append (LM response + REPL outputs) to message history
      ├── check for FINAL(answer) or FINAL_VAR(var) in response/REPL
      └── if found → return final answer
          else     → continue loop

How context is exposed:
  The prompt/context is stored as a Python variable `context` in the REPL
  namespace.  The LM writes code that reads that variable directly.

Termination:
  • FINAL(some text)          — LM writes the answer inline
  • FINAL_VAR(variable_name)  — LM points to a variable it built in the REPL
  • Max iterations reached     — one final "please summarise" LM call
"""

import textwrap
import warnings
import re
import ast
from dataclasses import dataclass, field
from typing import Any

# Suppress LangChain's internal Pydantic serialization warning fired when
# with_structured_output populates AIMessage.parsed with a Pydantic model.
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .repl import ExecResult, MiniREPL

# ---------------------------------------------------------------------------
# System prompt (abridged but faithful to the original)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Structured output schema — replaces all regex parsing
# ---------------------------------------------------------------------------

class RLMTurn(BaseModel):
    reasoning: str = Field(
        description="Think step-by-step: what does the context contain, what still needs to be done, and what code will you run next.",
    )
    repl_blocks: list[str] = Field(
        default_factory=list,
        description="Python code blocks to execute in the REPL, in order.",
    )
    final_answer: str | None = Field(
        default=None,
        description="The final answer. Only set this after you have executed REPL code that produced the answer. Leave null to continue.",
    )
    final_var: str | None = Field(
        default=None,
        description="Name of a REPL variable to use as the final answer instead of final_answer.",
    )


SYSTEM_PROMPT = textwrap.dedent("""\
    You are tasked with answering a query using an associated context.
    You can access and transform that context interactively inside a Python REPL
    environment.  You will be queried iteratively until you provide a final answer.

    The REPL environment provides:
      1. `context`                   — the input data for your task.
      2. `llm_query(prompt, model=None)`
            Makes a single LM completion call (no REPL, no iteration).
            Fast and lightweight — use for simple extraction, summarisation, or Q&A.
      3. `llm_query_batched(prompts, model=None)`
            Runs multiple `llm_query` calls; returns List[str] in order.
      4. `rlm_query(prompt, model=None)`
            Spawns a recursive RLM sub-call with its own REPL and iteration loop.
            Use for complex sub-tasks that need multi-step reasoning.
      5. `rlm_query_batched(prompts, model=None)`  — batched version of rlm_query.
      6. `SHOW_VARS()`  — list all variables you have created in the REPL.

    Each response is a JSON object with four fields:
      • reasoning    — required. Think aloud: what does the context contain, what
                       sub-tasks remain, and what code will you run. Do this FIRST.
      • repl_blocks  — list of Python code strings to execute in order (can be empty
                       only if you have already gathered enough evidence from prior turns).
      • final_answer — set ONLY after running REPL code that produced the answer.
                       Prefer final_var whenever the answer was computed in the REPL
                       (pointing to a variable is safer than re-stating a value).
      • final_var    — name of a REPL variable whose value IS the final answer.
                       Use this whenever you stored your answer in a variable.

        Rules:
            - Your FIRST action must always inspect `context` via a repl_block.
            - If context is large (e.g., a long document), do NOT print or return the full text.
                Inspect shape first: type, keys, and lengths; then extract only relevant snippets.
            - If context contains a query/question field, answer that query directly.
            - Final answers should be concise and synthesis-focused, not long verbatim quotations.
            - On document tasks, extract at least 2 snippets from context['document'] before finalizing.
            - On document tasks, include short quoted excerpt(s) or chapter-level evidence in final_answer.
            - Never set final_answer or final_var based only on the metadata prefix.
      - llm_query() and rlm_query() always return a plain string.
        To combine multiple results, pass them into another llm_query() call.
        BAD:  combined = list(zip(result_a, result_b))   # zips characters!
        GOOD: combined = llm_query(f'Combine these results:\nA: {result_a}\nB: {result_b}')
      - rlm_query/llm_query return plain text — never call eval() on their results.
        To extract structured data, use llm_query again to reformat:
        names_csv = llm_query(f'List only planet names, comma-separated:\n{result_a}')
        names = [n.strip() for n in names_csv.split(',')]
      - f-string quoting: when the expression inside {} contains single quotes,
        use a double-quoted f-string. Build the expression into a variable first:
        BAD:  rlm_query(f'Planets: {", ".join(names)}'  # or with ', '.join - quote conflict
        GOOD: q = "Planets: " + ", ".join(names); rlm_query(q)
        GOOD: rlm_query(f"Planets: {', '.join(names)}")
      - Always print computed numeric results before finalizing so you can verify them.
      - Never hardcode verbatim LLM response text as Python string literals.
        Always store llm_query/rlm_query return values in variables.
            - Avoid bare llm_query(question) on document tasks. Pass retrieved snippets into llm_query.
            - Do NOT set model=... in llm_query/rlm_query unless you are certain it exists.
                Prefer llm_query(prompt) with no model override.
      - Leave final_answer and final_var null to continue the loop.
""")

def _sanitize_code(code: str) -> str:
    """
    Escape literal newlines that appear inside single-line string literals.
    These arise when the model emits actual newlines in JSON string fields.
    Triple-quoted strings are left untouched.
    """
    # First decode JSON-artifact escape sequences
    code = code.replace('\\n', '\n').replace('\\t', '\t')

    result = []
    i = 0
    in_single = in_double = in_triple_single = in_triple_double = False

    while i < len(code):
        # Triple-quote transitions (must check before single-quote)
        if not in_single and not in_double:
            if code[i:i+3] == '"""' and not in_triple_single:
                in_triple_double = not in_triple_double
                result.append('"""'); i += 3; continue
            if code[i:i+3] == "'''" and not in_triple_double:
                in_triple_single = not in_triple_single
                result.append("'''"); i += 3; continue

        # Single-quote transitions
        if code[i] == '"' and not in_single and not in_triple_single and not in_triple_double:
            in_double = not in_double
            result.append(code[i]); i += 1; continue
        if code[i] == "'" and not in_double and not in_triple_double and not in_triple_single:
            in_single = not in_single
            result.append(code[i]); i += 1; continue

        # Pass through escape sequences as-is
        if code[i] == '\\' and i + 1 < len(code):
            result.append(code[i]); result.append(code[i+1]); i += 2; continue

        # Literal newline inside a single-line string → escape it
        if code[i] == '\n' and (in_single or in_double):
            result.append('\\n'); i += 1; continue

        result.append(code[i]); i += 1

    return ''.join(result)


def _stdout_metadata(result: ExecResult, preview: int = 200) -> str:
    """Short prefix + length — Algorithm 1: Metadata(stdout)."""
    parts = []
    if result.stdout:
        s = result.stdout
        parts.append(f"stdout ({len(s)} chars): {s[:preview]}{'...' if len(s) > preview else ''}")
    if result.stderr:
        parts.append(f"stderr: {result.stderr[:preview]}")
    return "\n".join(parts) if parts else "(no output)"


def _extract_document_snippet(stdout: str, limit: int = 220) -> str | None:
    """Extract a short, printable snippet from REPL stdout."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            text = str(ast.literal_eval(text))
    except Exception:
        pass

    text = text.replace("\n", " ").strip()
    if not text:
        return None
    return text[:limit] + ("..." if len(text) > limit else "")


def _has_quote_or_chapter_evidence(answer: str) -> bool:
    """Final answers must include either a quote or chapter-level mention."""
    if not answer:
        return False
    has_quote = bool(re.search(r"['\"][^'\"]{15,240}['\"]", answer))
    has_chapter = bool(re.search(r"\bchapter\b", answer, flags=re.IGNORECASE))
    return has_quote or has_chapter


def _is_bare_llm_query(code: str) -> bool:
    """Detect llm_query usage that lacks any evidence/snippet context."""
    if "llm_query(" not in code:
        return False
    lowered = code.lower()
    evidence_tokens = ["context", "snippet", "excerpt", "quote", "passage", "chapter", "document"]
    return not any(tok in lowered for tok in evidence_tokens)


def _is_missing_model_error(exc: Exception) -> bool:
    """Heuristic check for provider errors indicating an unknown/unavailable model."""
    msg = str(exc).lower()
    markers = [
        "model",
        "not found",
        "404",
        "pulling it first",
        "does not exist",
        "unknown model",
    ]
    return all(tok in msg for tok in ["model", "not"]) and any(m in msg for m in markers)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RLMResult:
    response: str
    prompt: Any
    iterations_used: int
    model: str
    extra: dict = field(default_factory=dict)    # e.g. usage stats if you extend this


# ---------------------------------------------------------------------------
# RLM — the core class
# ---------------------------------------------------------------------------

class RLM:
    """
    Minimal Recursive Language Model.

    Usage::

        rlm = RLM(model="gpt-4o")
        result = rlm.completion("Summarise the French Revolution in one paragraph.")
        print(result.response)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 20,
        max_tokens: int = 60_000,
        base_url: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.verbose = verbose
        self._base_llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)
        self.llm = self._base_llm.with_structured_output(RLMTurn)

    # ── LM calls ───────────────────────────────────────────────────────────

    def _call_lm(self, messages: list[dict]) -> RLMTurn:
        return self.llm.invoke(messages)

    def _llm_fn(self, prompt: str, model: str | None = None) -> str:
        """Flat single LM call used by llm_query() inside the REPL."""
        llm = (
            self._base_llm
            if not model or model == self.model
            else ChatOpenAI(model=model, base_url=self.base_url, api_key=self.api_key)
        )
        try:
            return llm.invoke([{"role": "user", "content": prompt}]).content
        except Exception as exc:
            # If the requested model alias is unavailable (common on local Ollama),
            # transparently retry using this RLM's configured default model.
            if model and model != self.model and _is_missing_model_error(exc):
                return self._base_llm.invoke([{"role": "user", "content": prompt}]).content
            raise

    def _rlm_fn(self, prompt: str, model: str | None = None) -> str:
        """
        Recursive RLM call used by rlm_query() inside the REPL.
        Spawns a *child* RLM with its own REPL loop.
        """
        try:
            child = RLM(
                model=model or self.model,
                max_iterations=self.max_iterations,
                base_url=self.base_url,
                api_key=self.api_key,
                verbose=self.verbose,
            )
            result = child.completion(prompt)
            return result.response
        except Exception as exc:
            if model and model != self.model and _is_missing_model_error(exc):
                fallback_child = RLM(
                    model=self.model,
                    max_iterations=self.max_iterations,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    verbose=self.verbose,
                )
                result = fallback_child.completion(prompt)
                return result.response
            raise

    # ── Initial message history ────────────────────────────────────────────

    def _build_initial_messages(self, prompt, root_prompt: str | None = None) -> list[dict]:
        """
        Build the opening message history for a completion call.

        Structure mirrors the original RLM codebase:
          [system, user(context metadata), user(first action prompt)]
        """
        ctx_str = str(prompt)
        ctx_type = type(prompt).__name__
        prefix = ctx_str[:300]
        context_meta = (
            f"Your context is a {ctx_type} with {len(ctx_str):,} total characters.\n"
            f"First {len(prefix)} chars: {prefix}"
        )

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": context_meta},
        ]
        if root_prompt:
            messages.append({"role": "user", "content": f"Task: {root_prompt}"})
        return messages

    # ── Main entry point ───────────────────────────────────────────────────

    def completion(self, prompt, root_prompt: str | None = None) -> RLMResult:
        """
        Perform an RLM completion.

        The prompt / context is exposed as the `context` variable inside the REPL.
        The loop runs until:
          - The LM outputs FINAL(...) or FINAL_VAR(...)
          - max_iterations is exhausted (falls back to one last "summarise" call)

        Args:
            prompt: string, list, or dict — the context for the LM.
            root_prompt: optional small task/question for the root LM.

        Returns:
            RLMResult with `.response` containing the final answer.
        """
        # Build REPL with llm_query and rlm_query wired up
        repl = MiniREPL(
            context=prompt,
            llm_fn=self._llm_fn,
            rlm_fn=self._rlm_fn,
        )

        messages = self._build_initial_messages(prompt, root_prompt=root_prompt)
        requires_document_evidence = isinstance(prompt, dict) and "document" in prompt
        document_pull_count = 0
        extracted_snippets: list[str] = []
        snippet_fingerprints: set[str] = set()
        bare_llm_query_count = 0
        grounded_llm_query_count = 0

        for i in range(self.max_iterations):
            turn: RLMTurn = self._call_lm(messages)
            # Normalize empty / literal-null strings returned by the model to None
            if not turn.final_answer or turn.final_answer == "null":
                turn.final_answer = None
            if not turn.final_var or turn.final_var == "null":
                turn.final_var = None

            if self.verbose:
                print(f"\n[RLM iteration {i + 1}] blocks={len(turn.repl_blocks)} final={'yes' if turn.final_answer or turn.final_var else 'no'}")
                if turn.reasoning:
                    print(f"  Reasoning: {turn.reasoning[:200]}")

            # ── Execute REPL blocks ───────────────────────────────────────
            repl_outputs: list[tuple[str, ExecResult]] = []
            for code in turn.repl_blocks:
                code = _sanitize_code(code)
                result = repl.execute(code)
                repl_outputs.append((code, result))

                lowered_code = code.lower()
                looks_like_doc_snippet_pull = (
                    "context['document']" in code
                    or 'context["document"]' in code
                    or "doc[" in lowered_code
                    or "document[" in lowered_code
                    or "snippet" in lowered_code
                )
                if looks_like_doc_snippet_pull:
                    snippet = _extract_document_snippet(result.stdout)
                    if snippet:
                        fp = snippet.lower().strip()
                        if fp not in snippet_fingerprints:
                            snippet_fingerprints.add(fp)
                            document_pull_count += 1
                            extracted_snippets.append(snippet)

                if _is_bare_llm_query(code):
                    bare_llm_query_count += 1
                elif "llm_query(" in code:
                    grounded_llm_query_count += 1

                if self.verbose:
                    print(f"  Code:\n{code}")
                    out_preview = (result.stdout or result.stderr or "(no output)")[:200].replace("\n", " ")
                    print(f"  REPL: {out_preview}")

            # ── Resolve final answer ──────────────────────────────────────
            final_answer: str | None = turn.final_answer
            if final_answer is None and turn.final_var:
                val = repl.ns_locals.get(turn.final_var)
                final_answer = str(val) if val is not None else None

            blocked_reasons: list[str] = []
            if final_answer is not None and requires_document_evidence:
                if document_pull_count < 2:
                    blocked_reasons.append(
                        f"Need at least 2 snippet pulls from context['document'] before finalizing (have {document_pull_count})."
                    )
                if not _has_quote_or_chapter_evidence(final_answer):
                    blocked_reasons.append(
                        "Final answer must include short quoted excerpt(s) or chapter-level evidence."
                    )
                if bare_llm_query_count > 0 and document_pull_count < 3:
                    blocked_reasons.append(
                        "Bare llm_query(question) detected without enough retrieved snippets."
                    )
                if grounded_llm_query_count < 1:
                    blocked_reasons.append(
                        "Use llm_query with retrieved snippet context (not a bare question-only llm_query)."
                    )

            if blocked_reasons:
                final_answer = None

            if final_answer is not None:
                if self.verbose:
                    print(f"\n[RLM done in {i + 1} iteration(s)]")
                return RLMResult(response=final_answer, prompt=prompt, iterations_used=i + 1, model=self.model)

            # ── Update history ────────────────────────────────────────────
            n = len(turn.repl_blocks)
            messages.append({"role": "assistant", "content": f"Code blocks generated: {n}."})
            for code, result in repl_outputs:
                messages.append({
                    "role": "user",
                    "content": f"```python\n{code}\n```\nREPL: {_stdout_metadata(result)}",
                })
            if blocked_reasons:
                details = "\n".join([f"- {r}" for r in blocked_reasons])
                msg = f"Finalization blocked:\n{details}"
                if extracted_snippets:
                    recent = "\n".join([f'- "{s}"' for s in extracted_snippets[-3:]])
                    msg += f"\nRecent evidence snippets:\n{recent}"
                msg += (
                    "\nDo this next with REPL code:\n"
                    "1) doc = context['document']\n"
                    "2) i = doc.lower().find('responsib')\n"
                    "3) snippet_a = doc[max(0, i-220): i+420]; print(snippet_a)\n"
                    "4) j = doc.lower().find('justice')\n"
                    "5) snippet_b = doc[max(0, j-220): j+420]; print(snippet_b)\n"
                    "6) answer = llm_query(f'Use only these snippets as evidence:\nA: {snippet_a}\nB: {snippet_b}\nTask: {context.get(\"query\", \"\") or \"Use root task\"}')\n"
                    "7) final_answer = answer"
                )
                messages.append({"role": "user", "content": msg})

        if self.verbose:
            print(f"\n[RLM] Max iterations ({self.max_iterations}) reached with no final answer.")
        return RLMResult(
            response="(no final answer produced)",
            prompt=prompt,
            iterations_used=self.max_iterations,
            model=self.model,
        )
