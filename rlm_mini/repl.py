"""Sandboxed persistent Python REPL used by rlm.py.

This module keeps the same tiny public surface as before:
- MiniREPL(context, llm_fn, rlm_fn=None)
- MiniREPL.execute(code) -> ExecResult(stdout, stderr, locals_snapshot)

Internally it borrows robust behavior from the original LocalREPL design:
- temporary working directory for execution
- scaffold restoration (context/tools cannot be permanently overwritten)
- thread-safe stdout/stderr capture
"""

import ast
import builtins
import io
import os
import shutil
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

_SAFE_NAMES = {
    "print", "len", "str", "int", "float", "list", "dict", "set", "tuple", "bool",
    "type", "isinstance", "issubclass", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "range", "min", "max", "sum", "abs", "round", "any", "all",
    "pow", "divmod", "chr", "ord", "hex", "bin", "oct", "repr", "ascii", "format",
    "hash", "id", "iter", "next", "slice", "callable", "hasattr", "getattr",
    "setattr", "delattr", "dir", "vars", "bytes", "bytearray", "memoryview",
    "complex", "object", "super", "property", "staticmethod", "classmethod",
    "__import__", "open",
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "FileNotFoundError", "OSError", "IOError",
    "RuntimeError", "NameError", "ImportError", "StopIteration", "AssertionError",
    "NotImplementedError", "ArithmeticError", "LookupError", "Warning",
}
_SAFE_BUILTINS: dict = {
    name: getattr(builtins, name) for name in _SAFE_NAMES if hasattr(builtins, name)
}
_SAFE_BUILTINS.update(
    {
        "eval": None,
        "exec": None,
        "compile": None,
        "input": None,
        "globals": None,
        "locals": None,
    }
)


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    locals_snapshot: dict = field(default_factory=dict)


class MiniREPL:
    """Persistent sandboxed Python REPL."""

    _RESERVED_GLOBAL_TOOLS = (
        "SHOW_VARS",
        "llm_query",
        "llm_query_batched",
        "rlm_query",
        "rlm_query_batched",
        "FINAL_VAR",
    )

    def __init__(self, context, llm_fn, rlm_fn=None):
        self._llm_fn = llm_fn
        self._rlm_fn = rlm_fn or llm_fn
        self._lock = threading.Lock()
        self._last_final_answer: str | None = None

        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"mini_repl_{uuid.uuid4()}_")

        self.ns_globals: dict = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
            "SHOW_VARS": self._show_vars,
            "llm_query": self._llm_query,
            "llm_query_batched": self._llm_query_batched,
            "rlm_query": self._rlm_query,
            "rlm_query_batched": self._rlm_query_batched,
            "FINAL_VAR": self._final_var,
        }
        self.ns_locals: dict = {"context": context}
        self._context_ref = context

    def _show_vars(self) -> str:
        available = {
            k: type(v).__name__ for k, v in self.ns_locals.items() if not k.startswith("_")
        }
        return f"Variables: {available}" if available else "No variables created yet."

    def _final_var(self, variable_name):
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer

        key = variable_name.strip().strip("\"'")
        if key in self.ns_locals:
            answer = str(self.ns_locals[key])
            self._last_final_answer = answer
            return answer

        available = [k for k in self.ns_locals if not k.startswith("_")]
        return f"Error: Variable '{key}' not found. Available variables: {available}."

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        return self._llm_fn(prompt, model)

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return [self._llm_fn(p, model) for p in prompts]

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        return self._rlm_fn(prompt, model)

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return [self._rlm_fn(p, model) for p in prompts]

    @contextmanager
    def _capture_output(self):
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _restore_scaffold(self):
        self.ns_globals["SHOW_VARS"] = self._show_vars
        self.ns_globals["llm_query"] = self._llm_query
        self.ns_globals["llm_query_batched"] = self._llm_query_batched
        self.ns_globals["rlm_query"] = self._rlm_query
        self.ns_globals["rlm_query_batched"] = self._rlm_query_batched
        self.ns_globals["FINAL_VAR"] = self._final_var
        self.ns_locals["context"] = self._context_ref

    def execute(self, code: str) -> ExecResult:
        """Execute code in persistent namespace; print repr of final expression."""
        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                tree = ast.parse(code)
                combined = {**self.ns_globals, **self.ns_locals}

                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    *rest_stmts, last_stmt = tree.body
                    if rest_stmts:
                        rest = ast.Module(body=rest_stmts, type_ignores=[])
                        exec(
                            compile(ast.fix_missing_locations(rest), "<repl>", "exec"),
                            combined,
                            combined,
                        )
                    last_expr = ast.Expression(body=last_stmt.value)
                    val = eval(
                        compile(ast.fix_missing_locations(last_expr), "<repl>", "eval"),
                        combined,
                        combined,
                    )
                    if val is not None:
                        print(repr(val))
                else:
                    exec(
                        compile(ast.fix_missing_locations(tree), "<repl>", "exec"),
                        combined,
                        combined,
                    )

                for key, value in combined.items():
                    if key not in self.ns_globals and not key.startswith("_"):
                        self.ns_locals[key] = value
            except Exception as exc:
                stderr_buf.write(f"{type(exc).__name__}: {exc}\n")
            finally:
                self._restore_scaffold()

        return ExecResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            locals_snapshot={
                k: v for k, v in self.ns_locals.items() if not k.startswith("_")
            },
        )

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.ns_globals.clear()
        self.ns_locals.clear()

    def __del__(self):
        self.cleanup()
