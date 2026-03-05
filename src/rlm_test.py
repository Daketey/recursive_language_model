import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# Load environment variables from .env before model initialization.
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rlm import RLM as OriginalRLM
except Exception:
    OriginalRLM = None

from rlm_mini.rlm import RLM as MiniRLM

LM_PROVIDER = os.getenv("LM_PROVIDER", "openai").strip().lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fast local testing knobs (tuned for Ollama qwen3:8b)
DOC_CHAR_LIMIT = int(os.getenv("DOC_CHAR_LIMIT", "120000"))
ORIG_MAX_ITER = int(os.getenv("ORIG_MAX_ITER", "30"))
MINI_MAX_ITER = int(os.getenv("MINI_MAX_ITER", "30"))
ORIG_MAX_TIMEOUT_SEC = float(os.getenv("ORIG_MAX_TIMEOUT_SEC", "120"))


def _provider_name() -> str:
    # Accept "openapi" as a user-friendly alias for "openai".
    if LM_PROVIDER in {"openai", "openapi"}:
        return "openai"
    return "ollama"


def _model_config() -> dict[str, str | None]:
    provider = _provider_name()
    if provider == "openai":
        return {
            "provider": "openai",
            "model": OPENAI_MODEL,
            "base_url": OPENAI_BASE_URL,
            "api_key": OPENAI_API_KEY,
        }
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_API_KEY,
    }


def load_frankenstein() -> str:
    txt_path = ROOT / "frankenstein.txt"
    text = txt_path.read_text(encoding="utf-8")
    if DOC_CHAR_LIMIT > 0:
        return text[:DOC_CHAR_LIMIT]
    return text


def run_original_rlm(document: str, query: str) -> None:
    if OriginalRLM is None:
        print("\n[Original RLM] Skipped: `from rlm import RLM` is not available.")
        return

    print("\n" + "=" * 60)
    print("ORIGINAL RLM")
    print("=" * 60)

    cfg = _model_config()

    rlm = OriginalRLM(
        backend="openai",
        backend_kwargs={
            "model_name": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key": cfg["api_key"],
        },
        max_iterations=ORIG_MAX_ITER,
        max_timeout=ORIG_MAX_TIMEOUT_SEC,
        verbose=True,
    )

    # Original RLM API supports: completion(prompt=..., root_prompt=...)
    # prompt carries the full context, root_prompt carries the user question.
    result = rlm.completion(
        prompt={"document": document},
        root_prompt=query,
    )
    print("\n[Original RLM Answer]")
    print(result.response)


def run_mini_rlm(document: str, query: str) -> None:
    print("\n" + "=" * 60)
    print("RLM MINI")
    print("=" * 60)

    cfg = _model_config()

    rlm = MiniRLM(
        model=cfg["model"],
        max_iterations=MINI_MAX_ITER,
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        verbose=True,
    )
    result = rlm.completion(
        prompt={"document": document},
        root_prompt=query,
    )

    print("\n[Mini RLM Answer]")
    print(result.response)


if __name__ == "__main__":
    cfg = _model_config()
    document = load_frankenstein()
    print(f"Provider: {cfg['provider']}")
    print(f"Model: {cfg['model']}")
    print(f"Document chars used: {len(document):,}")
    query = (
        "What are the main themes in Frankenstein, and how do Victor Frankenstein "
        "and the creature differ in their views of responsibility and justice?"
    )

    run_original_rlm(document, query)
    # run_mini_rlm(document, query)
