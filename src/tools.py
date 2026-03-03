import re
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Worker LLM for recursive analysis
worker_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

DOCUMENT = ""


def set_document(text: str):
    global DOCUMENT
    DOCUMENT = text


@tool
def probe_context(keyword: str, window: int = 200, max_hits: int = 10) -> list:
    """
    Probe the document for occurrences of a keyword.

    This tool scans the document and returns small snippets around
    each match along with their character offsets.

    Each result contains:
    - keyword: the matched keyword
    - start: starting character offset
    - end: ending character offset
    - text: snippet around the match

    Use this tool to locate potentially relevant parts of the document
    before performing deeper analysis.

    If a snippet looks important but truncated, you can use get_chunk(start, end)
    to retrieve a larger section of the document.
    """

    hits = []

    for match in re.finditer(rf"\b{re.escape(keyword)}\b", DOCUMENT, re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(DOCUMENT), match.end() + window)

        snippet = DOCUMENT[start:end]

        hits.append({
            "keyword": keyword,
            "start": start,
            "end": end,
            "text": snippet
        })

        if len(hits) >= max_hits:
            break

    return hits

@tool
def get_chunk(start: int, end: int) -> str:
    """
    Retrieve a specific slice of the document.
    """

    return DOCUMENT[start:end]

@tool
def recursive_analyze(snippet: str, question: str) -> str:
    """
    Ask a smaller worker LLM to analyze a specific snippet.
    """

    prompt = f"""
You are analyzing a small part of a large document.

Question:
{question}

Snippet:
{snippet}

Extract only the information relevant to the question.
"""

    response = worker_llm.invoke(prompt)

    return response.content

@tool
def combine_results(results: list) -> str:
    """
    Combine multiple recursive analysis results into a final answer.
    Accepts either:
    - list[str]
    - list[{"text": "..."}]
    """

    cleaned = []

    for r in results:
        if isinstance(r, dict):
            cleaned.append(r.get("text", ""))
        else:
            cleaned.append(str(r))

    joined = "\n\n".join(cleaned)

    prompt = f"""
Combine the following extracted findings into a clear final answer.

Findings:
{joined}

Final answer:
"""

    response = worker_llm.invoke(prompt)

    return response.content

tools = [
    probe_context,
    recursive_analyze,
    combine_results,
    get_chunk
]