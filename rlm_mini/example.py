"""
example.py — Minimal usage examples for rlm_mini.

Run with:
    cd rlm_mini
    OPENAI_API_KEY=sk-... python example.py
"""

import os
import sys

# Allow running directly from rlm_mini/ or from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
# Load environment variables from .env file before anything else
load_dotenv()


from rlm_mini.rlm import RLM

# ---------------------------------------------------------------------------
# Example 1: pure computation — LM writes code and uses the REPL for math
# ---------------------------------------------------------------------------
def example_computation():
    print("=" * 60)
    print("EXAMPLE 1: Computation in REPL")
    print("=" * 60)

    rlm = RLM(model="gpt-4o-mini", max_iterations=10, verbose=True)
    result = rlm.completion("What is the sum of the first 1000 prime numbers?")
    print(f"\nAnswer: {result.response}")
    print(f"Iterations used: {result.iterations_used}")


# ---------------------------------------------------------------------------
# Example 2: long-text Q&A — LM chunks the context and delegates to llm_query
# ---------------------------------------------------------------------------
def example_long_context():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Long-context Q&A")
    print("=" * 60)

    # Simulate a "long" document
    document = (
        "The French Revolution (1789–1799) was a period of radical political and "
        "societal change in France that overthrew the monarchy, established a republic, "
        "culminated in Napoleon's rise, and had enormous consequences for France and the world.\n\n"
        "Key events included the storming of the Bastille on July 14, 1789 (now celebrated as "
        "Bastille Day), the Declaration of the Rights of Man, the Reign of Terror under "
        "Robespierre (1793–1794), and the abolition of feudalism.\n\n"
        "The Revolution fundamentally changed the political landscape of Europe, spreading "
        "ideals of liberty, equality, and fraternity, and inspiring future revolutions worldwide."
        * 10  # repeat to make it longer
    )

    rlm = RLM(model="gpt-4o-mini", max_iterations=10, verbose=True)
    result = rlm.completion({
        "document": document,
        "query": "What were the main causes and outcomes of the French Revolution?"
    })
    print(f"\nAnswer: {result.response}")


# ---------------------------------------------------------------------------
# Example 3: recursive sub-call — rlm_query spawns a child RLM loop
# ---------------------------------------------------------------------------
def example_recursive():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Recursive sub-call (rlm_query)")
    print("=" * 60)

    rlm = RLM(model="gpt-4o-mini", max_iterations=10, verbose=True)
    result = rlm.completion(
        "I have two sub-tasks. Sub-task A: list the 5 largest planets in the solar system "
        "by diameter. Sub-task B: for each planet, give one interesting fact. "
        "Use rlm_query for each sub-task, then combine the results."
    )
    print(f"\nAnswer: {result.response}")


if __name__ == "__main__":
    example_computation()
    # Uncomment to run the others:
    example_long_context()
    example_recursive()
