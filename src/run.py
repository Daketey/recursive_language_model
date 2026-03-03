"""
This script defines and runs a recursive document analysis agent
using LangChain. The agent is designed to answer questions about a
document that is too large to fit into the model's context window.
"""
from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import tools, set_document

# Load environment variables from .env file before anything else
load_dotenv()


# --- Callbacks for Observability ---

class ToolCallback(BaseCallbackHandler):
    """A callback handler to print when a tool starts."""
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Prints the name of the tool that is starting."""
        tool_name = serialized.get("name", "unknown")
        print(f"    🔧 Tool started → {tool_name}")
        print(f"      Input: {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        """Prints the output of a tool when it finishes."""
        print(f"    ✅ Tool finished with output:\n{output.content[:200]}\n")


# --- Agent Setup ---

def create_document_agent():
    """Creates and configures the document analysis agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # The system prompt provides the instructions for the agent.
    system_prompt = """
You are a recursive document exploration agent.

The document is very large and cannot be read all at once.
Instead of reasoning over the entire document, you must explore it step-by-step
using tools to locate relevant sections and analyze them recursively.

Your goal is to gather evidence from different parts of the document
and synthesize a final answer.

Available tools:

- probe_context(keyword)
  Search the document for a keyword and return snippets around matches.
  Each result includes the snippet text and character offsets (start, end).
  Use this to locate potentially relevant regions of the document.
  Important: Keyword should be a single word, not a full question.

- recursive_analyze(snippet, question)
  Analyze a specific snippet using a smaller language model to extract
  information relevant to the question.

- get_chunk(start, end)
  Retrieve a larger section of the document using character offsets
  when a snippet appears relevant but incomplete.

- combine_results(results)
  Combine multiple extracted findings into a final coherent answer.


Exploration Strategy:

1. Identify important keywords from the question.
2. Use probe_context to locate relevant areas of the document.
3. Analyze promising snippets using recursive_analyze.
4. If a snippet looks incomplete, retrieve a larger section using get_chunk.
5. Continue probing with new keywords if necessary.
6. Once sufficient evidence is collected, synthesize the results using combine_results.

Important Rules:

- Do not assume any knowledge about the document beyond what you can retrieve with tools.
- If you cannot find relevant information, say you don't know rather than making up an answer.
- Do not attempt to reason over the entire document.
- Always explore the document using tools first.
- Break the task into smaller investigations.
- Information may be spread across multiple parts of the document.
- Stop exploring once you have enough evidence to confidently answer the question.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


# --- Main Application Logic ---

def load_document(file_path: str):
    """Loads the content of the specified document."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def main():
    """
    Main function to set up the document, run the agent,
    and print the final answer.
    """
    print("Loading document...")
    document = load_document("frankenstein.txt")
    set_document(document)
    print("Document loaded.\n")

    app = create_document_agent()
    tool_callback = ToolCallback()
    
    print("Enter your question about the document. Type 'exit' or 'quit' to end.")
    while True:
        question = input("You: ")
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question.strip():
            continue

        print(f"\nAsking question: {question.strip()}\n")

        result = app.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config={"callbacks": [tool_callback]}
        )

        print("\nFINAL ANSWER\n")
        # The actual response content is in the last message of the 'messages' list
        final_answer_message = result["messages"][-1]
        final_answer = final_answer_message.content
        print(final_answer)

        if hasattr(final_answer_message, "usage_metadata") and final_answer_message.usage_metadata:
            usage = final_answer_message.usage_metadata
            print(f"\nTokens consumed: Prompt={usage.get('input_tokens', 'N/A')}, Completion={usage.get('output_tokens', 'N/A')}, Total={usage.get('total_tokens', 'N/A')}\n")


if __name__ == "__main__":
    main()