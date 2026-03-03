# Recursive Document Exploration Agent

This project implements a recursive document analysis agent that can answer questions about documents that are larger than an LLM's context window.

Instead of loading the entire document into the model, the agent explores the document step-by-step using tools, retrieving only the parts needed to answer a question.

The design is inspired by the ideas presented in the paper *Recursive Language Models*, which proposes letting language models interact with large documents as an external environment rather than reading everything at once.

## Why This Exists

Large Language Models have limited context windows.

For example, the novel `frankenstein.txt` contains roughly 103k tokens. Passing the entire text into the model is expensive and often impossible for many models.

Traditional approaches include:

- Chunking documents
- Retrieval Augmented Generation (RAG)
- Summarization pipelines

However, these approaches retrieve context once and then reason.

This project demonstrates a different pattern: **Let the model explore the document interactively.**

The agent:

- Searches for relevant sections
- Analyzes small snippets
- Retrieves larger context if necessary
- Combines evidence from multiple sections

This allows the model to gradually construct an answer without needing the entire document.

## How It Works

The system turns the document into an environment that the model can explore using tools.

High-level flow:

```text
User Question
      │
      ▼
LLM Agent
      │
      ├── probe_context(keyword)
      │       Search document for relevant snippets
      │
      ├── recursive_analyze(snippet, question)
      │       Analyze snippet with smaller LLM call
      │
      ├── get_chunk(start, end)
      │       Retrieve larger section if snippet incomplete
      │
      └── combine_results(results)
              Aggregate findings into final answer
```

Instead of one large prompt, the system performs multiple smaller reasoning steps.

## Architecture

```text
User Question
      │
      ▼
LangChain Agent
      │
      ▼
Exploration Loop
      │
      ├── Search document
      ├── Inspect snippets
      ├── Retrieve larger context
      ├── Analyze findings
      └── Repeat until sufficient evidence
      │
      ▼
Final Answer
```

Each step uses only small portions of the document, keeping token usage low.

## Project Structure

```
.
├── agent.py
├── tools.py
├── frankenstein.txt
├── .env
└── README.md
```

-   `agent.py`: Creates and runs the recursive exploration agent.
-   `tools.py`: Defines tools used by the agent:
    -   `probe_context`
    -   `recursive_analyze`
    -   `get_chunk`
    -   `combine_results`
-   `frankenstein.txt`: Example document used for testing.

## Installation

1.  **Clone the repository.**
    ```bash
    git clone https://github.com/yourusername/recursive-doc-agent
    cd recursive-doc-agent
    ```

2.  **Install dependencies.**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file.**
    ```
    OPENAI_API_KEY=your_api_key
    ```

## Running the Agent

Start the program:

```bash
python agent.py
```

Example interaction:

```
Loading document...
Document loaded.

You: What books does the creature read?

🔧 Tool started → probe_context
Input: books

🔧 Tool started → recursive_analyze
Input: snippet about Paradise Lost

🔧 Tool started → recursive_analyze
Input: snippet about Plutarch's Lives

🔧 Tool started → combine_results

FINAL ANSWER

The creature reads three major books: Paradise Lost, Plutarch's Lives,
and The Sorrows of Werter. These texts influence his understanding
of morality, heroism, and human suffering.
```

## Example Exploration Flow

```text
User Question
      │
      ▼
Extract keywords
      │
      ▼
Search document
      │
      ▼
Analyze snippet
      │
      ▼
Need more context?
     / \
    yes  no
    │     │
Retrieve   Combine results
chunk
    │
Repeat exploration
```

## Observability

The project includes a LangChain callback that prints tool usage.

Example:

```
🔧 Tool started → probe_context
Input: creature

🔧 Tool started → recursive_analyze
Input: snippet text
```

This helps visualize how the agent explores the document step-by-step.

## What This Project Demonstrates

This implementation shows a simplified version of the ideas behind recursive language models:

- LLMs interacting with external documents
- Iterative exploration
- Recursive reasoning over smaller contexts
- Aggregation of partial findings

It provides a practical way to experiment with the concept using tools available in LangChain.

## Limitations

This is a simplified prototype.

The original paper describes more advanced capabilities such as:

- programmatic reasoning
- recursive function calls
- structured execution environments
- more complex planning strategies

This repository focuses only on the document exploration idea.

## Future Improvements

Possible extensions:

- Better search (BM25 / embeddings)
- Tool-calling planning improvements
- Structured memory of explored sections
- Visualization of exploration graph
- Streaming exploration logs