from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Create / Load Chroma Vector DB ---
chroma = PersistentClient(path="./rag_db")
collection = chroma.get_or_create_collection(name="frankenstein")


# --- Ingest Frankenstein from TXT ---
def ingest_document(file_path="frankenstein.txt"):
    """
    Load a text document, split it into chunks, and embed them for retrieval.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split text into chunks (e.g., by paragraphs)
    chunks = text.split("\n\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Batch processing for efficiency
    batch_size = 50
    total = len(chunks)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = chunks[batch_start:batch_end]
        
        ids = [f"chunk_{i}" for i in range(batch_start, batch_end)]
        metadatas = [{"source": file_path, "chunk_number": i} for i in range(batch_start, batch_end)]
        
        # Encode batch
        embeddings_list = embedder.encode(batch_chunks)
        
        # Add batch to Chroma
        collection.add(
            ids=ids,
            documents=batch_chunks,
            embeddings=embeddings_list.tolist(),
            metadatas=metadatas
        )
        
        progress = min(batch_end, total)
        print(f"  ✓ Ingested {progress}/{total} chunks...")


# --- Retrieve Chunks by Query ---
def retrieve_chunks(query, k=5):
    """
    Retrieve text chunks matching the query.
    """
    try:
        q_emb = embedder.encode([query])
        
        results = collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }


# --- Chat History (global) ---
chat_history = []   # VERY IMPORTANT


# --- Ask with RAG + Chat History ---
def ask_rag(query):
    """
    Query for relevant text chunks and get AI response.
    """
    results = retrieve_chunks(query, k=5)
    
    # Format retrieved chunks with metadata
    context_info = ""
    if results["ids"] and results["ids"][0]:
        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
        ):
            context_info += f"\n--- Context Chunk {i} ---\n"
            context_info += f"Source: {metadata.get('source', 'N/A')}\n"
            context_info += f"Chunk Number: {metadata.get('chunk_number', 'N/A')}\n"
            context_info += f"Relevance Score: {1 - distance:.2f}\n"
            context_info += f"Content:\n{doc}\n"
    else:
        context_info = "No relevant context found for your query."

    print("=" * 70)
    if results["ids"] and results["ids"][0]:
        print(f"Found {len(results['ids'][0])} relevant chunks:")
    else:
        print("No chunks retrieved")
    print(context_info)
    print("=" * 70)

    # ---- Build messages list dynamically ----
    messages = []

    # 1. SYSTEM MESSAGE
    messages.append({
        "role": "system",
        "content": (
            "You are an AI assistant that answers questions using only the provided context. "
            "Do not assume any external knowledge. Your knowledge is limited to the retrieved chunks from the document. "
            "Do not make up answers or use information not included in the context. If the context does not contain the answer, specify the parts you dont know"
            "Base your answers strictly on the retrieved context."
        )
    })

    # 2. CHAT HISTORY MESSAGES (assistant + user)
    for role, msg in chat_history:
        messages.append({"role": role, "content": msg})

    # 3. CURRENT TURN (user with context + question)
    messages.append({
        "role": "user",
        "content": (
            f"User Query:\n{query}\n\n"
            f"Retrieved Context from Frankenstein:\n{context_info}"
        )
    })

    # ---- Call Model ----
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content
    usage = response.usage

    # ---- Store current turn in history ----
    chat_history.append(("user", query))
    chat_history.append(("assistant", answer))

    return answer, usage


# -----------------------------------------------
# INGEST DOCUMENT ON FIRST RUN
# -----------------------------------------------
try:
    # Check if collection has any documents
    collection_count = collection.count()
    if collection_count == 0:
        print("Ingesting Frankenstein from frankenstein.txt...")
        ingest_document("frankenstein.txt")
        print(f"Successfully ingested document into database.")
    else:
        print(f"Using existing database with {collection_count} chunks.")
except Exception as e:
    print(f"Error checking database: {e}")
    print("Attempting to ingest document...")
    ingest_document("frankenstein.txt")

print("\n" + "="*60)
print("Frankenstein RAG Chat Ready!")
print("Query examples: 'who is victor?', 'what is the monster's story?', 'the arctic'")
print("Type 'exit' to quit.")
print("="*60 + "\n")

while True:
    q = input("You: ")
    if q.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    
    answer, usage = ask_rag(q)
    print("\nAssistant:", answer, "\n")
    if usage:
        print(f"Tokens consumed: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}\n")
