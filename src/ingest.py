import os
import sys
from langchain_community.document_loaders import PyPDFLoader

def ingest_pdf(file_path: str, output_dir: str = "data"):
    """
    Ingests a PDF file, extracts the text, and saves it to a text file.

    Args:
        file_path (str): The path to the PDF file.
        output_dir (str, optional): The directory to save the output text file. Defaults to "data".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()

    text_content = ""
    for doc in documents:
        text_content += doc.page_content + "\n"

    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Successfully ingested {file_path} to {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            ingest_pdf(pdf_path)
        else:
            print(f"Error: File not found at '{pdf_path}'")
    else:
        print("Usage: python ingestion.py <path_to_pdf_file>")
