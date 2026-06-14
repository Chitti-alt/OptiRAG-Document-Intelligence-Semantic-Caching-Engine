import os
import chromadb
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

def ingest_pdf_semantically(file_path):
    print(f"🚀 Running layout-aware vision parsing on {file_path}...")
    print("This may take a moment as it detects tables and formulas...")
    
    # 1. Extract elements using Vision/OCR to understand the layout
    # strategy="hi_res" forces the use of object detection models to find tables
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True, # Critical for keeping data tabular
        extract_images_in_pdf=False
    )
    
    # 2. Semantic Chunking: Group text under their respective section titles
    # This prevents chunking across different chapters
    chunks = chunk_by_title(
        elements,
        max_characters=1000,
        combine_text_under_n_chars=250
    )
    
    # Extract the raw text and metadata from the smart chunks
    processed_texts = []
    processed_metadata = []
    
    for i, chunk in enumerate(chunks):
        processed_texts.append(str(chunk))
        
        # We extract what type of block this is (e.g., Table vs Text) for metadata
        block_type = type(chunk).__name__
        processed_metadata.append({
            "source": file_path, 
            "chunk_id": i,
            "block_type": block_type
        })

    # 3. Load into Vector Database
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(name="engineering_docs")
    
    print(f"Embedding {len(processed_texts)} semantic chunks into ChromaDB...")
    
    collection.add(
        documents=processed_texts,
        metadatas=processed_metadata,
        ids=[f"semantic_chunk_{i}" for i in range(len(processed_texts))]
    )
    
    print("✅ Semantic Ingestion complete. Equations and tables are preserved.")

if __name__ == "__main__":
    # Ensure this points to the PDF you placed in the data folder
    ingest_pdf_semantically("data/turbomachinery_textbook.pdf")
