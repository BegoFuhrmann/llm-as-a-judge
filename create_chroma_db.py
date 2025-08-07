import logging
from pathlib import Path
from typing import List

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)

# Document processing
import pandas as pd


def load_documents(data_dir: Path) -> List[Document]:
    """Load and process documents from the given directory and all subdirectories."""
    documents = []
    
    # Supported file extensions
    exts = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"]
    
    # Get all files recursively from all subdirectories
    file_paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    
    print(f"Found {len(file_paths)} documents to process across all subdirectories")
    print(f"Scanning directory: {data_dir}")
    
    # Group files by subdirectory for better logging
    subdirs = {}
    for file_path in file_paths:
        subdir = file_path.parent.name
        if subdir not in subdirs:
            subdirs[subdir] = []
        subdirs[subdir].append(file_path)
    
    print(f"Found documents in {len(subdirs)} subdirectories:")
    for subdir, files in subdirs.items():
        print(f"  - {subdir}: {len(files)} files")
    
    processed_count = 0
    for file_path in file_paths:
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            elif suffix in [".xlsx", ".xls"]:
                # Process Excel files
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)
                docs = [Document(page_content=content, metadata={"source": str(file_path)})]
            else:
                continue
            
            # Add subdirectory info to metadata
            for doc in docs:
                doc.metadata["subdirectory"] = file_path.parent.name
                doc.metadata["file_type"] = suffix
                
            documents.extend(docs)
            processed_count += 1
            print(f"Processed ({processed_count}/{len(file_paths)}): {file_path.parent.name}/{file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return documents


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Source document folder - now pointing to the entire _new directory
    data_dir = Path(r"W:\T1368-WAM AI\03 Projects\02 AI Projects\01 Berenberg AI Assistant (BAIA)\15 newHQ data\_new")
    if not data_dir.exists():
        print(f"Source directory not found: {data_dir}")
        return
    
    # Output directory for Chroma DB
    chroma_dir = Path(__file__).parent / "agentic_system" / "chromaDB"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(data_dir)
    
    if not documents:
        print("No documents found to process.")
        return
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create Chroma vector store
    print(f"Creating Chroma DB with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="berenberg_newhq_docs",
        persist_directory=str(chroma_dir)
    )
    
    # Note: Chroma 0.4.x automatically persists, no need for manual persist()
    
    print(f"✅ Chroma DB created successfully!")
    print(f"   Location: {chroma_dir}")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Collection name: berenberg_newhq_docs")


if __name__ == "__main__":
    main()
