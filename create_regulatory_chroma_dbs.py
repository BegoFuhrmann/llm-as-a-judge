import logging
from pathlib import Path
from typing import List

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


def load_single_pdf(file_path: Path, document_name: str) -> List[Document]:
    """Load and process a single PDF document."""
    print(f"Loading {document_name} from: {file_path}")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return []
    
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        # Add metadata to identify the document
        for doc in docs:
            doc.metadata["document_name"] = document_name
            doc.metadata["source_file"] = str(file_path)
            doc.metadata["file_type"] = ".pdf"
            
        print(f"✅ Successfully loaded {len(docs)} pages from {document_name}")
        return docs
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []


def create_chroma_db(documents: List[Document], collection_name: str, db_name: str, output_dir: Path):
    """Create a ChromaDB from the provided documents."""
    if not documents:
        print(f"❌ No documents provided for {db_name}")
        return
    
    # Create output directory
    chroma_dir = output_dir / db_name
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Split documents into chunks
    print(f"Splitting {len(documents)} documents into chunks for {db_name}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create Chroma vector store
    print(f"Creating ChromaDB with {len(chunks)} chunks for {db_name}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(chroma_dir)
    )
    
    print(f"✅ {db_name} ChromaDB created successfully!")
    print(f"   Location: {chroma_dir}")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Collection name: {collection_name}")
    print()


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Output directory for ChromaDBs
    output_dir = Path(r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system\data")
    
    # Document paths
    eu_ai_act_path = Path(r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Documents\Regulatory\EU_AI_ACT (2024-1689).pdf")
    gdpr_path = Path(r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Documents\Regulatory\GDPR (2016-679).pdf")
    
    print("=== Creating Regulatory ChromaDBs ===")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load EU AI Act
    print("1. Processing EU AI Act...")
    eu_ai_act_docs = load_single_pdf(eu_ai_act_path, "EU AI Act (2024-1689)")
    
    # Load GDPR
    print("\n2. Processing GDPR...")
    gdpr_docs = load_single_pdf(gdpr_path, "GDPR (2016-679)")
    
    # Create ChromaDBs
    print("\n=== Creating ChromaDBs ===")
    
    # Create EU AI Act ChromaDB
    create_chroma_db(
        documents=eu_ai_act_docs,
        collection_name="eu_ai_act_2024",
        db_name="eu_ai_act_chroma_db",
        output_dir=output_dir
    )
    
    # Create GDPR ChromaDB
    create_chroma_db(
        documents=gdpr_docs,
        collection_name="gdpr_2016",
        db_name="gdpr_chroma_db",
        output_dir=output_dir
    )
    
    print("=== Summary ===")
    print("✅ All regulatory ChromaDBs created successfully!")
    print(f"📁 EU AI Act DB: {output_dir / 'eu_ai_act_chroma_db'}")
    print(f"📁 GDPR DB: {output_dir / 'gdpr_chroma_db'}")


if __name__ == "__main__":
    main()
