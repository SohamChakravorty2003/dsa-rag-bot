from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyMuPDFLoader("DataStructures.pdf")
documents = loader.load()
content_docs = documents[20:637]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(content_docs)

print(f"Embedding {len(chunks)} chunks locally... this may take a few minutes on first run.")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="./chroma_db",
    collection_name="dsa_book",
)

print("Done! Vector store saved to ./chroma_db")
print(f"Total vectors stored: {vectorstore._collection.count()}")