from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Versi terbaru pakai 's'
from langchain_huggingface import HuggingFaceEmbeddings            # Standar baru
from langchain_chroma import Chroma                                # Standar baru

# 1. Load PDF
# Pastikan file "UU_PDP_27_2022.pdf" ada di folder yang sama dengan file RAG.py
try:
    loader = PyPDFLoader("UU Nomor 27 Tahun 2022.pdf")
    documents = loader.load()
except Exception as e:
    print(f"❌ Gagal memuat PDF: {e}")
    exit()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# 3. Embedding (Menggunakan standar terbaru langchain-huggingface)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
# Catatan: persist() otomatis dilakukan di versi Chroma terbaru saat inisialisasi
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("RAG siap digunakan. Database disimpan di folder './chroma_db'")
print("RAG siap digunakan")


