from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

print("Memuat sistem UU PDP...")

# Load embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load LLM gratis dari Ollama
llm = Ollama(model="phi3")

print("\n📘 Chat UU PDP (Ollama) siap digunakan!")
print("Ketik 'exit' untuk keluar.\n")

while True:
    question = input("Pertanyaan: ")

    if question.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Anda adalah asisten hukum yang hanya menjawab berdasarkan UU Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""

    answer = llm.invoke(prompt)

    print("\n📝 Jawaban:")
    print(answer)

    print("\n📌 Sumber:")
    for i, doc in enumerate(docs, 1):
        print(f"Sumber {i}: Halaman {doc.metadata.get('page', 'Tidak diketahui')}")

    print("\n" + "="*60 + "\n")

