# =========================
# 0. KURULUM
# =========================
#!pip install -U llama-index llama-index-llms-openai llama-index-embeddings-huggingface pypdf

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# =========================
# 1. OPENAI AYARLARI
# =========================
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in .env file")

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=False
)



# =========================
# 2. RAG (PDF -> VECTOR)
# =========================
pdf_path = "/content/drive/MyDrive/AgentProject/gumruk_kanunu.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError("❌ gumruk_kanunu.pdf bulunamadı! Drive'a yükle.")

documents = SimpleDirectoryReader(
    input_files=[pdf_path]
).load_data()

index = VectorStoreIndex.from_documents(documents)

print("✅ RAG sistemi başarıyla kuruldu.")


# =========================
# 3. TOOL TANIMI
# =========================
query_engine = index.as_query_engine(similarity_top_k=5)

gumruk_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="gumruk_mevzuat_araci",
        description=(
            "Gümrük Kanunu, muafiyet sınırları, "
            "vergi oranları ve ithalat prosedürleri hakkında bilgi verir."
        )
    ),
)


# =========================
# 4. SYSTEM PROMPT + AGENT
# =========================
system_prompt = """
Sen uzman bir Gümrük Danışmanısın.

Öncelikle verilen dokümanları (gümrük mevzuatı) kullanarak cevap üret.
Eğer sorunun cevabı dokümanlarda açık ve yeterli şekilde bulunmuyorsa:

- Bunu açıkça belirt
- Ardından genel mesleki bilgilere dayalı bir açıklama yap
- Bu kısmın mevzuat değil, genel bilgi olduğunu özellikle söyle

Asla uydurma madde numarası verme.
Emin olmadığın konularda kesin ifadeler kullanma.
Her zaman Türkçe cevap ver.
"""

agent = ReActAgent(
    tools=[gumruk_tool],
    llm=Settings.llm,
    system_prompt=system_prompt,
    verbose=True,
    max_iterations=3
)


# =========================
# 5. TEST
# =========================
#print(query_engine.query("300 dolarlık ürün için vergi var mı?"))
while True:

  soru = input("Soru sorun !")
  if soru=="q":
    break
  else:
    print(f"\n[KULLANICI]: {soru}")
    response = query_engine.query(soru)

    print("\n[AJANIN NİHAİ CEVABI]")
    print(response)
