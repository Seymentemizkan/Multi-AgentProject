import time
import pandas as pd
import re
import logging
from datetime import datetime
from IPython.display import display
!pip install -U llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pypdf pandas
import os
import logging
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time
import pandas as pd # Tablo oluşturmak için
from datetime import datetime

# =========================
# 0. LOGLAMA AYARLARI (Sadece bu kısım eklendi)
# =========================
log_filename = "/content/drive/MyDrive/AgentProject/gumruk_asistani_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# 1. AYARLAR
# =========================
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0, temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

PERSIST_DIR = "/content/drive/MyDrive/AgentProject/storage"
pdf_path = "/content/drive/MyDrive/AgentProject/gumruk_kanunu.pdf"

# =========================
# 2. INDEX YÜKLEME VEYA OLUŞTURMA
# =========================
if not os.path.exists(PERSIST_DIR):
    logger.info("🔍 İlk kurulum: PDF okunuyor...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f" PDF dosyası bulunamadı: {pdf_path}")

    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    logger.info(" Yeni index oluşturuldu ve kaydedildi.")
else:
    logger.info("Mevcut index klasörden yükleniyor...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    logger.info("Index başarıyla yüklendi.")

# =========================
# 3. TOOL TANIMI
# =========================
query_engine = index.as_query_engine(similarity_top_k=3)

gumruk_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="gumruk_mevzuat_araci",
        description="Gümrük Kanunu ve vergi oranları hakkında bilgi verir."
    ),
)

# =========================
# 4. AGENT KURULUMU
# =========================
agent = ReActAgent(
    tools=[gumruk_tool],
    llm=Settings.llm,
    verbose=True,
    system_prompt="""
Sen uzman bir Gümrük Danışmanısın. Temel görevin, gümrük mevzuatı ile ilgili soruları 'gumruk_mevzuat_araci' kullanarak yanıtlamaktır.

ÇALIŞMA KURALLARI:
1. Bir soru aldığında, cevabı kendi bilgilerinle vermeden önce MUTLAKA 'gumruk_mevzuat_araci' üzerinden araştırma yap.
2. Öncelikle dokümanlardaki bilgileri (madde numaralarını belirterek) kullan.
3. Eğer aradığın bilgi dokümanlarda açıkça bulunmuyorsa:
   - Önce: "İlgili mevzuat dokümanlarında bu konu hakkında spesifik bir bilgiye ulaşılamamıştır." ifadesini kullan.
   - Sonra: "Ancak genel mesleki bilgilere dayanarak şu açıklamayı yapabilirim:" diyerek genel bir açıklama yap.
   - Bu kısmın resmi mevzuat değil, genel bir bilgilendirme olduğunu özellikle vurgula.

YASAKLAR:
- Asla uydurma madde numarası veya kanun bendi verme.
- Emin olmadığın yasal konularda kesin hüküm bildiren ifadeler (olacaktır, zorunludur vb.) kullanma; bunun yerine "değerlendirilmektedir, olabilir" gibi ihtimal belirten ifadeler kullan.
- Her zaman Türkçe cevap ver.

Cevaplarını profesyonel, anlaşılır ve güvenilir bir tonda hazırla.
"""
)

# =========================
# 5. ETKİLEŞİMLİ SORU-CEVAP DÖNGÜSÜ
# =========================
print("\n--- Sistem Hazır (Çıkış: q) ---")
logger.info("Oturum başlatıldı.")

while True:
    soru = input("\nSorunuz: ")
    if soru.lower() == 'q':
        logger.info("Oturum kapatıldı.")
        break

    # Loglama eklendi
    logger.info(f"SORU: {soru}")

    response = query_engine.query(soru)

    # Loglama eklendi
    logger.info(f"CEVAP: {response}")
    print(f"\n[CEVAP]: {response}")