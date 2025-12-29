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
        raise FileNotFoundError(f"❌ PDF dosyası bulunamadı: {pdf_path}")

    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    logger.info("✅ Yeni index oluşturuldu ve kaydedildi.")
else:
    logger.info("🚀 Mevcut index klasörden yükleniyor...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    logger.info("✅ Index başarıyla yüklendi.")

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
# 5. TEST DÖNGÜSÜ
# =========================
#print("\n--- Sistem Hazır (Çıkış: q) ---")
#logger.info("Oturum başlatıldı.")

"""while True:
    soru = input("\nSorunuz: ")
    if soru.lower() == 'q':
        logger.info("Oturum kapatıldı.")
        break

    # Loglama eklendi
    logger.info(f"SORU: {soru}")

    response = query_engine.query(soru)

    # Loglama eklendi
    logger.info(f"CEVAP: {response}")
    print(f"\n[CEVAP]: {response}")"""

# ==========================================================
# 5 HAKEMLİ HİBRİT BENCHMARK ALGORİTMASI (LOG DESTEKLİ)
# ==========================================================

def llm_hakem_denetimi(soru, cevap, kaynaklar):
    """
    Cevabı 5 kriter üzerinden denetleyen Hakem fonksiyonu.
    """
    hakem_prompt = f"""
    Sen kıdemli bir Gümrük Başmüfettişisin. Verilen cevabı aşağıdaki 5 kriter üzerinden 1 ile 5 arasında puanla.

    SORU: {soru}
    KAYNAK MEVZUAT: {kaynaklar}
    SİSTEMİN CEVABI: {cevap}

    Lütfen SADECE aşağıdaki formatta yanıt ver:
    Sadakat: [Puan]
    Sayisal_Dogruluk: [Puan]
    Atif_Dogrulugu: [Puan]
    Eksiksizlik: [Puan]
    Uslup: [Puan]
    Gerekce: [Kısa açıklama]
    """

    try:
        raw_eval = Settings.llm.complete(hakem_prompt).text
        # Puanları regex ile ayıkla
        scores = re.findall(r"(\d)", raw_eval)
        puanlar = [int(s) for s in scores[:5]] if len(scores) >= 5 else [0,0,0,0,0]
        return puanlar, raw_eval
    except Exception as e:
        logger.error(f"Hakem denetimi sırasında hata: {e}")
        return [0,0,0,0,0], f"Hakem Hatası: {str(e)}"

def run_hybrid_benchmark_with_logs(soru_listesi):
    final_results = []

    # Log dosyasına başlangıç işareti
    logger.info("="*60)
    logger.info(f"📊 BENCHMARK BAŞLADI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Toplam Soru Sayısı: {len(soru_listesi)}")
    logger.info("="*60)

    print(f"🚀 5 Hakemli Hibrit Benchmark Başladı ({len(soru_listesi)} Soru)\n")

    for i, soru in enumerate(soru_listesi):
        print(f"🔄 Test Ediliyor [{i+1}/{len(soru_listesi)}]: {soru[:60]}...")
        logger.info(f"TEST [{i+1}] - SORU: {soru}")

        start_time = time.time()

        try:
            # 1. ADIM: Cevabı Üret (Query Engine üzerinden)
            response = query_engine.query(soru)
            duration = round(time.time() - start_time, 2)

            cevap_metni = str(response)
            kaynak_metinler = "\n".join([n.node.get_content()[:500] for n in response.source_nodes])

            # 2. ADIM: 5 Hakem Denetimi
            puanlar, detayli_rapor = llm_hakem_denetimi(soru, cevap_metni, kaynak_metinler)
            ortalama_skor = sum(puanlar) / 5 if sum(puanlar) > 0 else 0

            # Sonuçları Logla
            logger.info(f"SÜRE: {duration}sn | GENEL SKOR: {ortalama_skor}")
            logger.info(f"HAKEM RAPORU:\n{detayli_rapor}")

            # Verileri listeye ekle
            final_results.append({
                "No": i + 1,
                "Soru": soru,
                "Cevap": cevap_metni,
                "Süre (Sn)": duration,
                "Sadakat": puanlar[0],
                "Sayisal_D.": puanlar[1],
                "Atif_D.": puanlar[2],
                "Eksiksizlik": puanlar[3],
                "Uslup": puanlar[4],
                "GENEL_SKOR": ortalama_skor,
                "Hakem_Gerekce": detayli_rapor.strip().replace("\n", " | ")
            })

        except Exception as e:
            logger.error(f"Soru {i+1} işlenirken kritik hata: {str(e)}")
            final_results.append({"No": i+1, "Soru": soru, "GENEL_SKOR": 0, "Hata": str(e)})

    # DataFrame Oluştur
    df = pd.DataFrame(final_results)

    # Ekrana özet tablo bas
    print("\n" + "="*75)
    print("📊 5 HAKEMLİ HİBRİT DENETİM ÖZETİ")
    print("="*75)
    display(df[["No", "Süre (Sn)", "Sadakat", "Sayisal_D.", "GENEL_SKOR"]])

    # Dosya Kayıtları
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    file_path = f"/content/drive/MyDrive/AgentProject/100sorulukgenel_hibrit_benchmark_sonuclari_{ts}.csv"
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

    logger.info("="*60)
    logger.info(f"✅ BENCHMARK TAMAMLANDI - Rapor: {file_path}")
    logger.info("="*60)

    print(f"\n💾 Detaylı CSV raporu: {file_path}")
    print(f"📄 Hakem gerekçeleri ve detaylar log dosyasında saklandı.")

    return df

# =========================
# ÇALIŞTIRMA
# =========================

benchmark_sorulari = [
# --- GRUP 1: TEMEL TANIMLAR VE GENEL HÜKÜMLER (15 Soru) ---
    "4458 sayılı Gümrük Kanunu'na göre 'Gümrük Beyanı' nedir?",
    "Gümrük Kanunu uyarınca 'Eşyanın Gümrük Statüsü' ne anlama gelir?",
    "Gümrük idareleri tarafından verilen 'Bağlayıcı Tarife Bilgisi' (BTB) nedir?",
    "Gümrük Kanunu kapsamında 'Gümrük Gözetimi' tanımını yapınız.",
    "Gümrük Kanunu'na göre 'Gümrük Kontrolü' neleri kapsar?",
    "Eşyanın gümrükçe onaylanmış bir işlem veya kullanıma tabi tutulması ne demektir?",
    "Gümrük Kanunu kapsamında 'Kişi' tanımı kimleri kapsar?",
    "Gümrük idaresine karşı 'Yükümlü' kimdir?",
    "Gümrük Kanunu'na göre 'Serbest Dolaşımda Bulunan Eşya' ne demektir?",
    "Gümrük bölgesine giren araçların gümrük idaresine bildirilmesi zorunlu mudur?",
    "Gümrük idarelerinin çalışma saatleri dışında işlem yapılması mümkün müdür?",
    "Gümrük Kanunu'na göre 'Türkiye Gümrük Bölgesi' sınırları neresidir?",
    "Gümrük beyannamesinde 'Temsil Hakkı' nasıl kullanılır?",
    "Dolaylı temsilci ile doğrudan temsilci arasındaki fark nedir?",
    "Gümrük müşavirlerinin hukuki sorumluluğu hangi maddede düzenlenmiştir?",

    # --- GRUP 2: SAYISAL LİMİTLER, MUAFİYETLER VE SÜRELER (20 Soru) ---
    "Yolcu beraberinde getirilen hediyelik eşyada 430 Euro sınırı kimler için geçerlidir?",
    "15 yaşından küçük yolcular için hediyelik eşya muafiyet sınırı kaç Euro'dur?",
    "Posta yoluyla gelen kitap ve benzeri basılı yayınlarda muafiyet sınırı var mıdır?",
    "Gümrük vergilerinin ödenmesi için tanınan 10 günlük süre ne zaman başlar?",
    "Gümrük Kanunu'na göre itiraz süresi kaç gündür?",
    "Gümrük idaresi itirazları kaç gün içinde karara bağlamalıdır?",
    "Geçici depolanan eşya deniz yoluyla gelmişse kaç gün içinde rejim beyan edilmelidir?",
    "Karayoluyla gelen eşya için geçici depolama süresi kaç gündür?",
    "Gümrük Kanunu Madde 241 uyarınca usulsüzlük cezası miktarı nasıl belirlenir?",
    "Yolcu beraberinde getirilen nakit paranın beyan edilmemesinin cezası nedir?",
    "Muafiyet kapsamında getirilen bir aracın kaç yıl süreyle satılması yasaktır?",
    "Gümrük vergilerinde zamanaşımı süresi kaç yıldır?",
    "Gümrük idaresine verilen teminatlar hangi durumlarda iade edilir?",
    "İthalat vergilerinden tam muafiyet sağlanan durumlar nelerdir?",
    "Posta kargolarında 150 Euro üzerindeki kişisel eşyalar nasıl vergilendirilir?",
    "Hangi miktar üzerindeki numuneler ticari nitelikte sayılır?",
    "Gümrük vergisi geri verme veya kaldırma başvurusu kaç yıl içinde yapılmalıdır?",
    "Gümrük beyannamesinde düzeltme yapılması için süre sınırı var mıdır?",
    "Tasfiyelik hale gelen eşya için bekleme süresi kaç gündür?",
    "Bağlayıcı Menşe Bilgisi kaç yıl süreyle geçerlidir?",

    # --- GRUP 3: GÜMRÜK REJİMLERİ (25 Soru) ---
    "Serbest Dolaşıma Giriş Rejimi nedir?",
    "Transit Rejimi kapsamında eşya taşınırken teminat zorunlu mudur?",
    "Antrepo Rejimi nedir ve kaç tip antrepo vardır?",
    "Dahilde İşleme Rejimi (DİR) şartlı muafiyet sistemi nasıl çalışır?",
    "DİR kapsamında 'Eşdeğer Eşya' kullanımı nedir?",
    "Hariçte İşleme Rejimi nedir?",
    "Gümrük Kontrolü Altında İşleme Rejimi hangi durumlarda tercih edilir?",
    "Geçici İthalat Rejimi nedir?",
    "İhracat Rejimi nedir?",
    "Dahilde İşleme Rejimi'nde 'Geri Ödeme Sistemi' nedir?",
    "Antrepoda bulunan eşyanın başkasına devri mümkün müdür?",
    "A Tipi antrepo ile B Tipi antrepo arasındaki fark nedir?",
    "Transit Rejimi'nde 'Varış İdaresi' sorumlulukları nelerdir?",
    "Dahilde İşleme Rejimi'nde işlem görmüş ürünün vergilendirilmesi nasıl yapılır?",
    "Ekonomik etkili gümrük rejimleri nelerdir?",
    "Geçici ithal edilen eşyanın Türkiye'de kalma süresi uzatılabilir mi?",
    "İhracat sayılan satış ve teslimler nelerdir?",
    "Dahilde İşleme İzin Belgesi (DİİB) süreleri ne kadardır?",
    "Antrepoda yapılan 'Elleçleme' faaliyetleri nelerdir?",
    "Transit rejiminde 'Asıl Sorumlu' kimdir?",
    "Gümrük kontrolü altında işleme rejiminde 'İşlem Görmüş Ürün' tanımı nedir?",
    "İhracat beyannamesi verildikten sonra eşya ne kadar sürede yurt dışı edilmelidir?",
    "Serbest bölgeler gümrük bölgesi dışında mı sayılır?",
    "Serbest bölgelere giren yabancı menşeli eşyanın gümrük statüsü nedir?",
    "Dahilde işleme rejiminde firesi çıkan eşyanın akıbeti ne olur?",

    # --- GRUP 4: EŞYANIN KIYMETİ VE MENŞEİ (15 Soru) ---
    "Eşyanın gümrük kıymeti belirlenirken ilk kullanılan yöntem nedir?",
    "Satış bedeli yöntemi hangi durumlarda reddedilir?",
    "Gümrük kıymetine dahil edilmesi gereken nakliye giderleri nereye kadar olan kısmı kapsar?",
    "Sigorta giderleri gümrük kıymetine nasıl dahil edilir?",
    "Rücu hakları ve royalti ödemeleri gümrük kıymetini etkiler mi?",
    "İndirgenmiş değer yöntemi nedir?",
    "Hesaplanmış değer yöntemi nasıl uygulanır?",
    "Eşyanın menşei (Origin) nasıl belirlenir?",
    "Tamamen bir ülkede elde edilen eşya ne demektir?",
    "Yeterli işçilik ve işleme kriteri nedir?",
    "Tercihli menşe ile tercihli olmayan menşe arasındaki fark nedir?",
    "A.TR dolaşım belgesi hangi ticaret kapsamında kullanılır?",
    "EUR.1 ve EUR-MED belgeleri arasındaki fark nedir?",
    "Menşe şahadetnamesi hangi durumlarda zorunludur?",
    "Kümülasyon (Menşe birikimi) nedir?",

    # --- GRUP 5: İHTİLAFLAR, CEZALAR VE TASFİYE (15 Soru) ---
    "Gümrük Kanunu Madde 234 uyarınca vergi farkı cezaları nasıl hesaplanır?",
    "Eşyanın mülkiyetinin kamuya geçirilmesi kararı hangi durumlarda verilir?",
    "Gümrük Kanunu Madde 235 uyarınca yasaklı eşya ithalinin cezası nedir?",
    "Gümrük uzlaşma komisyonu kararlarına karşı dava açılabilir mi?",
    "Gümrük cezalarında indirim sağlanan durumlar nelerdir?",
    "Pişmanlık ve ıslah gümrük cezalarında uygulanır mı?",
    "Gümrük idaresine eksik miktar beyan edilmesinin cezası nedir?",
    "Gümrük Kanunu Madde 236 uyarınca antrepo rejimi ihlallerinin cezası nedir?",
    "Kaçakçılıkla Mücadele Kanunu ile Gümrük Kanunu arasındaki ilişki nedir?",
    "Gümrük vergilerinin tecil ve taksitlendirilmesi mümkün müdür?",
    "Tasfiyelik eşyanın satış yöntemleri nelerdir?",
    "Gümrük vergilerine itiraz edilmesi ödemeyi durdurur mu?",
    "İdari yargı sürecinde gümrük davaları nerede açılır?",
    "Gümrük müşavirinin cezai sorumluluğu hangi durumlarda şahsidir?",
    "Eşyanın gümrük idaresince alıkonulması durumunda ardiye ücreti kim tarafından ödenir?",

    # --- GRUP 6: ZOR VE UÇ SENARYOLAR (10 Soru) ---
    "Türkiye'den ihraç edilen bir ürünün yabancı ülkede gümrükten çekilemeyip geri gelmesi (Madde 168) prosedürü nedir?",
    "Enkaz ve atık haline gelmiş eşyanın gümrük statüsü nasıl değişir?",
    "Gümrük Kanunu'nda mücbir sebep olarak kabul edilen haller nelerdir?",
    "Açık denizde parçalanan bir geminin parçalarının kıyıya vurması durumunda gümrük işlemi nasıl yapılır?",
    "Fikri ve sınai mülkiyet hakları kapsamında gümrükte durdurulan eşya süreci nasıldır?",
    "Geçici ithal edilen bir eşyanın Türkiye'de çalınması durumunda gümrük yükümlülüğü doğar mı?",
    "İthalat sırasında faturada gösterilmeyen bir iskonto sonradan gümrük kıymetinden düşülebilir mi?",
    "Bağlayıcı tarife bilgisinin hata nedeniyle iptali durumunda, bu bilgiye güvenerek işlem yapan yükümlünün durumu ne olur?",
    "Gümrük Kanunu Madde 174 uyarınca eşyanın imha edilmesi vergi borcunu siler mi?",
    "Kripto madencilik cihazlarının ithalatında ÖTV ve KDV matrahı nasıl oluşturulur?"
]

# Benchmark'ı başlatmak için alttaki satırı aktif edin
df_sonuc = run_hybrid_benchmark_with_logs(benchmark_sorulari)