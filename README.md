# Gümrük Asistanı (AI Agent)

Bu proje, LlamaIndex ve Ollama kullanarak geliştirilmiş, Gümrük Kanunu ve mevzuatı hakkında soruları yanıtlayan yapay zeka tabanlı bir asistandır. RAG (Retrieval-Augmented Generation) mimarisini kullanarak, kendisine sağlanan PDF dokümanları üzerinden doğru ve kaynaklı bilgiler sunmayı amaçlar.

## Proje Hakkında

Gümrük Asistanı, kullanıcıların gümrük mevzuatı ile ilgili sorularını yanıtlamak için tasarlanmıştır. Sistem, `llama3.1` modelini ve `sentence-transformers` embedding modelini kullanarak dokümanlar üzerinde semantik arama yapar ve ilgili kısımları referans göstererek cevaplar üretir.

### Temel Özellikler
- **RAG Mimarisi:** Soruları yanıtlarken öncelikle yerel bilgi tabanını (PDF dokümanları) kullanır.
- **ReAct Agent:** Soruyu analiz eder, gerekirse araçları (tools) kullanarak araştırma yapar ve adım adım düşünerek cevap verir.
- **Kaynak Önceliği:** Cevaplarda öncelikle resmi dokümanlardaki (Gümrük Kanunu) maddeleri esas alır.
- **Güvenilir Yanıtlar:** Dokümanda bilgi bulamazsa, genel bilgiye dayalı olduğunu belirten bir uyarı ile cevap verir.

##  Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler
- Python 3.8+
- [Ollama](https://ollama.com/) (Llama 3.1 modeli yüklü olmalıdır)

### Kütüphanelerin Yüklenmesi
Gerekli Python kütüphanelerini yüklemek için:

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pypdf pandas
```

### Ollama Kurulumu
Bu proje yerel LLM olarak Ollama kullanmaktadır. Ollama'yı kurduktan sonra terminalde şu komutu çalıştırarak gerekli modeli indirin:

```bash
ollama pull llama3.1
```

##  Dosya Yapısı

- **`main.py`**: Projenin ana çalışma dosyasıdır. Agent'ı başlatır, indexi yükler ve sorguları işler.
- **`benchmark.py`**: Sistemin performansını ölçmek için kullanılan test senaryolarını içerir.
- **`storage/`**: Vektör veritabanının (index) kaydedildiği klasördür. İlk çalıştırmada oluşturulur.
- **`gumruk_kanunu.pdf`**: Sistemin bilgi kaynağı olarak kullandığı dokümandır (Not: Bu dosya proje dizininde bulunmalıdır).
- **`*.csv`**: Benchmark test sonuçlarını içeren rapor dosyalarıdır.

##  Kullanım

Proje dizininde `gumruk_kanunu.pdf` dosyasının bulunduğundan emin olun. Ardından asistanı başlatmak için:

```bash
python main.py
```

İlk çalıştırmada sistem PDF dosyasını okuyup vektör veritabanını (`storage/` klasörü altına) oluşturacaktır. Sonraki çalıştırmalarda bu kayıtlı veritabanı kullanılacaktır.

##  Benchmark ve Değerlendirme

Proje, üretilen cevapların kalitesini ölçmek için bir benchmark sistemine sahiptir. Değerlendirme kriterleri şunlardır:
- **Sadakat:** Cevabın kaynak metne ne kadar sadık olduğu.
- **Sayısal Doğruluk:** Sayısal verilerin doğruluğu.
- **Atıf Doğruluğu:** Referans verilen maddelerin doğruluğu.
- **Eksiksizlik:** Cevabın soruyu tam olarak karşılayıp karşılamadığı.
- **Üslup:** Cevabın dili ve anlatımı.

Örnek benchmark sonuçları CSV dosyalarında (örn: `100sorulukgenel_hibrit_benchmark_sonuclari...csv`) bulunabilir.

##  Önemli Notlar

- Sistem, `/content/drive/MyDrive/AgentProject/` yollarını kullanacak şekilde yapılandırılmış olabilir. Kendi yerel ortamınızda çalıştırırken `main.py` içindeki `PERSIST_DIR` ve `pdf_path` değişkenlerini kendi dosya yolunuza göre güncellemeyi unutmayın.
- Loglar `gumruk_asistani_log.txt` dosyasına kaydedilir.
