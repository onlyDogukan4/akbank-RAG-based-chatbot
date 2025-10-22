import os
from dotenv import load_dotenv 
import streamlit as st
import re
import time 
import base64 # YENİ EKLEME: Base64 için
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate

# --- YENİ HTML/CSS YÜKLEME FONKSİYONU ---
def load_css():
    """İstenen tüm düzeltmelerle güncellenmiş CSS (POSITION FIXED)"""
    custom_css = """
    <style>
        /* GENEL VE KAPSAYICILAR */
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f5f7fa 0%, #e0e0e0 100%);
            min-height: 100vh;
        }
        .stApp { 
            max-width: 1200px; 
            margin: auto;
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        /* Tüm Paragraf Yazıları Siyah Yapıldı */
        p, .example-text {
            color: #000000 !important;
            font-size: 15px;
        }
        
        /* SIDEBAR TAMAMEN KALDIRILDI */
        .sidebar { display: none !important; }
        
        /* Ana içerik (Padding ayarı) */
        .main-content {
            width: 100%; 
            padding: 40px;
            color: #000000; 
        }

        /* --- BAŞLIKLAR SİYAH YAPILDI --- */
        h1, h4, 
        h1 *, h4 *,
        .title, 
        .title *, 
        .analysis-item h4, 
        .analysis-item h4 * {
            color: #000000 !important;
            fill: #000000 !important;
        }
        
        /* Başlık Stili */
        .title {
            text-align: center;
            margin-bottom: 30px;
            color: #000000;
            font-size: 36px; 
            font-weight: 900;
        }
        
        .title span {
            color: #000000;
            font-style: italic;
            font-weight: 900;
        }

        /* Streamlit Columns Yapısı */
        .main-content > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) {
            display: flex;
            gap: 40px; 
            /* FIXED alanın boşluğunu simüle etmek için sağdaki içeriği kaydırıyoruz */
            padding-left: 250px; 
        }
        
        /* --- BİLGE ADAM SÜTUNU (POSITION FIXED) --- */
        .wise-man-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 30px;
            padding: 10px; /* Padding azaltıldı, görselin daha rahat sığması için */
            border-right: 1px solid #e0e0e0;
            
            /* **KESİN ÇÖZÜM: POSITION FIXED** */
            position: fixed; /* Sayfa kaydından bağımsız olarak sabitler */
            top: 150px;       /* Üst Sınır: Başlık ve giriş alanından sonra başlar */
            
            /* Genişliği ayarla */
            width: 200px; 
            
            /* Ana kapsayıcının (stApp) merkezi 50% iken, bu kutuyu tam sola hizalamak için:
               left: 50% -> (Ekranın ortası)
               margin-left: -600px (stApp'ın yarısı) + 40px (stApp padding'i) = -560px
            */
            left: 50%;
            margin-left: -560px; /* 1200px genişliğe göre sola hizalar */
            
            height: auto; 
            z-index: 1000; /* Her şeyin üzerinde görünürlük */
            align-self: unset; 
            
            background-color: #ffffff;
        }
        
        /* Konuşma Balonu */
        .speech-bubble {
            background: #ffffff;
            color: #000000; 
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            font-size: 14px;
            line-height: 1.5;
            text-align: center;
            position: relative; 
            width: 100%; 
            max-width: 350px;
            margin-bottom: 10px; /* Görselle arasındaki boşluk azaltıldı */
            order: 1; 
        }
        .speech-bubble::after {
            content: '';
            position: absolute;
            bottom: -15px; 
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 15px solid transparent;
            border-right: 15px solid transparent;
            border-top: 15px solid #ffffff; 
        }
        
        /* Bilge Adam Görseli - Sığması için net boyut kontrolü */
        .wise-man-area img {
            width: 100%; /* Kapsayıcı genişliğine uyum sağlar (200px) */
            max-width: 180px; /* Maksimum 180px genişlikte kalır */
            height: auto;
            border-radius: 50%; 
            order: 2;
            display: block; /* Görselin tam olarak ortalanması için */
            margin: 0 auto;
        }

        /* --- SKORBOARD: KAPSAYICI, BOX VE OPTİK İNCE AYAR --- */
        
        .simulation {
            display: flex;
            gap: 40px;
            margin-bottom: 30px;
            align-items: flex-start;
        }
        
        /* Vücut Görseli Kapsayıcısı */
        .body-image-container { 
            width: 250px; 
            height: 380px; 
            border-radius: 15px;
            overflow: hidden;
            background: #f0f0f0;
            border: 2px solid #1a535c; 
            padding: 10px;
            position: relative;
            margin: 0 auto; 
            margin-bottom: 10px; 
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
        }
        
        /* Tek Score Box Kapsayıcısı */
        .single-score-container {
            display: flex;
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            margin-top: 10px; 
            padding: 10px;
            
            width: 100%; 
            box-sizing: border-box; 
        }

        .score-box {
            background: linear-gradient(145deg, #252525, #151515); 
            border-radius: 12px; 
            
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5), 
                        0 8px 15px rgba(0, 0, 0, 0.6),  
                        inset 0 0 5px rgba(255, 255, 255, 0.15); 

            border: 1px solid rgba(0, 255, 255, 0.3); 
            
            width: 180px; 
            height: 100px; 
            
            position: relative; 
            
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out; 
            z-index: 5; 
            
            margin: 0 auto 10px auto; 
        }

        .score-box:hover {
             transform: translateY(-5px) scale(1.02); 
             box-shadow: 0 0 20px rgba(0, 255, 255, 0.8), 
                         0 10px 20px rgba(0, 0, 0, 0.8);
        }

        /* Skor Değeri (Sayı) */
        .score-value {
            font-size: 50px; 
            font-weight: 900; 
            
            position: absolute; 
            top: 50%; 
            left: 50%; 
            
            transform: translate(-50%, -50%); 
            
            color: #33FFFF !important; 
            
            text-shadow: 0 0 10px #00FFFF, 
                         0 0 20px #00FFFF,
                         0 0 30px #00FFFF; 

            line-height: 1;
            margin: 0; 
            padding: 0;
            z-index: 10; 
            background: none; 
            
            transition: transform 0.3s ease-in-out, text-shadow 0.3s ease-in-out;
        }
        
        .score-box:hover .score-value {
            text-shadow: 0 0 15px #00FFFF, 
                         0 0 30px #00FFFF,
                         0 0 50px #00FFFF; 
            transform: translate(-50%, -50%) scale(1.05); 
        }
        
        /* GENEL SKOR ETİKETİ */
        .score-label-text {
            color: #1a535c; 
            font-size: 18px; 
            font-weight: 700;
            text-align: center;
            padding: 5px 0;
            letter-spacing: 1px;
            transition: transform 0.3s ease-in-out, color 0.3s ease-in-out, text-shadow 0.3s ease-in-out;
        }
        
        .score-box:hover + .score-label-text {
             transform: translateY(-5px); 
             color: #00FFFF; 
             text-shadow: 0 0 5px rgba(0, 255, 255, 0.7);
        }

        .score-label {
            display: none !important; 
        }

        /* --- ANALİZ GRID --- */
        .analysis-grid {
             display: grid;
             grid-template-columns: 1fr 1fr;
             gap: 20px;
             margin-top: 20px;
        }

        .analysis-item {
            background: #f7f7f7; 
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #1a535c; 
        }
        
        .analysis-item h4 {
            color: #000000; 
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: 600;
            border-bottom: 2px solid #1a535c; 
            padding-bottom: 8px;
        }
        
        /* Responsive Düzenlemeler - FIXED'i mobil görünümde kaldırmak kritik */
        @media (max-width: 900px) {
            .main-content > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) {
                 flex-direction: column !important; 
                 gap: 0;
                 padding-left: 0; 
            }
            .simulation {
                flex-direction: column;
                align-items: center;
                gap: 10px;
            }
            .body-image-container {
                width: 100%;
                max-width: 250px; 
                margin: 0 auto;
            }
            .analysis-grid {
                grid-template-columns: 1fr;
            }
            .wise-man-area {
                position: relative; /* Mobil görünümde FIXED iptal edildi */
                border-right: none; 
                border-bottom: 1px solid #e0e0e0; 
                padding-bottom: 20px;
                width: 100%; 
                left: unset;
                top: unset;
                margin-left: unset;
                background-color: transparent; 
            }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- SABİT AYARLAR ---
BILGE_ADAM_AVATAR = "👨‍🔬" 
JSON_PATH = "stylist_rule_set.json" 
CHROMA_DB_DIR = "./chroma_db_gemini_ui" 

# !!! GÖRSEL VE VÜCUT TİPİ EŞLEŞMELERİ !!!
GÖRSEL_KLASÖR = "görseller" 
VUCUT_TIPI_HARITASI = {
    "kum saati": "kumsaati.png",
    "üçgen": "üçgen.png",
    "armut": "armut.png", 
    "ters üçgen": "dikdörtgen.png", 
    "dikdörtgen": "dikdörtgen.png",
    "elma": "elma.png",
    "oval": "elma.png"
}

BILGE_ADAM_PNG_YOLU = "bilge_adam.png"
if not os.path.exists(BILGE_ADAM_PNG_YOLU):
    # Eğer aynı dizinde yoksa görseller klasörüne bak
    BILGE_ADAM_PNG_YOLU = os.path.join(GÖRSEL_KLASÖR, "bilge_adam.png")
    
# Kontrol: Eğer dosya hala bulunamazsa, kullanıcıya uyarı vermek faydalı olabilir.
if not os.path.exists(BILGE_ADAM_PNG_YOLU):
    print(f"UYARI: Bilge Adam görseli şu yollarda bulunamadı: bilge_adam.png veya {os.path.join(GÖRSEL_KLASÖR, 'bilge_adam.png')}")


# --- YARDIMCI FONKSİYONLAR ---

def image_to_base64(image_path):
    """Görsel dosyasını Base64 string'e dönüştürür. Görünürlük sorununu çözer."""
    if not os.path.exists(image_path):
        return "" # Dosya yoksa boş string döndür
        
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # PNG formatında Base64 verisi
        return f"data:image/png;base64,{encoded_string}" 
    except Exception as e:
        print(f"Base64 dönüştürme hatası: {e}")
        return ""


def get_body_type_image_path(body_type):
    """Vücut tipi metnini, 'görseller' klasöründeki dosya yoluyla eşleştirir."""
    normalized_type = body_type.lower().strip()
    filename = VUCUT_TIPI_HARITASI.get(normalized_type, None)
    if filename:
        full_path = os.path.join(GÖRSEL_KLASÖR, filename)
        if os.path.exists(full_path):
            return full_path
        if os.path.exists(filename):
             return filename
    return None 

# --- RAG VE LLM KURULUMU ---
load_dotenv() 
if not os.getenv("GOOGLE_API_KEY"):
    llm = None
else:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 

@st.cache_resource
def setup_rag_chain():
    if not llm:
        return None, None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    template = """
    Sen, kullanıcının kıyafet kombinasyonlarını sadece detaylı stil yorumu ile değerlendiren bir moda stilistisin.
    
    CEVABININ TAMAMINI markdown formatında yaz. Hesaplama detaylarını, ağırlıkları, puanlamaları veya skorları (Genel Skor hariç) ASLA yazma.
    Yorumunu 4 ana parametreye odaklanarak **aşağıdaki formatta, her bir başlık ve yorum arasında iki yeni satır (paragraf ayrımı) bırakarak** hazırla.
    
    Yorumunun en sonuna, sadece ve sadece tek bir satırda, Genel Stil Skorunu (0-100 arasında) '[OVERALL_SCORE:XX]' formatında ekle. XX yerine skoru yaz.
    
    BAĞLAM (Stil Kuralları ve Örnekleri):
    {context}

    KULLANICI GİYSİLERİ VE DURUM: {question}

    ---
    CEVAP YAPISI (Çıktıyı bu sırayla verin):
    ---

    **1. Silüet ve Oran Değerlendirmesi**
    \n\n[Bu kısma sadece, vücut tipine göre giysilerin silüet ve oran dengesine dair detaylı yorum gelecek.]\n\n

    **2. Renk Uyumu ve Palet Analizi**
    \n\n[Bu kısma sadece, renklerin uyumu, psikolojisi ve ten rengine uygunluğuna dair detaylı yorum gelecek.]\n\n

    **3. Kumaş Tipi ve Mevsim Uyumu**
    \n\n[Bu kısma sadece, kumaşların mevsim, etkinlik ve genel doku uyumuna dair detaylı yorum gelecek.]\n\n

    **4. Pratik Denge ve Aksesuar Estetiği**
    \n\n[Bu kısma sadece, kombinin genel estetiği, aksesuar dengesi ve pratikliğine dair detaylı yorum gelecek.]\n\n

    [OVERALL_SCORE:XX] 
    """
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(template)
    
    if not os.path.exists(JSON_PATH):
        documents = [Document(page_content="Stil kuralı veri seti yüklenemedi. Genel moda bilgisi ile analiz yapılacaktır.")]
        st.warning(f"JSON veri seti '{JSON_PATH}' bulunamadı. Genel moda bilgisi kullanılacak.")
    else:
        jq_schema = '.[]' 
        loader = JSONLoader(file_path=JSON_PATH, jq_schema=jq_schema, text_content=False)
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    return retriever, RAG_PROMPT_CUSTOM

def extract_info(query):
    query_lower = query.lower()
    
    match_ust = re.search(r'üst(?:üme|üm| olarak)?\s+(.+?)(?:,\s+altıma| altıma| giydim|$)', query_lower)
    match_alt = re.search(r'alt(?:ıma|ım| olarak)?\s+(.+?)(?: giydim|$)', query_lower)

    vucut_tipi_keywords = ["kum saati", "üçgen", "armut", "ters üçgen", "dikdörtgen", "elma", "oval"]
    vucut_tipi_raw = "Belirtilmedi"
    
    for tip in vucut_tipi_keywords:
        if re.search(r'\b' + re.escape(tip) + r'\b', query_lower):
            vucut_tipi_raw = tip
            break
        
    ust = match_ust.group(1).strip() if match_ust else "Belirtilmedi"
    alt = match_alt.group(1).strip() if match_ust else "Belirtilmedi" 
    
    st.session_state.simulated_outfit = {
        "ust": ust.capitalize(), 
        "alt": alt.capitalize(), 
        "vucut_tipi": vucut_tipi_raw.capitalize() 
    }

    return ust.capitalize(), alt.capitalize(), vucut_tipi_raw.capitalize()

def parse_response_and_score(full_response):
    score_match = re.search(r'\[OVERALL_SCORE:(\d+)\]', full_response)
    
    if score_match:
        overall_score = score_match.group(1)
        comment_only = re.sub(r'\[OVERALL_SCORE:\d+\]', '', full_response).strip()
    else:
        overall_score = "??"
        comment_only = full_response
        
    return comment_only, overall_score

def parse_analysis_sections(comment_only):
    sections = {
        "siluet": "Analiz alınamadı.",
        "renk": "Analiz alınamadı.",
        "kumas": "Analiz alınamadı.",
        "aksesuar": "Analiz alınamadı."
    }
    
    pattern_siluet = r"\*\*1\. Silüet ve Oran Değerlendirmesi\*\*\s*\n\n(.*?)(?=\n\n\*\*|\Z)"
    pattern_renk = r"\*\*2\. Renk Uyumu ve Palet Analizi\*\*\s*\n\n(.*?)(?=\n\n\*\*|\Z)"
    pattern_kumas = r"\*\*3\. Kumaş Tipi ve Mevsim Uyumu\*\*\s*\n\n(.*?)(?=\n\n\*\*|\Z)"
    pattern_aksesuar = r"\*\*4\. Pratik Denge ve Aksesuar Estetiği\*\*\s*\n\n(.*?)(?=\n\n\*\*|\Z)"

    match_siluet = re.search(pattern_siluet, comment_only, re.DOTALL)
    match_renk = re.search(pattern_renk, comment_only, re.DOTALL)
    match_kumas = re.search(pattern_kumas, comment_only, re.DOTALL)
    match_aksesuar = re.search(pattern_aksesuar, comment_only, re.DOTALL)

    if match_siluet:
        sections["siluet"] = match_siluet.group(1).strip()
    if match_renk:
        sections["renk"] = match_renk.group(1).strip()
    if match_kumas:
        sections["kumas"] = match_kumas.group(1).strip()
    if match_aksesuar:
        sections["aksesuar"] = match_aksesuar.group(1).strip()
        
    return sections

def get_wise_comment(user_input):
    comments = [
        "Hm, ilginç bir kombinasyon düşünüyorsunuz! Bakalım analizimiz ne gösterecek...",
        "Vay canına, bu tarz bir arayayış içindesiniz! Size özel tavsiyelerim var.",
        "Harika bir stil anlayışınız var! Ancak küçük dokunuşlarla mükemmele ulaşabilirsiniz.",
        "Bu kombinasyon üzerinde biraz çalışmamız gerekecek gibi görünüyor!",
        "Mükemmel bir başlangıç noktası! Gelin birlikte bu kombinasyonu geliştirelim.",
        "Vücut tipiniz için bazı harika seçenekler önerebilirim!",
        "Renk seçiminiz dikkat çekici! Ancak silüetiniz için daha iyi alternatifler de mevcut."
    ]
    
    import random
    return random.choice(comments)


# --- STREAMLIT ARAYÜZÜ ---

st.set_page_config(page_title="Absürt Stil Danışmanı", layout="wide") 
load_css() 

# --- RAG SİSTEMİ BAŞLATMA ---
try:
    retriever, RAG_PROMPT_CUSTOM = setup_rag_chain() 
    if not retriever and os.getenv("GOOGLE_API_KEY"):
        st.error("RAG sistemi başlatılamadı. Veri seti (JSON) veya ChromaDB hatası olabilir.")
        st.stop()
    if not os.getenv("GOOGLE_API_KEY") and llm is None:
         st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")
except Exception as e:
    st.error(f"Sistem Başlatılamadı: {e}. Model adı veya API anahtarınızı kontrol edin.")
    st.stop()

# --- OTURUM DURUMU BAŞLATMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.simulated_outfit = {"ust": "Henüz", "alt": "Girilmedi", "vucut_tipi": "Belirtilmedi"}
    st.session_state.last_overall_score = "??" 
    st.session_state.last_comment = "" 
    st.session_state.analysis_parts = {}
    st.session_state.show_results = False
    st.session_state.wise_comment = "Merhaba! Vücut tipinizi ve giyim tercihinizi anlatan bir mesaj yazın, size özel moda önerileri sunayım."

# --- ANA KAPSAYICI ---
main_container = st.container()

with main_container:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # 1. Başlık
    st.markdown('<h1 class="title">Moda ve Stil Danışmanı <span>Profesör Zıpır</span></h1>', unsafe_allow_html=True)
    
    # 2. İki Ana Sütun Oluşturma 
    col_professor, col_content = st.columns([1.2, 2.8]) 

    # --- Sol Sütun: Profesör (FIXED) ---
    with col_professor:
        
        wise_comment = st.session_state.wise_comment
        
        # *** GÜNCELLEME: Görsel Base64 ile yükleniyor ***
        image_src_b64 = image_to_base64(BILGE_ADAM_PNG_YOLU)
        
        # Eğer Base64 dönüşümü başarılı olmazsa, bu görsel gözükmeyecektir. 
        # Base64 dönüşümü başarılıysa, bu görsel tarayıcıda doğrudan çizilir.

        if image_src_b64:
            wise_man_html = f"""
            <div class="wise-man-area">
                <div class="speech-bubble">{wise_comment}</div>
                <img src="{image_src_b64}" alt="Bilge Adam Avatarı">
            </div>
            """
        else:
            # Base64 dönüşümü başarısız olursa, orijinal dosya yolunu deneriz, 
            # ancak tarayıcı bunu muhtemelen göremeyecektir (Sorunun nedeni buydu).
            wise_man_html = f"""
            <div class="wise-man-area">
                <div class="speech-bubble">{wise_comment}</div>
                <p style='color:red; text-align:center;'>Görsel yüklenemedi. 'bilge_adam.png' dosyasını kontrol edin.</p>
                <img src="{BILGE_ADAM_PNG_YOLU}" alt="Bilge Adam Avatarı">
            </div>
            """
        
        st.markdown(wise_man_html, unsafe_allow_html=True)


    # --- Sağ Sütun: Giriş ve Sonuçlar (Kaydırılabilir İçerik) ---
    with col_content:
        
        # Giriş Bölümü
        with st.form("moda_analiz_form"):
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            
            user_input = st.text_area(
                "Moda Durumunuzu Açıklayın",
                placeholder="Örneğin: Kum saati vücut tipine sahibim ve iş için resmi bir kombin arıyorum. Mavi bir ceket ve siyah pantolon düşünüyorum. Sizce bu kombin uygun mu?",
                height=120,
                key="user_input"
            )
            
            st.markdown('<div class="example-text">Vücut tipinizi, giyim tercihinizi ve özel durumunuzu detaylı şekilde açıklayın.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze_clicked = st.form_submit_button("Moda Analizi Yap", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sonuç Bölümü
        if 'show_results' in st.session_state and st.session_state.show_results:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            
            # Simülasyon ve Skor
            st.markdown('<div class="simulation">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Vücut Tipi Görseli Kapsayıcısı (body-image-container)
                st.markdown('<div class="body-image-container">', unsafe_allow_html=True)
                display_body_type = st.session_state.simulated_outfit["vucut_tipi"]
                body_type_path = get_body_type_image_path(display_body_type)
                
                if body_type_path and os.path.exists(body_type_path):
                    # Not: Bu görsel st.image ile yüklendiği için Streamlit tarafından doğru sunulur
                    st.image(body_type_path, use_container_width=True)
                    st.markdown(f'<div class="body-info-label">Vücut Tipi: {display_body_type}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="body-info-label"><strong>{display_body_type} Vücut Tipi</strong><br>Görsel bulunamadı</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True) 
            
            with col2:
                # NİHAİ SKOR KUTUSU
                current_score = st.session_state.last_overall_score
                
                score_html = f"""
                <div class="single-score-container">
                    <div class="score-box">
                        <div class="score-value">{current_score}</div>
                    </div>
                    <div class="score-label-text">GENEL SKOR</div>
                </div>
                """
                st.markdown(score_html, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 2x2 Analiz Grid
            if 'analysis_parts' in st.session_state and st.session_state.analysis_parts:
                parts = st.session_state.analysis_parts
                st.markdown('<div class="analysis-grid">', unsafe_allow_html=True)
                
                # Kutu 1: Silüet
                st.markdown('<div class="analysis-item">', unsafe_allow_html=True)
                st.markdown('<h4>1. Silüet ve Oran Değerlendirmesi</h4>', unsafe_allow_html=True)
                st.markdown(parts["siluet"], unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Kutu 2: Renk
                st.markdown('<div class="analysis-item">', unsafe_allow_html=True)
                st.markdown('<h4>2. Renk Uyumu ve Palet Analizi</h4>', unsafe_allow_html=True)
                st.markdown(parts["renk"], unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Kutu 3: Kumaş
                st.markdown('<div class="analysis-item">', unsafe_allow_html=True)
                st.markdown('<h4>3. Kumaş Tipi ve Mevsim Uyumu</h4>', unsafe_allow_html=True)
                st.markdown(parts["kumas"], unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Kutu 4: Aksesuar
                st.markdown('<div class="analysis-item">', unsafe_allow_html=True)
                st.markdown('<h4>4. Pratik Denge ve Aksesuar Estetiği</h4>', unsafe_allow_html=True)
                st.markdown(parts["aksesuar"], unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) 


# --- FORM GÖNDERİM İŞLEMİ ---
if analyze_clicked and user_input:
    st.session_state.show_results = True

    st.session_state.wise_comment = get_wise_comment(user_input)
    
    ust_giyim, alt_giyim, vucut_tipi = extract_info(user_input)
    
    full_prompt_content = user_input
    current_body_type = vucut_tipi
    current_upper = ust_giyim
    current_lower = alt_giyim
    
    full_prompt_content += f" (Vücut Tipi: {current_body_type}, Üst Giyim: {current_upper}, Alt Giyim: {current_lower})"

    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # LLM yoksa analizi atla
    if not llm:
        st.session_state.last_overall_score = "??"
        st.session_state.analysis_parts = {k: "API Anahtarı eksik olduğu için analiz yapılamadı." for k in ["siluet", "renk", "kumas", "aksesuar"]}
        st.rerun() 

    with st.spinner("Absürt Bilge Adam Kuralları Analiz Ediyor ve Yorumluyor..."):
        try:
            retrieved_docs = retriever.invoke(full_prompt_content)
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            
            final_prompt_value = RAG_PROMPT_CUSTOM.format(
                context=context,
                question=full_prompt_content
            )
            
            llm_response = llm.invoke(final_prompt_value)
            full_response = llm_response.content
            
            comment_only, overall_score = parse_response_and_score(full_response)
            
            analysis_parts = parse_analysis_sections(comment_only)
            
            st.session_state.last_comment = comment_only
            st.session_state.last_overall_score = overall_score
            st.session_state.analysis_parts = analysis_parts
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            st.rerun() 
            
        except Exception as e:
            error_msg = f"Absürt Bilge Adam şu anda yanıt veremiyor. Bir hata oluştu: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})