import os
from dotenv import load_dotenv 
import streamlit as st
import re
import time 
# KRİTİK DÜZELTME 1: LangChain'in yeni versiyonunda 'Document' sınıfı buradan geliyor
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
from PIL import Image

# --- HTML/CSS YÜKLEME FONKSİYONU ---
def load_css():
    """CSS'i okur ve Streamlit'e enjekte eder, Bilge Adam arayüzüne uygun stil ekler."""
    custom_css = """
    <style>
        /* Genel Font Ayarı */
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        /* Sol Sütun (Bilge Adam ve Görsel Alan) */
        [data-testid="stSidebar"] + div > [data-testid="stVerticalBlock"] > div:first-child {
            background-color: #f7f7f7; 
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            max-width: 100%;
        }

        /* Bilge Adam Avatarı (Karikatürize Görünüm için) */
        .bilge-adam-avatar { 
            font-size: 5em; 
            text-align: center; 
            margin-bottom: 20px; 
            animation: bounce 1s infinite alternate; /* Absürtlük için hafif animasyon */
        }
        
        /* Kullanıcı Fotoğraf Alanı */
        .user-photo-container {
            border: 2px dashed #007bff;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin-bottom: 15px;
            background-color: #ffffff;
        }
        .user-photo-container img {
            border-radius: 8px;
            max-width: 100%;
            height: auto;
        }

        /* Konuşma Balonu (ANA ASİSTAN YANITI) */
        .assistant-bubble {
            background-color: #e6f7ff; 
            border-left: 7px solid #007bff; 
            padding: 25px;
            border-radius: 10px;
            margin-top: 25px; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
            min-height: 250px; 
        }
        .assistant-bubble h3, .assistant-bubble strong { color: #0056b3; } 
        
        /* Kullanıcı Mesajı (Geçmiş Mesajlar) */
        .user-message-compact {
            background-color: #ffffff;
            border-bottom: 1px solid #eee;
            padding: 8px;
            border-radius: 0;
            margin-bottom: 5px;
            font-size: 0.85em;
            text-align: right;
            color: #6c757d;
        }

        /* Animasyon */
        @keyframes bounce {
            0% { transform: translateY(0); }
            100% { transform: translateY(-5px); }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- SABİT AYARLAR ---
BILGE_ADAM_AVATAR = "👨‍🔬" # Absürt Profesör/Bilim Adamı (Görsel bulunamazsa yedek emoji)
# !!! DÜZELTİLDİ: Dosya adınız ile eşleşmesi için !!!
BILGE_ADAM_IMAGE_PATH = "bilge_adam.png" 

STYLING_COLUMN_WIDTH = 0.3
CHAT_COLUMN_WIDTH = 0.7

JSON_PATH = "stylist_rule_set.json" 
CHROMA_DB_DIR = "./chroma_db_gemini_ui" 

# --- RAG VE LLM KURULUMU ---
load_dotenv() 
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY ortam değişkeni ayarlanmamış.")
    st.stop() 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 

# --- FONKSİYON: RAG ZİNCİRİNİ BAŞLATMA ---
@st.cache_resource
def setup_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # PROMPT: (Puanlama Prompt'u)
    template = """
    Sen, kullanıcının kıyafet kombinasyonlarını puanlayan ve detaylı stil eleştirisi yapan bir moda stilisti ve yapay zeka jürisisin.
    Görevin, kombinasyonu aşağıdaki **4 ana parametre** üzerinden ayrı ayrı değerlendirip puanlamak ve en son genel skoru hesaplamaktır.
    
    CEVABININ TAMAMINI markdown formatında yaz. Cevap akışını bozma, her zaman puanlamayla başla.

    **PUANLAMA ÇARPANLARI:** Silüet %40, Renk %30, Kumaş/Mevsim %20, Pratik Estetik %10.

    BAĞLAM (Stil Kuralları ve Örnekleri):
    {context}

    KULLANICI GİYSİLERİ VE DURUM: {question}

    Kullanıcı Bilgileri: Boy: {boy_bilgisi}, Kilo: {kilo_bilgisi}.

    ---
    CEVAP YAPISI (Çıktıyı bu sırayla verin):
    ---

    ### 1. Silüet Değerlendirmesi ve Puanı (Ağırlık: %40)
    1.1. Analiz, İhlaller ve Düşülen Puanlar.
    **[Silüet Puanı] / 100**

    ### 2. Renk Değerlendirmesi ve Puanı (Ağırlık: %30)
    2.1. Analiz, İhlaller ve Düşülen Puanlar.
    **[Renk Puanı] / 100**

    ### 3. Kumaş Tipi ve Mevsim Değerlendirmesi ve Puanı (Ağırlık: %20)
    3.1. Analiz, İhlaller ve Düşülen Puanlar.
    **[Kumaş/Mevsim Puanı] / 100**

    ### 4. Pratik Denge ve Estetik İndirimi/Bonusu (Ağırlık: %10)
    4.1. Analiz ve Detaylı Öneriler.
    4.2. Uygulanan İndirim/Bonus Puanı.
    **[Estetik İndirim/Bonus Puanı] / 10**

    ### 5. Genel (OVERALL) Stil Skoru
    Ağırlıklı ortalama ile Genel Skoru hesapla.
    **[Puan] / 100**
    """
    RAG_PROMPT_CUSTOM = PromptTemplate.from_template(template)
    
    if not os.path.exists(JSON_PATH):
        documents = [Document(page_content="Stil kuralı veri seti yüklenemedi. Genel moda bilgisi ile analiz yapılacaktır.")]
        st.warning(f"JSON veri seti '{JSON_PATH}' bulunamadı. Lütfen dosyayı oluşturun.")
    else:
        jq_schema = '.[]' 
        loader = JSONLoader(file_path=JSON_PATH, jq_schema=jq_schema, text_content=False)
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    st.success("Stil Kuralları ve Model Başlatıldı!")
    return retriever, RAG_PROMPT_CUSTOM

# --- FONKSİYON: VÜCUT TİPİNİ VE KIYAFETLERİ SORGUDAN ÇIKARMA ---
def extract_info(query):
    # Sadece giyim öğelerini çıkar
    match_ust = re.search(r'üst(üme|üm| olarak)\s+(.+?)(?:,\s+altıma| altıma| giydim|$)', query.lower())
    match_alt = re.search(r'alt(ıma|ım| olarak)\s+(.+?)(?: giydim|$)', query.lower())
    
    ust = match_ust.group(2).strip() if match_ust else "Belirtilmedi"
    alt = match_alt.group(2).strip() if match_alt else "Belirtilmedi"

    return ust.capitalize(), alt.capitalize()

# --- FONKSİYON: TAHMİNİ YAZMA ETKİSİ (TYPING EFFECT) ---
def stream_response(response_text):
    """Yanıt metnini yavaş yavaş yazdırarak Stream efektini taklit eder."""
    full_text = ""
    placeholder = st.empty()
    
    words = response_text.split()
    for word in words:
        full_text += word + " "
        placeholder.markdown(full_text, unsafe_allow_html=True)
        time.sleep(0.015) 
    
    return full_text

# --- STREAMLIT ARAYÜZÜ (MAIN) ---

st.set_page_config(page_title="Absürt Stil Danışmanı", layout="wide")
load_css() 

st.title("🎩 AI Stil Danışmanı ve Puanlayıcı")
st.caption("Kombininizin fotoğrafını yükleyin ve detayları yazın. Bilge Adam, oyun karakteri titizliğiyle puanlayacak!")

try:
    retriever, RAG_PROMPT_CUSTOM = setup_rag_chain() 
except Exception as e:
    st.error(f"Sistem Başlatılamadı: {e}.")
    st.stop()


# --- OTURUM DURUMU BAŞLATMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.simulated_outfit = {"ust": "Henüz", "alt": "Girilmedi"}
    st.session_state.boy = ""
    st.session_state.kilo = ""
    st.session_state.uploaded_file = None
    st.session_state.streamed_last_message = False

    initial_message = (
        "**Selam Genç Stil Avcısı!** Ben senin Absürt Stil Uzmanınım. Sol tarafa fotoğrafını yükle, "
        "boy/kilo bilgilerini gir ve **aşağıdaki kutuya vücut tipini, mevsimi ve kıyafet detaylarını** yaz. "
        "Puanlamaya hazır ol!"
    )
    st.session_state.messages.append({"role": "assistant", "content": initial_message})


# İki ana sütun oluşturma
col_styling, col_chat = st.columns([STYLING_COLUMN_WIDTH, CHAT_COLUMN_WIDTH])

# --- SOL SÜTUN (STİL UZMANI VE GİRDİ ALANLARI) ---
with col_styling:
    
    # Bilge Adam Görsel Alanı
    if os.path.exists(BILGE_ADAM_IMAGE_PATH):
        st.image(BILGE_ADAM_IMAGE_PATH, caption="Stil Uzmanı: Absürt Bilge Adam", use_column_width=True)
    else:
        # Eğer dosya bulunamazsa yedek emoji kullan
        st.markdown(f'<div class="bilge-adam-avatar">{BILGE_ADAM_AVATAR}</div>', unsafe_allow_html=True)
        st.header("Stil Uzmanı: Bilge Adam (Görsel Yüklenemedi)")
    
    st.markdown("---")
    
    st.subheader("Kombin Fotoğrafı Yükle")
    uploaded_file = st.file_uploader("Boydan Çekilmiş Fotoğraf (Opsiyonel)", type=["jpg", "jpeg", "png"], key="photo_uploader")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.markdown('<div class="user-photo-container">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Analiz Edilecek Kombin", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state.uploaded_file = None
        st.markdown('<div class="user-photo-container">Analiz için fotoğraf yükleyin.</div>', unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("Kullanıcı Bilgileri")
    # Boy ve Kilo Girişi
    st.session_state.boy = st.text_input("Boy (cm):", value=st.session_state.boy, key="boy_input")
    st.session_state.kilo = st.text_input("Kilo (kg):", value=st.session_state.kilo, key="kilo_input")
    
    if st.session_state.uploaded_file:
         st.info("Görsel yüklendi. Görsel analiz yeteneği (Gemini) ile kombin detaylarını tahmin edebilirim.")

# --- SAĞ SÜTUN (BİLGE ADAM'IN KONUŞMA ALANI) ---
with col_chat:
    
    st.subheader("Stil Analizi Sonuçları")
    
    # 1. Önceki Analizler (Kompakt Mesajlar)
    if len(st.session_state.messages) > 1:
        st.markdown("**Geçmiş Analizler:**")
        # Son asistan yanıtını gösterme
        for message in st.session_state.messages[:-1]:
            if message["role"] == "user":
                 st.markdown(f'<div class="user-message-compact">👤 **Ben:** {message["content"]}</div>', unsafe_allow_html=True)
            else: 
                 pass 

    # 2. Son Yanıtı BÜYÜK BALONDA göster
    st.markdown(f'<div class="assistant-bubble">', unsafe_allow_html=True)
    # En son asistan yanıtını çek
    last_response = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    
    # Eğer yanıt zaten stream edildiyse, tam metni göster.
    if last_response and st.session_state.get('streamed_last_message', False):
         st.markdown(last_response) 
    elif last_response and not st.session_state.get('streamed_last_message', False):
        # İlk render'da mesajı gösterme (boş kalmaması için)
        pass 
         
    st.markdown('</div>', unsafe_allow_html=True)


# --- ANA KULLANICI GİRİŞİ (Sayfanın Altına Eklenir) ---
if prompt := st.chat_input("Vücut tipini (örn: Kum Saati), mevsimi ve giysi detaylarını yazın..."):
    
    # Yeni soru geldiğinde stream bayrağını sıfırla
    st.session_state.streamed_last_message = False

    # 1. Kullanıcı mesajını anında ekrana yansıt 
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Kıyafetleri çıkar 
    ust_giyim, alt_giyim = extract_info(prompt)
    st.session_state.simulated_outfit = {"ust": ust_giyim, "alt": alt_giyim}
    
    # LLM'e göndermek için gerekli veriyi topla
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    # Prompt'u son kez oluştur
    final_prompt_value = RAG_PROMPT_CUSTOM.format(
        context=context,
        question=prompt,
        boy_bilgisi=st.session_state.boy if st.session_state.boy else "Belirtilmedi",
        kilo_bilgisi=st.session_state.kilo if st.session_state.kilo else "Belirtilmedi"
    )

    # --- Puanlama ve Stream Etkisi ---
    with st.spinner("Absürt Bilge Adam Kuralları Analiz Ediyor ve Puanlama Yapıyor..."):
        try:
            llm_response = llm.invoke(final_prompt_value)
            full_response = llm_response.content
            
            # Yanıtı oturum durumuna ekle
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # KRİTİK DÜZELTME 3: st.rerun() kullan
            st.rerun()
            
        except Exception as e:
            # Hata oluştuğunda da arayüzü güncelle
            error_msg = f"Absürt Bilge Adam şu anda yanıt veremiyor. Bir hata oluştu: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.streamed_last_message = True 
            st.rerun()

# --- Yeniden Çalıştırma Sonrası Stream Efekti ---
if 'messages' in st.session_state and st.session_state.messages:
    last_assistant_message = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    
    if last_assistant_message and not st.session_state.get('streamed_last_message', False):
        
        # Son mesajı, daha önce gösterilmemişse stream et
        with col_chat:
            # Tekrar bir balon içine alıp yavaş yazdır
            st.markdown(f'<div class="assistant-bubble">', unsafe_allow_html=True)
            
            # Stream efektini uygula
            stream_response(last_assistant_message["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Stream tamamlandı işaretini koy
            st.session_state.streamed_last_message = True