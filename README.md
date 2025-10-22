🤖 Akbank RAG Tabanlı Sohbet Botu (Moda Danışmanı)
1. Projenin Amacı
Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, Retrieval Augmented Generation (RAG) mimarisini kullanarak kullanıcılara "moda bilginizi ve tarzınızı bir sonraki aşamaya taşımak" amacı doğrultusunda kişiselleştirilmiş stil ve trend danışmanlığı sunan gelişmiş bir yapay zeka bilgi robotudur.

Projenin temel hedefi, güçlü gemini-2.5-flash modelinin yaratıcı yeteneklerini, akademik kaynaklardan elde edilen kanıtlanmış bilgi bankasının doğruluğu ile birleştirerek, kullanıcının stil ve moda sorularına hızlı, ilgili ve halüsinasyonsuz yanıtlar sağlamaktır.

2. Veri Seti Hakkında Bilgi
Bu robotun bilgi tabanını, Google Scholar'da yayımlanan akademik moda makalelerinden türetilmiş anlamsal önermeler oluşturmaktadır.
Toplanış/Hazırlanış Metodolojisi: Veri seti, Google Scholar platformunda yer alan akademik moda makalelerinden elde edilen, en kritik ve anlamsal önerme barındıran metin parçaları toplanarak hazırlanmıştır. Bu metodoloji, botun verdiği bilgilerin akademik ve güvenilir bir temele dayanmasını sağlamıştır.
İçerik: Moda tarihi, güncel trend analizi, sürdürülebilirlik, tekstil bilimi ve stil psikolojisi gibi akademik konuları kapsar.

3. Kullanılan Yöntemler ve Çözüm Mimarisi
Projemiz, LangChain orkestrasyon çerçevesi kullanılarak tasarlanmış bir RAG (Retrieval Augmented Generation) mimarisi üzerinde çalışır
RAG frameworkü = LangChain,Tüm RAG sürecini yönetmek
LLM modeli = Gemini API (gemini-2.5-flash) ,Çekilen bağlamı kullanarak nihai ve kaliteli yanıtı üretmek.
Vektör Veritabanı = ChromaDB ,Vektörleri depolamak ve anlamsal arama yapmak. 
Embedding = Gemini API Embedding Modeli , Metin parçalarını vektörel gösterimlere dönüştürmek.

RAG Akışı:
Parçalama: Akademik önermeler, LangChain'in TextSplitter bileşenleri ile küçük parçalara ayrılır.
Gömme ve Depolama: Parçalar, Gemini'nin Embedding modeli kullanılarak vektörlere dönüştürülür ve ChromaDB'ye kaydedilir.
Arama: Kullanıcının sorusu bir vektöre dönüştürülür ve LangChain'in Retriever bileşeni ile ChromaDB'den en ilgili bağlamlar çekilir.
Üretim: Çekilen bu bağlam, gemini-2.5-flash modeline gönderilerek nihai, kanıta dayalı moda önerisi oluşturulur.

4. Elde Edilen Sonuçlar (Sonuçlar Kriteri)
Doğruluk oranı kesin metriklerle ölçülmemiş olsa da, sohbet botu kullanıcı sorgularına karşı tutarlı, derinlikli ve çok güzel öneriler sunmaktadır. Akademik makalelerden elde edilen bağlam sayesinde, botun yanıtları genel internet kaynaklarından farklı olarak güçlü ve yetkin bir bilgi birikimine dayanmaktadır.

5. Web Arayüzü & Product Kılavuzu
Projenin arayüzü Streamlit kullanılarak VS Code IDE'sinde geliştirilmiştir.
Projenin Çalışma Adresi: http://localhost:8501
>>>>>>> f6513113aa7416e8a8cb5224b45f2ebee84342d1

Aşağıdaki GIF, projenin arayüzde nasıl çalıştığını ve temel kabiliyetlerini (soru sorma, bağlama dayalı yanıt alma) göstermektedir:

assets/gif.gif

<<<<<<< HEAD
Geliştirme Ortamı ve Çalışma Kılavuzu
1-Gereksinimler: Python 3.10+, Git.

2-Depoyu Klonlama ve Ortam Kurulumu: git clone https://github.com/onlyDogukan4/akbank-RAG-based-chatbot.git cd akbank-RAG-based-chatbot python -m venv venv .\venv\Scripts\activate # Windows için 3-Bağımlılıkları Yükleme: pip install -r requirements.txt

4-API Anahtarını Ayarlama: GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

5-Çalıştırma: streamlit run app.py
=======
6. Geliştirme Ortamı ve Çalışma Kılavuzu

  1-Gereksinimler: Python 3.10+, Git.

  2-Depoyu Klonlama ve Ortam Kurulumu:
    git clone https://github.com/onlyDogukan4/akbank-RAG-based-chatbot.git
    cd akbank-RAG-based-chatbot
    python -m venv venv
    .\venv\Scripts\activate  # Windows için
  3-Bağımlılıkları Yükleme:
      pip install -r requirements.txt

  4-API Anahtarını Ayarlama:
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
  
  5-Çalıştırma:
    streamlit run app.py



>>>>>>> f6513113aa7416e8a8cb5224b45f2ebee84342d1
