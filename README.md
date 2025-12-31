# 🏥 Diyabet Tahmin Sistemi - Gelişmiş Yapay Sinir Ağı (ANN)

Bu proje, **Pima Indians Diabetes** veri setini kullanarak, bir kişinin sağlık parametrelerine (Glikoz, BMI, Yaş vb.) dayanarak diyabet riskini tahmin eden bir **Derin Öğrenme** modelidir. Proje kapsamında veri temizleme, özellik mühendisliği ve katmanlı sinir ağı mimarisi (ANN) kullanılmıştır.

## 🚀 Proje Özellikleri ve Uygulanan Teknikler

Bu çalışmada, Yapay Sinir Ağları derslerinde işlenen temel ve ileri düzey kavramlar kodlanmıştır:

* **Veri Ön İşleme:** Eksik verilerin (Glikoz, BMI içindeki 0 değerleri) medyan ile doldurulması ve verinin ölçeklendirilmesi (**StandardScaler**).
* **Özellik Mühendisliği:** Model performansını artırmak için `Glucose_Insulin_Ratio` gibi yeni öznitelikler türetilmiştir.
* **Mimari:** Çok katmanlı, ileri beslemeli (**Feedforward**) bir Yapay Sinir Ağı.
* **Aktivasyon Fonksiyonları:** Gizli katmanlarda `ReLU`, çıkış katmanında ikili sınıflandırma için `Sigmoid`.
* **Optimizasyon:** Ağırlıkların güncellenmesi için **Adam Optimizer** kullanılmıştır.
* **Düzenlileştirme (Regularization):** Ezberlemeyi önlemek için **Dropout** ve öğrenmeyi hızlandırmak için **BatchNormalization** katmanları eklenmiştir.

## 🧠 Model Mimarisi

Model, TensorFlow/Keras kullanılarak şu yapıda oluşturulmuştur:
1. **Giriş Katmanı:** 8+ Özellik (Feature)
2. **Gizli Katmanlar:** 64, 32 ve 16 nöronluk kademeli yapı.
3. **BatchNormalization & Dropout:** Eğitim stabilitesi ve aşırı öğrenmeyi (Overfitting) engelleme.
4. **Çıkış Katmanı:** 1 Nöron (Sigmoid) ile olasılık tahmini.

## 📊 Performans ve Görselleştirme

Eğitim süreci sonunda elde edilen başarı metrikleri ve grafikler:
* **Eğitim/Validasyon Kaybı (Loss) ve Doğruluğu (Accuracy)** grafikleri oluşturulmuştur.
* **Confusion Matrix** ile modelin tahmin başarısı analiz edilmiştir.
* **ROC Eğrisi** ile modelin ayırt ediciliği doğrulanmıştır.

> **Not:** Grafik detaylarına `diabetes_model_results.png` dosyasından ulaşabilirsiniz.

## 🛠️ Kullanılan Teknolojiler

* **Python 3.x**
* **TensorFlow / Keras** (Derin Öğrenme Modeli)
* **Scikit-Learn** (Veri İşleme ve Metrikler)
* **Pandas & Numpy** (Veri Analizi)
* **Matplotlib & Seaborn** (Görselleştirme)

---
*Bu proje bir eğitim çalışması olarak geliştirilmiştir.*