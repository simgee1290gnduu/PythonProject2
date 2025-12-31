# 🏥 Diyabet Tahmin Sistemi - Gelişmiş Derin Öğrenme Projesi (ANN)

Bu proje, tıbbi ölçüm verilerini analiz ederek bir bireyin diyabet olup olmadığını yüksek doğrulukla tahmin eden bir **Yapay Sinir Ağı (Artificial Neural Network)** modelidir. Proje, veri ön işlemeden modelin yayına hazırlanmasına kadar tüm uçtan uca veri bilimi süreçlerini kapsamaktadır.

## 📊 1. Veri Seti Analizi (Pima Indians Diabetes)
Veri seti, Ulusal Diyabet ve Sindirim ve Böbrek Hastalıkları Enstitüsü'nden alınmıştır. Model, aşağıdaki 8 temel özelliği (feature) girdi olarak kabul eder:

* **Pregnancies:** Hamile kalma sayısı.
* **Glucose:** 2 saatlik oral glikoz tolerans testindeki plazma glikoz konsantrasyonu. (0 değerleri medyan ile temizlenmiştir).
* **BloodPressure:** Diyastolik kan basıncı (mm Hg).
* **SkinThickness:** Triceps deri kıvrım kalınlığı (mm).
* **Insulin:** 2 saatlik serum insülini (mu U/ml).
* **BMI (Vücut Kitle İndeksi):** Kilo / (Boy)^2.
* **DiabetesPedigreeFunction:** Soy ağacına dayalı diyabet olasılık fonksiyonu.
* **Age:** Yaş (Yıl).

## 🧠 2. Gelişmiş Model Mimarisi
Model, doğrusal olmayan karmaşık ilişkileri öğrenebilmek için çok katmanlı bir yapı üzerine inşa edilmiştir:

| Katman | Tip | Özellik | Aktivasyon |
| :--- | :--- | :--- | :--- |
| **Giriş** | Dense | 8 Özellik Girişi | ReLU |
| **Gizli 1** | Dense | 64 Nöron + BatchNormalization | ReLU |
| **Düzenleme** | Dropout | %30 Oranında Söndürme | - |
| **Gizli 2** | Dense | 32 Nöron + BatchNormalization | ReLU |
| **Gizli 3** | Dense | 16 Nöron | ReLU |
| **Çıkış** | Dense | 1 Nöron (Sınıflandırma) | Sigmoid |

### Uygulanan Teknik Detaylar:
* **Backpropagation:** Hataların minimize edilmesi için geri yayılım algoritması kullanılmıştır.
* **Optimization:** Hızlı yakınsama için **Adam Optimizer** (LR: 0.001) tercih edilmiştir.
* **Regularization:** Aşırı öğrenmeyi (Overfitting) engellemek için **Dropout** ve her katmanda veriyi normalize eden **BatchNormalization** eklenmiştir.
* **Callbacks:** `EarlyStopping` ile modelin bozulmaya başladığı noktada eğitim durdurulmuş, `ReduceLROnPlateau` ile takılma noktalarında öğrenme hızı otomatik düşürülmüştür.

## 📈 3. Eğitim Grafikleri ve Görselleştirme
Modelin eğitim sürecindeki başarısı ve hata payının düşüşü aşağıdaki grafiklerde net bir şekilde görülmektedir:

![Model Performans Analizi](diabetes_model_results.png)

*Yukarıdaki grafikte; eğitim ve doğrulama (validation) süreçlerinin birbirine yakınlığı, modelin ezberlemediğini (generalization) kanıtlamaktadır.*

## 🎯 4. Başarı Metrikleri (Model Evaluation)
Test verileri üzerinde elde edilen detaylı performans sonuçları:

* **Doğruluk (Accuracy):** %74.00+
* **Kesinlik (Precision):** %65.50 (Pozitif tahminlerin doğruluğu)
* **Duyarlılık (Recall):** %62.00 (Gerçek hastaları yakalama oranı)
* **ROC-AUC Skoru:** 0.80+ (Modelin sınıfları birbirinden ayırma gücü)

## 🛠️ 5. Kurulum ve Kullanım
Projeyi yerel makinenizde çalıştırmak için:

1. Depoyu klonlayın: `git clone https://github.com/simgee1290gnduu/PythonProject2.git`
2. Kütüphaneleri kurun: `pip install -r requirements.txt` (veya pandas, tensorflow, seaborn, matplotlib kurun).
3. Modeli çalıştırın: `python yeni.py`

---
**Geliştiren:** [Simge]  
**Eğitim:** Yapay Sinir Ağları ve Derin Öğrenme Kursu Projesi