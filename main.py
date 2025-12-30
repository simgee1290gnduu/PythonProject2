import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. VERİ YÜKLEME
try:
    # Dosyalar artık yan yana olduğu için sadece ismini yazıyoruz
    df = pd.read_csv('diabetes.csv')
    print("✅ Veri başarıyla yüklendi! Eğitim başlıyor...")

    # 2. VERİ ÖN İŞLEME
    # Mantıksız 0'ları medyan ile dolduruyoruz
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 3. VERİYİ AYIRMA
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. ÖLÇEKLENDİRME
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. YAPAY SİNİR AĞI MİMARİSİ
    model = Sequential([
        Dense(12, input_dim=8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras import metrics

    # Modeli derleme kısmını bu şekilde değiştir:
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )
    # 7. SONUÇLARI TEST ETME
    results = model.evaluate(X_test, y_test, verbose=0)

    # Modelin döndürdüğü 4 metriği değişkenlere atıyoruz
    loss = results[0]
    accuracy = results[1]
    precision = results[2]
    recall = results[3]

    print(f"\n🎯 --- MODEL PERFORMANS RAPORU ---")
    print(f"Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}")
    print(f"Kesinlik (Precision): %{precision * 100:.2f}")
    print(f"Duyarlılık (Recall): %{recall * 100:.2f}")

    if recall < 0.60:
        print("⚠️ Not: Recall değerin düşük. Model hastaların çoğunu gözden kaçırıyor olabilir.")

# BURAYA DİKKAT: try bloğu burada bitmeli ve except bloğu en sola yaslı olmalı
except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")