import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# Grafik stili
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("🏥 DİYABET TAHMİN SİSTEMİ - GELİŞMİŞ YAPAY SİNİR AĞI")
print("=" * 70)

# 1. VERİ YÜKLEME VE KEŞİF
try:
    df = pd.read_csv('diabetes.csv')
    print(f"\n✅ Veri başarıyla yüklendi!")
    print(f"📊 Toplam Kayıt: {len(df)}")
    print(f"📋 Özellik Sayısı: {df.shape[1] - 1}")
    print(f"\n💊 Diyabet Dağılımı:")
    print(df['Outcome'].value_counts())
    print(f"   Diyabetli: %{(df['Outcome'].sum() / len(df)) * 100:.1f}")

    # 2. VERİ TEMİZLEME VE ÖN İŞLEME
    print("\n🔧 Veri temizleme başlıyor...")

    # Sıfır değerleri kontrol et
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = (df[cols_to_fix] == 0).sum()
    print(f"\n⚠️  Sıfır değer tespit edildi:")
    for col, count in zero_counts.items():
        if count > 0:
            print(f"   {col}: {count} adet")

    # Sıfırları medyan ile doldur
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Yeni özellikler oluştur (Feature Engineering)
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    print("✅ Veri temizleme tamamlandı!")

    # 3. VERİYİ AYIRMA
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📂 Veri bölünmesi:")
    print(f"   Eğitim Seti: {len(X_train)} kayıt")
    print(f"   Test Seti: {len(X_test)} kayıt")

    # 4. ÖLÇEKLENDİRME
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. GELİŞMİŞ YSA MİMARİSİ
    print("\n🧠 Yapay Sinir Ağı oluşturuluyor...")

    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Optimizer
    optimizer = Adam(learning_rate=0.001)

    # Model derleme
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'Precision', 'Recall']
    )

    print("\n📐 Model Mimarisi:")
    model.summary()

    # 6. CALLBACK'LER (Otomatik İyileştirme)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # 7. MODELİ EĞİTME
    print("\n🚀 Eğitim başlıyor...\n")

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 8. SONUÇLARI DEĞERLENDİRME
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANS SONUÇLARI")
    print("=" * 70)

    # Test seti değerlendirmesi
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test_scaled, y_test, verbose=0
    )

    print(f"\n🎯 Test Seti Metrikleri:")
    print(f"   Doğruluk (Accuracy): %{test_accuracy * 100:.2f}")
    print(f"   Kesinlik (Precision): %{test_precision * 100:.2f}")
    print(f"   Duyarlılık (Recall): %{test_recall * 100:.2f}")
    print(f"   F1-Score: %{(2 * test_precision * test_recall / (test_precision + test_recall)) * 100:.2f}")

    # Tahminler
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Detaylı Rapor
    print("\n📈 Detaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Sağlıklı', 'Diyabetli']))

    # 9. GRAFİKLER
    print("\n📊 Grafikler oluşturuluyor...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Eğitim-Validasyon Kaybı
    axes[0, 0].plot(history.history['loss'], label='Eğitim Kaybı', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validasyon Kaybı', linewidth=2)
    axes[0, 0].set_title('Model Kaybı (Loss)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Eğitim-Validasyon Doğruluğu
    axes[0, 1].plot(history.history['accuracy'], label='Eğitim Doğruluğu', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validasyon Doğruluğu', linewidth=2)
    axes[0, 1].set_title('Model Doğruluğu (Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Sağlıklı', 'Diyabetli'],
                yticklabels=['Sağlıklı', 'Diyabetli'])
    axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Gerçek Değer')
    axes[1, 0].set_xlabel('Tahmin')

    # ROC Eğrisi
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Eğrisi', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('diabetes_model_results.png', dpi=300, bbox_inches='tight')
    print("✅ Grafikler 'diabetes_model_results.png' olarak kaydedildi!")

    # 10. MODEL KAYDETME
    model.save('diabetes_model.h5')
    print("\n💾 Model 'diabetes_model.h5' olarak kaydedildi!")

    print("\n" + "=" * 70)
    print("✨ EĞİTİM BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)

    # Örnek tahmin
    print("\n🔮 Örnek Tahmin:")
    sample_data = X_test_scaled[0:1]
    prediction = model.predict(sample_data, verbose=0)[0][0]
    print(f"   Diyabet Olasılığı: %{prediction * 100:.2f}")
    print(f"   Sonuç: {'DİYABETLİ' if prediction > 0.5 else 'SAĞLIKLI'}")

except FileNotFoundError:
    print("❌ HATA: 'diabetes.csv' dosyası bulunamadı!")
    print("   Lütfen dosyanın aynı dizinde olduğundan emin olun.")
except Exception as e:
    print(f"❌ Beklenmedik hata: {e}")
    import traceback

    traceback.print_exc()
    from tensorflow.keras.utils import plot_model

    # Modelin şemasını bir resim dosyası olarak kaydeder
    plot_model(model, to_file='model_mimari_semasi.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB',  # Yukarıdan aşağıya akış
               expand_nested=True,
               dpi=96)

    print("\n🖼️ Model mimari şeması 'model_mimari_semasi.png' olarak kaydedildi!")