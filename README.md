# PathGuard: Missense Genetik Varyant Sınıflandırma Sistemi

PathGuard, missense genetik varyantların patojenite (hastalık yapıcı) durumlarını tahmin etmek için geliştirilmiş, yüksek performanslı bir makine öğrenmesi pipeline'ıdır. Sistem, özellikle klinik Recall (duyarlılık) hedeflerine ve gen paneli bazlı transfer öğrenmeye odaklanmıştır.

## 📂 Proje Yapısı

- `src/`: Çekirdek kaynak kodlar (veri yükleme, ön işleme, modeller).
- `scripts/`: Eğitim ve tahmin işlemlerini başlatan CLI araçları.
- `models/`: Eğitilmiş model ağırlıkları ve encoder nesneleri.
- `outputs/`: Eğitim sonrası oluşan metrik raporları, SHAP grafikleri ve PR eğrileri.
- `data/raw/`: Ham veri setleri (CSV).
- `data/processed/`: Ara işleme dosyaları için ayrılmış klasör (Şu an sistem hızı için işlemler bellekte yapılmaktadır).

## 🚀 Kurulum

Sistemi çalıştırmak için Python 3.9+ ve gerekli kütüphanelerin yüklü olması gerekir:

```bash
# Sanal ortam oluşturma ve bağımlılıkları yükleme
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS kullanıyorsanız LightGBM için libomp gereklidir:
# brew install libomp
```

## 🏋️‍♂️ Eğitim (Training)

Tüm sistemi (Master model ve Gen Panelleri) eğitmek için:

```bash
./.venv/bin/python scripts/train.py --trials 30
```

- `--trials`: Optuna ile yapılacak hiperparametre arama sayısıdır. Hızlı test için `2`, üretim için `30+` önerilir.
- Eğitim sonunda modeller `models/` klasörüne, performans raporları ise `outputs/` klasörüne kaydedilir.

## 🔮 Tahmin (Inference / Prediction)

Eğitilmiş modelleri kullanarak yeni veriler üzerinde tahmin yapmak için:

```bash
# Genel Master model ile tahmin
./.venv/bin/python scripts/predict.py data/raw/TEST_FILE.csv --panel MASTER

# Spesifik bir panel (örn: CFTR) için tahmin
./.venv/bin/python scripts/predict.py data/raw/CFTR_TEST.csv --panel CFTR
```

## 🧠 Sistem Nasıl Çalışıyor? (Basitçe)

1.  **Veri Temizleme:** Tekrarlanan veya çelişkili etiketli veriler temizlenir.
2.  **Özellik Mühendisliği:** İsimsiz kolonlar frekans kodlamasından geçer, eksik değerler doldurulur.
3.  **Hibrit Modelleme:** LightGBM ve XGBoost modelleri beraber eğitilir (Ensemble).
4.  **Kalibrasyon:** Modelin verdiği olasılık puanları, klinik gerçekliğe uygun hale getirilmesi için Isotonic Regression ile düzeltilir.
5.  **Eşik Optimizasyonu:** "Bir hastayı bile kaçırmama" (Recall >= 0.90) kısıtı altında en iyi başarıyı veren karar eşiği otomatik bulunur.
6.  **Transfer Öğrenme:** Küçük gen panelleri (CFTR vb.), Master modelden gelen bilgiyi kullanarak özelleşmiş alt modeller oluşturur.

---
**Takım:** MEN-INA | **Proje:** PathGuard
