# PathGuard: Missense Genetik Varyant Sınıflandırma Sistemi

PathGuard, missense genetik varyantların patojenik/benign sınıflandırması için geliştirilmiş, rapor uyumlu bir makine öğrenmesi pipeline'ıdır. Sistem; veri kalite kontrolü, sızıntı farkındalığı, LightGBM/XGBoost ensemble, Logistic Regression stacking, olasılık kalibrasyonu, klinik recall odaklı eşik seçimi, panel bazlı OOF validasyon ve SHAP açıklanabilirliği içerir.

## Proje Yapısı

- `src/`: Veri yükleme, ön işleme, model wrapper'ları, pipeline, değerlendirme ve açıklanabilirlik kodları.
- `src/models/`: LightGBM, XGBoost, ensemble ve panel meta-learner bileşenleri.
- `scripts/`: Eğitim ve tahmin CLI araçları.
- `data/raw/`: Yarışma eğitim CSV dosyaları.
- `models/`: Eğitim sonrası üretilen model artifact'leri. Git'e alınmaz.
- `outputs/`: Metrikler, veri kalite raporları, hata analizleri, feature importance, SHAP ve grafik çıktıları. Git'e alınmaz.
- `docs/`: Yarışma şartnamesi ve proje raporu.

## Kurulum

Python sanal ortam ile:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda ile:

```bash
conda env create -f environment.yml
conda activate pathguard
```

macOS kullanıyorsanız LightGBM için `libomp` gerekebilir:

```bash
brew install libomp
```

## Eğitim

Tam eğitim:

```bash
./.venv/bin/python scripts/train.py --trials 30
```

Hızlı smoke test:

```bash
./.venv/bin/python scripts/train.py --trials 1 --cv-repeats 1 --skip-shap
```

CLI seçenekleri:

- `--trials`: Master LightGBM için Optuna deneme sayısı.
- `--cv-repeats`: Küçük panel OOF validasyonu için repeated stratified CV tekrar sayısı. Varsayılan `10`.
- `--calibration-mode`: `oof` veya `holdout`. Varsayılan `oof`.
- `--enable-stacking` / `--no-enable-stacking`: Logistic Regression stacking katmanını açar/kapatır. Varsayılan açık.
- `--skip-shap`: SHAP üretimini atlar, hızlı doğrulama için kullanılır.

## Tahmin

Master model ile:

```bash
./.venv/bin/python scripts/predict.py data/raw/TEST_FILE.csv --panel MASTER --output submission.csv
```

Panel modeli ile:

```bash
./.venv/bin/python scripts/predict.py data/raw/CFTR_TEST.csv --panel CFTR --output cftr_submission.csv
```

Girdi CSV'si eğitimde öğrenilen feature şemasıyla uyumlu olmalıdır. Eksik veya fazla kolon varsa tahmin script'i açık hata mesajı verir.

## Üretilen Çıktılar

Eğitim sonunda başlıca çıktılar şunlardır:

- `outputs/data_quality_report.json`: Satır sayıları, sınıf dağılımları, missing oranları, duplicate/çelişkili duplicate bilgisi ve master-panel örtüşmeleri.
- `outputs/experiment_log.jsonl`: Eğitim olayları, Optuna sonucu ve metrik kayıtları.
- `outputs/master_metrics.json`: Master OOF metrikleri.
- `outputs/panel_*_metrics.json`: Panel OOF metrikleri.
- `outputs/master_ensemble_comparison.json`: Soft voting ve logistic stacking karşılaştırması.
- `outputs/error_analysis_*.csv`: FP/FN örnekleri.
- `outputs/feature_importance_*.csv`: Gain ve permutation importance raporları.
- `outputs/*_pr_curve.png`, `outputs/*_reliability.png`: PR ve kalibrasyon grafikleri.
- `outputs/*_shap_summary.png`, `outputs/*_shap_waterfall_*.png`: Açıklanabilirlik grafikleri.
- `models/*.joblib`: Encoder, base modeller, kalibratörler, seçilen ensemble meta bilgisi ve panel modelleri.

## Sistem Akışı

1. **Veri kalite kontrolü:** Eğitim setleri yüklenir; boş veri, eksik hedef, tek sınıf, duplicate ve çelişkili duplicate kontrolleri yapılır.
2. **Sızıntı farkındalığı:** Panel `Variant_ID` değerlerinin master set ile örtüşmesi raporlanır. Örtüşme varsa panel metrikleri `Leakage_Aware=true` olarak işaretlenir.
3. **Ön işleme:** Kategorik kolonlar missing/rare/unseen güvenli encoding ve frequency encoding'den geçer. Eksik değer oranları raporlanır; linear/stacking yolu için imputasyon ve ölçekleme uygulanır; tree modeller için NaN desteği korunur.
4. **Master model:** LightGBM için Optuna ile PR-AUC optimizasyonu yapılır. LightGBM ve XGBoost OOF tahminleri kalibre edilir.
5. **Ensemble seçimi:** Soft voting ve Logistic Regression stacking OOF metrikleriyle karşılaştırılır; Macro F1'i yüksek olan seçilir.
6. **Eşik seçimi:** Klinik recall hedefi gözetilerek karar eşiği optimize edilir.
7. **Panel modelleri:** KANSER, PAH ve CFTR panellerinde master soft prediction meta-feature olarak kullanılır. Panel metrikleri final train-set üzerinden değil, repeated stratified OOF tahminlerden hesaplanır.
8. **Analiz çıktıları:** Metrikler, hata analizi, feature importance, reliability diagram ve SHAP grafikleri üretilir.

## Son Eğitim Özeti

`./.venv/bin/python scripts/train.py --trials 30` çalıştırması sonucunda seçilen master ensemble `logistic_stacking` olmuştur.

| Model | Macro F1 | PR-AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Master OOF | 0.7929 | 0.9356 | 0.9002 | 0.6775 |
| KANSER OOF | 0.8581 | 0.9455 | 0.9283 | 0.7750 |
| PAH OOF | 0.7101 | 0.9124 | 0.9349 | 0.4500 |
| CFTR OOF | 0.8825 | 0.9808 | 0.9556 | 0.8095 |

Panel metrikleri leakage-aware yorumlanmalıdır. Son eğitimde master-panel örtüşmeleri KANSER için `246/388`, PAH için `255/372`, CFTR için `77/111` olarak raporlanmıştır.

## Notlar

- Modeller ve çıktılar bilinçli olarak Git dışında tutulur.
- Dış veri kaynaklarından etiket sorgulama yapılmaz; yarışma kısıtına uygun biçimde yalnız verilen varyant profilleri kullanılır.
- Panel sonuçlarında yüksek recall klinik önceliğe uygundur; düşük specificity görülen panellerde panel bazlı eşik optimizasyonu ayrıca değerlendirilebilir.

---

**Takım:** MEN-INA | **Proje:** PathGuard
