# CLAUDE.md — PathGuard

Bu dosya, gelecekteki oturumlarda projeyi hızlıca kavramak için proje hafızasıdır.

## Proje nedir?

**PathGuard** (Takım: MEN-INA), TEKNOFEST 2026 missense genetik varyant sınıflandırma
yarışması için bir makine öğrenmesi pipeline'ıdır. Görev: varyantları **Patojenik (class 1)**
veya **Benign (class 0)** olarak sınıflandırmak (ACMG etiketleri ground truth). Yarışma
**eksternal validasyon** odaklıdır: modeller, eğitimde görülmeyen test verisinde değerlendirilir.

## EN ÖNEMLİ HEDEF

**Modelin hiç görmediği test verisinde patojenik sınıf (class 1) F1 skorunu maksimize etmek.**
Yarışma sıralama metriği yalnızca F1'dir (TP/FP/FN üzerinden, patojenik sınıfa odaklı).

## Doküman önceliği (çelişki olursa)

`docs/soru-cevap.md` (Q&A) **>** `docs/yarisma-raporu.md` (rapor) **>** `docs/yarisma-sartnamesi.md` (şartname)

- **Q&A en yüksek otoritedir.** Rapor veya şartname ile çelişirse Q&A esas alınır.
- `docs/yarisma-raporu.md` **dondurulmuştur** (teslim edilmiş resmi rapor — DÜZENLENMEZ).
- Rapordan tüm sapmalar `docs/rapor_guncellemeleri.md` içinde izlenir. Değişiklik yaparken
  oraya kayıt düş.

## Kritik yarışma bilgileri

- **4 ayrı/bağımsız model** gerekir (Q&A şartı): MASTER (Genel), KANSER, PAH, CFTR. Tek
  birleşik model değil. Skorlar birleştirilerek sıralama yapılır.
- **Veri dağılımı (Q&A doğru, ŞARTNAME HATALI):** Eğitim ~%80 patojenik / %20 benign;
  test ~%20 patojenik / %80 benign (tersine). Şartnamedeki dengeli sayılar (2909 patojenik
  vb.) yanlıştır — Q&A'yı kullan.
- Kolon isimleri ve genomik adres (kromozom/pozisyon/rsID) bilinçli gizlenmiştir; yalnızca
  sayısal varyant profilleri kullanılır, dış veri tabanı sorgusu YOK.
- Eksik değerler ile sıfır değerleri FARKLI ele alınmalı (sıfır = eksik değil).
- Çıktı: doğrudan ikili tahmin (Patojenik/Benign).

## Mimari ve kritik dosyalar

- `scripts/train.py` — eğitim CLI girişi.
- `scripts/predict.py` — tahmin CLI (`--panel MASTER|KANSER|PAH|CFTR`).
- `src/pipeline.py` — uçtan uca pipeline: `run_master_pipeline()` (Genel model: kalibre
  LGBM+XGB soft-voting/stacking ensemble) ve `run_panel_pipelines()` (her panel bağımsız).
- `src/config.py` — yollar, sabitler, arama uzayları, `CLINICAL_RECALL_TARGET`.
- `src/models/panel_model.py` — `PanelVariantModel`: master'dan **bağımsız**, tek
  düzenlileştirilmiş LightGBM.
- `src/models/ensemble.py` — kalibrasyon, soft-voting, stacking, `optimize_decision_threshold`.
- `src/models/lgbm_model.py`, `xgb_model.py`, `base_model.py` — model wrapper'ları.
- `src/preprocessing.py` — `VariantFeatureEncoder` (kategorik encoding, imputasyon, RobustScaler).
- `src/data_loader.py` — yükleme, deduplikasyon, CV split'leri, master-panel örtüşme raporu.
- `src/evaluation.py` — metrikler (Class1_F1 dahil), grafikler, hata analizi.
- `src/explainability.py` — SHAP.
- `data/raw/YARISMA_TRAIN_{MASTER,KANSER,PAH,CFTR}.csv` — eğitim verileri.

## Bu turda yapılan kararlar (2026-06-21)

1. **Paneller master'dan bağımsızlaştırıldı.** Eski `PanelMetaLearner`, master'ın
   `master_soft_prediction` çıktısını panel özelliği olarak ekliyordu. Master tüm master
   setiyle eğitildiği ve paneller onun alt kümesi olduğu için (örtüşme %63–69) bu, **sızıntılı
   ve yanıltıcı yüksek** panel OOF F1 üretiyordu. Artık `PanelVariantModel` yalnızca kendi
   panel verisiyle eğitilir. → Panel OOF F1'in **düşmesi normal ve istenen** sonuçtur (dürüst skor).
2. **Recall ≥ 0.90 klinik eşik kısıtı kaldırıldı** (`CLINICAL_RECALL_TARGET = 0.0`). Eşik
   artık doğrudan Class 1 F1'i maksimize eder. Raporun bu iki konudaki özgünlük iddialarından
   (panel-aware meta-learning + klinik risk odaklı eşik) **vazgeçildi**.
3. **Test prior'una göre eşik** (`TEST_PATHOGENIC_PRIOR = 0.20`). Eğitim ~%80 / test ~%20 ters
   dağılımı nedeniyle eşik artık test prior'u altında F1'i (`Class1_F1_TestPrior`) maksimize
   eder; `optimize_decision_threshold` + `calculate_metrics` güncellendi. Bu, raporun
   Özgünlük #2'sini (asimetrik dağılıma özel kalibrasyon+eşik) tamamlar. Beklenen test F1:
   Master 0.60, KANSER 0.65, PAH 0.50, CFTR 0.74.

## Komutlar

```powershell
# Windows — venv kur ve aktive et
py -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt

# Hızlı smoke test
py scripts/train.py --trials 1 --cv-repeats 1 --skip-shap

# Tam eğitim (~2–3 dk)
py scripts/train.py --trials 30

# Tahmin — panel tahmini master ağırlığı gerektirmez
py scripts/predict.py data/raw/YARISMA_TRAIN_CFTR.csv --panel CFTR --submission-only --output cftr.csv
```

## Sonraki adımlar / yol haritası

Önceliklendirilmiş ilerleme planı: **`docs/sonraki-adimlar.md`**. (İlk iş: `feat/independent-panels-and-recall-removal` branch'ini PR ile main'e merge etmek.)

## Bilinen takip konuları (kullanıcı "sonra konuşacağız" dedi — kendiliğinden yapma)

- ~~Eğitim/test prior kayması~~ → çözüldü (karar #3).
- **Olasılık prior-kalibrasyonu:** olasılıkların da test prior'una göre kalibrasyonu (F1 için
  gerekmez; kalibrasyon kalitesi için).
- **Panel mimarisi:** Panellere kalibrasyon ve/veya LGBM+XGB ensemble eklenmesi.
- **Raporun küçük uyumsuzlukları:** çok-seed CV, Optuna 50–100 deneme, soft-voting'e kalibre LR,
  eksiklik bayrağı/aykırı değer/label smoothing.

## Çalışma tarzı notları

- Kullanıcı Türkçe iletişim kuruyor; yanıtları Türkçe ver.
- Değişiklik yapınca `docs/rapor_guncellemeleri.md`'yi güncel tut.
- Rapora (`yarisma-raporu.md`) mümkün mertebe sadık kal, ama Q&A önceliklidir.
