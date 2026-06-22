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
- `src/models/panel_model.py` — `PanelVariantModel`: master'dan **bağımsız**. Varsayılan tek
  düzenlileştirilmiş LightGBM; `use_ensemble=True` ise LGBM+XGB **ham (kalibrasyonsuz)** soft-voting.
  Artefakt formatı `panel_v2` (geriye dönük uyumlu load). Ensemble yalnızca panel-başına geçiş
  (gate) OOF test-prior F1'i tek-LGBM'i geçerse kullanılır.
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

## Bu turda yapılan kararlar (2026-06-22 — F1 maksimizasyon turu)

Detay: `docs/rapor_guncellemeleri.md` Değişiklik 5–7. **Ortalama beklenen test F1: 0.641 → 0.659
(+1.81pp), ~6.6 dk.** Doğrulama metriği **`Class1_F1_TestPrior`** (OOF F1 değil).

1. **Eksiklik bayrağı özellikleri** (`preprocessing.py`): yüksek-eksiklikli (≥%30) kolonlara
   `{col}_is_missing` + `missing_concentration`. İzole +0.52pp ort. (KANSER +2.5pp).
2. **Master XGB Optuna ayarı + soft-voting ağırlık opt** (`pipeline.py`): XGB önceden default'tu;
   ağırlık OOF'ta optimize edilir, `ensemble_meta.joblib`'e yazılır, `predict.py` oradan okur.
   XGB trial = LGBM'in yarısı (10 dk bütçesi). Master'a marjinal etki.
3. **Panellere ham soft-voting ensemble + panel-başına geçiş** (`panel_model.py`, `pipeline.py`):
   KANSER +4.83pp, PAH +3.65pp (ensemble seçildi); CFTR tek-LGBM kaldı (gate kararı).
4. **KAVRAMSAL DÜZELTME:** "Panellere **kalibrasyon** → ranking iyileşir" iddiası **F1 için
   yanlış** (izotonik monotoniktir, eşik sonradan optimize → F1/AUC değişmez). Üstelik OOF-fit
   kalibratörü full-data modele uygulamak inference eşiğini bozuyordu (**CFTR all-zero hatası**).
   → Paneller **kalibrasyonsuz** ham ortalama kullanır. Master kalibrasyonu korunur (büyük veride
   transfer sağlıklı). Kazanç ensemble **çeşitliliğinden** gelir, kalibrasyondan değil.
5. **Bütçe kararı:** Faz 1.4 (Optuna 30→50) **yapılmadı** (zaten ~bütçe sınırı).

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
- ~~Panel mimarisi (ensemble)~~ → çözüldü (2026-06-22 karar #3; **ham** soft-voting, kalibrasyonsuz).
- ~~Eksiklik bayrağı~~ → eklendi (2026-06-22 karar #1).
- **Olasılık prior-kalibrasyonu:** olasılıkların da test prior'una göre kalibrasyonu (F1 için
  gerekmez; kalibrasyon kalitesi için).
- **PAH/CFTR ince ayar:** PAH-özel hiperparametre tuning; CFTR'de eksiklik-bayrağı için panel-bazlı
  özellik dışlama (ranking sağlam, ertelendi). Bkz. `docs/sonraki-adimlar.md`.
- **Raporun küçük uyumsuzlukları (açık):** çok-seed CV, Optuna 50–100 deneme (bütçe: 30+XGB 15),
  soft-voting'e kalibre LR, aykırı değer/label smoothing.

## Çalışma tarzı notları

- Kullanıcı Türkçe iletişim kuruyor; yanıtları Türkçe ver.
- Değişiklik yapınca `docs/rapor_guncellemeleri.md`'yi güncel tut.
- Rapora (`yarisma-raporu.md`) mümkün mertebe sadık kal, ama Q&A önceliklidir.
