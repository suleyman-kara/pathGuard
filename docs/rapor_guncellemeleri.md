# Rapor Güncellemeleri / Sapma Günlüğü

Bu dosya, kodun resmi yarışma raporundan (`docs/yarisma-raporu.md`) **bilinçli sapmalarını**
kayıt altına alır. Resmi rapor teslim edilmiş haliyle **dondurulmuştur** (düzenlenmez);
final aşaması için yapılan tüm değişiklikler yalnızca burada izlenir.

## Doküman önceliği

Bir çelişki olduğunda öncelik sırası: **`soru-cevap.md` > `yarisma-raporu.md` >
`yarisma-sartnamesi.md`**. Yani Q&A (soru-cevap) en yüksek otoritedir; rapor ile çelişirse
Q&A esas alınır.

## Final aşaması ana hedefi

Modelin **hiç görmediği** test verisinde **patojenik sınıf (class 1) F1 skorunu maksimize
etmek**. Yarışma sıralama metriği yalnızca TP/FP/FN üzerinden hesaplanan F1'dir
(`yarisma-sartnamesi.md`) ve patojenik (label 1) sınıfa odaklıdır (`soru-cevap.md`).

---

## Değişiklik 1 — Panel modelleri master modelinden bağımsızlaştırıldı

**Tarih:** 2026-06-21

**Ne yapıldı?**
- `src/models/panel_model.py`: `PanelMetaLearner` sınıfı `PanelVariantModel` olarak yeniden
  yazıldı. Master ensemble'ın yumuşak tahminini (`master_soft_prediction`) panel özelliklerine
  ekleyen `_augment_features` mekanizması tamamen kaldırıldı. Panel modeli artık yalnızca
  kendi panel verisi üzerinde, tek ve sıkı düzenlileştirilmiş bir LightGBM olarak eğitiliyor.
- `src/pipeline.py`: Panel CV ve final eğitim döngüleri master ensemble'a referans vermiyor.
  `run_panel_pipelines` artık master MODELİNE değil, yalnızca paylaşılan ön-işleme
  encoder'ına bağlı.
- `scripts/predict.py`: Panel tahmini master ağırlıklarını yüklemeden çalışıyor; panel eşiği
  `ensemble_meta.joblib` içindeki `panel_thresholds`'tan okunuyor.

**Neden?**
- **Yanıltıcı F1 (sızıntı):** Master model TÜM master setiyle eğitiliyor; paneller ise
  master'ın alt kümesi (örtüşme: KANSER `246/388`, PAH `255/372`, CFTR `77/111` ≈ %63–69).
  Bu yüzden `master_soft_prediction`, panel doğrulama satırlarında **ezberlenmiş/sızdırılmış
  etiket** taşıyordu. Sonuç: panel OOF F1 (0.92–0.95) gerçek dışı yüksek; görülmemiş test
  verisinde çökerdi.
- **Q&A şartı:** `soru-cevap.md` her panel için **ayrı/bağımsız** model şart koşuyor
  ("four separate models", "each panel requires its own model rather than a single model").
  Meta-learning yaklaşımı panelleri master'a bağladığı için bu şartla çelişiyordu → Q&A
  önceliği gereği bağımsız modele geçildi.

**Rapordaki ilgili bölüm (artık geçerli değil):**
- "Özgünlük" → *Panel-aware meta-learning* maddesi (genel modelin soft prediction'larının
  panel modeline meta-özellik olarak aktarılması). **Bu özgünlük iddiasından vazgeçilmiştir.**
- "Öğrenme Süreci ve Teknik Evrim" → *Panel bazlı genelleme sorunu* maddesindeki iki aşamalı
  (master → panel meta-özellik) yaklaşım. Artık paneller bağımsız eğitiliyor.

**Etki:** Panel OOF F1 değerlerinin **düşmesi beklenir ve istenir** — bu, dürüst/gerçekçi
skoru ve görülmemiş test performansını yansıtır.

**Not (encoder):** Ön-işleme encoder'ı hâlâ master setinde fit ediliyor. Bu yalnızca özellik
istatistiğidir (medyan/scaler/frekans) ve etiket sızıntısı yaratmaz; küçük panellerde
imputasyonu daha sağlam kılar. Tam bağımsız (per-panel) encoder, ileride değerlendirilebilecek
bir takip konusudur.

---

## Değişiklik 2 — Klinik recall ≥ 0.90 eşik kısıtı kaldırıldı

**Tarih:** 2026-06-21

**Ne yapıldı?**
- `src/config.py`: `CLINICAL_RECALL_TARGET = 0.90` → `0.0`.
- `src/models/ensemble.py` (`optimize_decision_threshold`): Fonksiyon değişmeden çalışır;
  `recall_target = 0.0` olduğu için her eşik kısıtı sağlar ve doğrudan **Class 1 F1'i
  maksimize eden** eşik seçilir. Docstring/yorumlar güncellendi.

**Neden?**
- Yarışma sıralama metriği yalnızca F1 (patojenik sınıf) olduğu için, recall ≥ 0.90 kısıtı
  ulaşılabilir en yüksek Class 1 F1'i sınırlıyordu. Kısıtın kaldırılması eşik seçimini yarışma
  metriğiyle birebir hizalar.

**Rapordaki ilgili bölüm (artık geçerli değil):**
- "Performans Metrikleri" → "Sensitivite ≥ 0.90 kısıtı altında Özgüllük maksimizasyonu".
- "Sınıf Dengesi ve Risk Perspektifi" → recall için alt sınır belirleyerek eşik optimizasyonu.
- "Öğrenme Süreci" → "Recall ≥ 0.90 kısıtı altında F1 maksimizasyonu".
- "Özgünlük" → *Klinik risk odaklı değerlendirme çerçevesi* (yanlış negatif maliyetine dayalı
  eşik gerekçesi). **Bu özgünlük iddiasından vazgeçilmiştir.**

**Etki:** Eşikler artık yalnızca Class 1 F1'i maksimize eder. (recall_target parametresi
korunmuştur; ileride klinik bir taban istenirse > 0 verilerek yeniden etkinleştirilebilir.)

---

## Değişiklik 3 — Şartnamedeki veri dağılımı düzeltmesi (kayıt)

**Ne?** `yarisma-sartnamesi.md` patojenik/benign sayılarını (2909 patojenik; ~1381+1500
benign) verir; bu dengeli görünüm **hatalıdır**. Q&A (`soru-cevap.md`) gerçek dağılımı şöyle
tanımlar:
- **Eğitim seti:** ~%80 patojenik / ~%20 benign.
- **Test seti:** ~%20 patojenik / ~%80 benign (tersine çevrilmiş, gerçekçi ve zorlayıcı).

**Neden kayıt altına alındı?** Doküman önceliği gereği Q&A dağılımı esas alınır. Bu asimetri,
eğitimde seçilen kararın test dağılımına göre yanlı olabileceği anlamına gelir
(prior/dağılım kayması) — bilinen bir takip konusudur (aşağıya bakınız).

---

## Değişiklik 4 — Test prior'una göre eşik seçimi (Özgünlük #2'nin tamamlanması)

**Tarih:** 2026-06-21

**Ne yapıldı?**
- `src/config.py`: `TEST_PATHOGENIC_PRIOR = 0.20` eklendi (Q&A: test ~%20 patojenik).
- `src/models/ensemble.py` (`optimize_decision_threshold`): Eşik artık eğitim/OOF dağılımında
  değil, **test prior'u altında** hesaplanan Class 1 F1'i (F1_π) maksimize ediyor. Yeni yardımcı
  `prior_adjusted_class1_f1`: recall (TPR) ve FPR prior'dan bağımsız olduğu için precision'ı
  hedef prior π altında yeniden kurar: `precision_π = π·TPR / (π·TPR + (1−π)·FPR)`.
- `src/evaluation.py` (`calculate_metrics`): Her rapora `Class1_F1_TestPrior` ve
  `Precision_TestPrior` alanları eklendi — bunlar **görülmemiş test setinde beklenen** Class 1
  F1'i öngörür. OOF (eğitim dağılımı) metrikleri referans olarak korunur.
- `src/pipeline.py`: Master ensemble seçimi (soft-voting vs stacking) de `Class1_F1_TestPrior`
  üzerinden yapılıyor; loglar hem OOF hem beklenen-test F1'ini gösteriyor.

**Neden?**
- Yarışma metriği test dağılımında (%20 patojenik) ölçülecek; oysa eşik eğitim dağılımında
  (%80) seçiliyordu. Bu, test setinde precision'ı çökertiyordu (Q&A dağılım şartı + raporun
  satır 98'de bizzat uyardığı "karar sınırının test dağılımına göre yanlı kalması").
- Raporda **koruduğumuz** Özgünlük #2 ("Asimetrik dağılıma özel kalibrasyon + eşik pipeline'ı")
  ancak bu adımla fiilen teslim edilmiş olur.

**Etki (30 trial, ölçülen):** Eşik yükseldi (master 0.51 → 0.68); OOF F1 düştü ama **beklenen
test F1 belirgin arttı**:

| Model | OOF Class1 F1 | Beklenen Test F1 (önce ~) | Beklenen Test F1 (sonra) |
| --- | ---: | ---: | ---: |
| Master | 0.792 | ~0.50 | **0.595** |
| KANSER | 0.789 | ~0.56 | **0.645** |
| PAH | 0.788 | ~0.38 | **0.497** |
| CFTR | 0.741 | ~0.54 | **0.741** |

PAH en zayıf (ROC-AUC 0.76 → sınıf ayrımı düşük); CFTR en iyi (specificity 1.0). Bu, eşiğin
gerçek değerlendirme dağılımına hizalanmasının doğrudan kazanımıdır.

**Not:** Olasılıkların kendisinin prior'a göre yeniden kalibrasyonu (reliability diagram kalitesi
için) opsiyonel bir sonraki adım olarak bırakılmıştır; F1 sıralaması için eşik düzeltmesi yeterli.

---

## Değişiklik 5 — Eksiklik bayrağı (missing indicator) özellikleri

**Tarih:** 2026-06-22

**Ne yapıldı?**
- `src/preprocessing.py` (`VariantFeatureEncoder`): İmputasyon **öncesi** NaN maskesinden
  yüksek-eksiklikli (≥%30, yani `mid_missing_cols + dropped_cols`) sürekli kolonlar için
  `{col}_is_missing` ikili bayrak özellikleri ve satır-düzeyi `missing_concentration` özelliği
  eklendi. Düşük-eksiklikli kolonlara bayrak eklenmez (neredeyse sabit → gürültü). Düşürülen
  (>%60) kolonların değerleri atılsa da **eksiklik sinyali korunur**.

**Neden?** Rapor (özellik mühendisliği) "eksikliğin kendisi bilgilendiricidir; ikili eksiklik
bayrağı eklenir" diyor — kod bunu içermiyordu. Eksiklik bayrağı sıralamayı (ranking) değiştirir
ve AUC/F1'i artırabilir; aynı zamanda rapor-kod boşluğunu kapatır.

**Etki (izole, 30 trial):** Master +0.85pp, KANSER +2.5pp, PAH +0.4pp, CFTR −1.67pp;
**ortalama beklenen test F1 +0.52pp**. CFTR'deki düşüş gerçek bir ayrım kaybı DEĞİLDİR
(PR-AUC 0.97, specificity 1.0 → precision sabit; F1 recall-bağımlı ve 111 örnekte 99-noktalı
eşik ızgarasının granülaritesinden kaynaklı). Net pozitif olduğu için global tutuldu.

---

## Değişiklik 6 — Master XGBoost Optuna ayarı + soft-voting ağırlık optimizasyonu

**Tarih:** 2026-06-22

**Ne yapıldı?**
- `src/pipeline.py`: XGBoost önceden **default** parametrelerle eğitiliyordu (`XGB_PARAM_SPACE`
  config'de tanımlı ama kullanılmıyordu). `_optimize_xgb_hyperparams` eklendi; XGB artık Optuna
  ile ayarlanıyor (LGBM'in yarısı kadar trial — tam trial master'a ~0.1pp katarken süreyi ~3 dk
  artırıyordu; 10 dk bütçesi için sınırlandı).
- `_select_master_ensemble`: Sabit `0.6/0.4` soft-voting ağırlıkları yerine OOF üzerinde
  test-prior F1'i maksimize eden ağırlık grid araması (11 nokta). Seçilen ağırlık
  `ensemble_meta.joblib`'e yazılıyor; `scripts/predict.py` artık ağırlığı meta'dan okuyor
  (eğitim/inference tutarlılığı).

**Neden?** Rapor LGBM+XGB+(LR) ağırlıklı soft-voting iddia ediyor; XGB ayarsız ve ağırlık sabitti.
Ayarlı XGB + optimize ağırlık ensemble çeşitliliğini/ranking'i iyileştirir.

**Etki:** Master'a marjinal (+0.1–0.2pp; Optuna stokastikliği nedeniyle koşular arası ~0.2pp
oynar). Stacking sıkça seçildiğinden ağırlık-opt bazı koşularda master'ı etkilemez (soft-voting
adayı seçilmezse). Düşük getiri/maliyet — bu yüzden XGB trial'ı sınırlandı.

---

## Değişiklik 7 — Panellere ham (kalibrasyonsuz) soft-voting ensemble + panel-başına geçiş

**Tarih:** 2026-06-22

**Ne yapıldı?**
- `src/models/panel_model.py`: `PanelVariantModel` artık `use_ensemble` modunu destekliyor:
  LGBM + (sıkı düzenlileştirilmiş) XGBoost, **ham 0.5/0.5 ağırlıklı ortalama** (soft-voting).
  Artefakt formatı `panel_v2` (her şey tek dosyada kapsüllü; geriye dönük uyumlu `load`).
  `scripts/predict.py` değişmeden çalışır.
- `src/pipeline.py` (`run_panel_pipelines`): OOF döngüsünde hem LGBM hem XGB eğitilir; **panel-başına
  geçiş (gate)** ensemble'ı yalnızca OOF test-prior F1 tek-LGBM'i geçerse kullanır.

**Neden kalibrasyon YOK (önemli düzeltme):** İlk uygulama panel olasılıklarını izotonik kalibre
edip ortalıyordu. Ancak OOF üzerinde fit edilen kalibratörü full-data modelin olasılıklarına
uygulamak, OOF'ta seçilen eşiğin **inference'a transfer olmamasına** yol açtı: CFTR'de inference
olasılıkları (maks 0.75) eşiğin (0.82) altında kalıp **tüm tahminleri 0** yaptı. Ham ortalama
tek-model gibi sorunsuz transfer eder ve çeşitlilik kazancını (ensemble'ın asıl faydası) korur.
İlke web kanıtıyla uyumlu: izotonik kalibrasyon monotoniktir → eşik sonradan optimize edildiğinde
F1'i değiştirmez; kalibrasyonun tek rolü ortalama öncesi ölçek hizalamadır ve burada zararı
faydasından büyüktü.

**Etki (gate kararları, 30 trial / cv-repeats 10):**
- KANSER → ENSEMBLE: 0.6655 → **0.7138** (+4.83pp; ROC-AUC 0.887→0.916)
- PAH → ENSEMBLE: 0.5236 → **0.5601** (+3.65pp)
- CFTR → SINGLE-LGBM (gate ensemble'ı reddetti; ham ensemble küçük recall-bağımlı panelde fayda
  etmedi): 0.7671 (Değişiklik 5 kaynaklı, baseline 0.7838'in −1.67pp altında)

---

## Toplam etki (baseline → güncel, beklenen test class 1 F1)

| Model | Baseline | Güncel | Δ | Final model tipi |
| --- | ---: | ---: | ---: | --- |
| Master | 0.5896 | 0.5939 | +0.43pp | kalibre LGBM+XGB soft-voting |
| KANSER | 0.6655 | **0.7138** | +4.83pp | LGBM+XGB ham ensemble |
| PAH | 0.5236 | **0.5601** | +3.65pp | LGBM+XGB ham ensemble |
| CFTR | 0.7838 | 0.7671 | −1.67pp | tek LGBM (gate kararı) |
| **Ortalama** | **0.6406** | **0.6587** | **+1.81pp** | |

Çalışma süresi ~6.6 dk (10 dk bütçesi içinde). Tüm 4 modelin inference dağılımı sağlıklı (CFTR
all-zero hatası düzeltildi). Bu, **dürüst ve dağıtılabilir** bir iyileşmedir.

---

## Korunan özgünlükler

Aşağıdaki rapor özgünlükleri değişmeden korunmaktadır:
1. **Kolon-isimsiz SHAP özellik gruplaması:** Kolon adı olmayan veride özelliklerin SHAP
   tabanlı kümeleme ile biyolojik gruplara atanması.
2. **Asimetrik dağılıma özel kalibrasyon + eşik pipeline'ı:** Isotonic kalibrasyon ve eşik
   optimizasyonunun birlikte uygulandığı değerlendirme düzeni.

---

## Bilinen takip konuları

- ~~Eğitim/test prior kayması~~ → **Değişiklik 4 ile çözüldü** (test prior'una göre eşik).
- ~~Panel mimarisi (ensemble)~~ → **Değişiklik 7 ile çözüldü** (ham soft-voting + panel-başına geçiş).
- ~~Eksiklik bayrağı / XGB ayarı~~ → **Değişiklik 5 & 6 ile eklendi.**
- **KAVRAMSAL DÜZELTME — "panellere kalibrasyon":** Önceki yol haritası "panellere kalibrasyon →
  ranking iyileşir" diyordu; bu **F1 için yanlıştır.** İzotonik kalibrasyon monotoniktir; eşik
  zaten taranarak optimize edildiğinden örnek sıralamasını (ve dolayısıyla AUC/F1'i) değiştirmez.
  Kalibrasyonun tek faydası iki modeli ortalamadan önce ölçek hizalamaktır; ancak full-data modele
  OOF-fit kalibratör uygulamak inference eşiğini bozuyordu (CFTR all-zero). Bu yüzden paneller
  **ham (kalibrasyonsuz) soft-voting** kullanır. (Master kalibrasyonu korunur — büyük veride
  inference transferi sağlıklı.)
- **Olasılık prior-kalibrasyonu:** Eşiğe ek olarak olasılık çıktılarının da test prior'una göre
  yeniden kalibrasyonu (reliability diagram / Brier kalitesi için). F1 için gerekli değil.
- **PAH'ın zayıf ayrımı:** Ensemble ile +3.65pp iyileşti (ROC-AUC ~0.80). Daha ileri panel-özel
  hiperparametre tuning'i (paneller şu an sabit param) opsiyonel — küçük sette overfit riski.
- **CFTR'nin missing-flag hassasiyeti:** Eksiklik bayrakları CFTR'de recall granülaritesi nedeniyle
  ~−1.7pp; panel-bazlı özellik dışlama ile geri kazanılabilir (belirsiz, +0.4pp; ranking sağlam
  olduğundan ertelendi).
- **Raporun küçük uyumsuzlukları (açık):** çok-seed CV, Optuna 50–100 deneme (bütçe nedeniyle 30 +
  XGB 15), soft-voting'e kalibre LR, aykırı değer/label smoothing.
