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

## Korunan özgünlükler

Aşağıdaki rapor özgünlükleri değişmeden korunmaktadır:
1. **Kolon-isimsiz SHAP özellik gruplaması:** Kolon adı olmayan veride özelliklerin SHAP
   tabanlı kümeleme ile biyolojik gruplara atanması.
2. **Asimetrik dağılıma özel kalibrasyon + eşik pipeline'ı:** Isotonic kalibrasyon ve eşik
   optimizasyonunun birlikte uygulandığı değerlendirme düzeni.

---

## Bilinen takip konuları (bu turda YAPILMADI, sonra konuşulacak)

- **Eğitim/test prior kayması:** Eşik, eğitim/OOF dağılımında (~%80 patojenik) seçiliyor;
  oysa test ~%20 patojenik. Eşiğin test prior'una göre ayarlanması Class 1 F1'i etkileyebilir.
- **Panel mimarisi iyileştirmeleri:** Panellere kalibrasyon ve/veya LGBM+XGB ensemble
  eklenmesi (şu an her panel tek, düzenlileştirilmiş LightGBM).
