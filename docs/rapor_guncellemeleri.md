# Rapor ↔ Kod Farkları — Rapora Ne Ekle / Rapordan Ne Çıkar

Bu dosyanın **tek amacı**: dondurulmuş resmi rapor (`docs/yarisma-raporu.md`) ile **şu anki
kod/implementasyon** arasındaki farkları, nedenleriyle listelemek. Raporu güncelleyen kişi
buradan **(A) rapordan neyi çıkaracağını/düzelteceğini** ve **(B) rapora neyi ekleyeceğini** görür.

> Rapor bir **plan/öneri** raporudur (gelecek zamanlı). Bazı planlar uygulanmadı (→ A), bazı
> uygulananlar raporda yok (→ B). Doküman önceliği: `soru-cevap.md` (Q&A) > rapor > şartname.

---

## A) RAPORDAN ÇIKAR / DÜZELT — rapor iddia ediyor, kod yapmıyor

| # | Rapordaki iddia (bölüm) | Gerçek durum (kod) | Aksiyon — neden |
|---|---|---|---|
| A1 | **Panel-aware meta-learning**: genel modelin soft-prediction'ları panele meta-özellik (Özgünlük #3; "Öğrenme Süreci → Panel bazlı genelleme"; iki aşamalı transfer) | Paneller **tamamen bağımsız**, master modeline hiç bağlı değil | **ÇIKAR.** Master tüm master setiyle eğitiliyor, paneller onun alt kümesi (örtüşme %63–69) → master tahmini panel doğrulama satırlarına **etiket sızdırıyordu** (sahte-yüksek OOF F1). Q&A da "4 ayrı/bağımsız model" şart koşuyor. |
| A2 | **Klinik recall ≥ 0.90 kısıtı / klinik-risk odaklı eşik** (Özgünlük #4; Sınıf Dengesi; "Sensitivite ≥0.90 altında Özgüllük maks"; Öğrenme Süreci; Parametre Seçimi) | Recall kısıtı **kaldırıldı** (`CLINICAL_RECALL_TARGET=0.0`); eşik yalnız class-1 F1 maksimize eder | **ÇIKAR.** Yarışma yalnız class-1 F1 ile sıralıyor; recall tabanı ulaşılabilir en yüksek F1'i kısıtlıyordu. |
| A3 | Eşik **F1-macro**'yu maksimize eder (Performans Metrikleri; Parametre Seçimi) | Eşik **test-prior class-1 F1**'i maksimize eder (macro değil) | **DÜZELT.** Metrik patojenik (class-1) F1; macro değil. (Yöntem için bkz. B1) |
| A4 | **3 farklı seed** ile tekrarlı CV (Deney Protokolü) | Tek seed (42); paneller Repeated 5-Fold ×10 ama yine tek seed | **ÇIKAR/DÜZELT.** Uygulanmadı (F1 etkisi ~0). |
| A5 | Optuna **50–100 deneme** (Parametre Seçimi) | **30 (LGBM) + 15 (XGB)** | **DÜZELT → "30 (+15 XGB)".** 10 dk bütçesi; tam trial master'a yalnız ~0.1pp katıyordu. |
| A6 | **Aykırı değer taraması**: Z>5 inceleme, %1–%99 persentil dışı inceleme (Aykırı değer işlemi; Veri Kalitesi) | Yalnız **RobustScaler** (uç etkiyi azaltır); açık Z>5/persentil taraması **yok** | **ÇIKAR/yumuşat.** Sadece RobustScaler kaldı. |
| A7 | **Label Smoothing / düşük-ağırlıkla kullanma** (gürültülü etiket) (Veri Kalitesi) | **Yok.** Çelişkili-etiketli kopyalar dedup ile **silinir** (label smoothing değil) | **DÜZELT.** Label smoothing'i çıkar; "çelişkili duplikasyonlar silinir"i koru. (Kolon adları gizli → "benign ama yüksek patojenite skoru" tespiti zaten imkânsız.) |
| A8 | **Özellik seçimi / boyut indirgeme**: düşük katkılı özellikleri at (Boyut indirgeme) | Önem (gain + permütasyon) **hesaplanıp kaydediliyor** ama özellik **düşürülmüyor** | **DÜZELT.** "Önem raporlanır" de; "boyut atılır/özellik seçilir" iddiasını çıkar. |
| A9 | Ensemble: LGBM + XGB **+ kalibre LR** soft-voting (3 üye) (Ensemble Katmanı) | Soft-voting yalnız **LGBM+XGB** (2 üye); **LR yalnız stacking meta-modeli** (soft-voting'in alternatifi) | **DÜZELT.** LR'yi "stacker (alternatif ensemble)" olarak tanımla; soft-voting üyesi değil. |
| A10 | Kalibrasyon **eğitimin %20'lik holdout'unda** (Parametre Seçimi) | Varsayılan kalibrasyon **OOF tahminleri** üzerinde (`calibration_mode="oof"`); %20 holdout opsiyonel, varsayılan değil | **DÜZELT.** "OOF üzerinde izotonik kalibrasyon" yaz. |
| A11 | LR/kalibrasyon içinde **StandardScaler** (Ölçekleme) | Yalnız **RobustScaler** uygulanıyor; ayrı StandardScaler yok | **DÜZELT (minör).** RobustScaler yaz. |
| A12 | Kategorik eksik → **mod** ile doldurma (Eksik değer yönetimi) | Kategorik eksik kendi **"Missing" kategorisi** + frekans kodlaması olarak korunur (mod yok) | **DÜZELT (minör).** Q&A "eksik≠sıfır"; ayrı "Missing" kategorisi sinyali korur. |

> **Doğru kalanlar (dokunma):** `environment.yml` var ✓; panel CV "Repeated 5-Fold ×10" ✓;
> Optuna hedefi PR-AUC ✓; `scale_pos_weight` ✓; `random_state=42 tam tekrarlanabilir` ✓
> (artık Optuna da seedli — gerçekten deterministik); SHAP grup haritalaması + waterfall ✓.

---

## B) RAPORA EKLE — kod yapıyor, rapor anmıyor

| # | Eklenecek (kod ne yapıyor) | Rapor bölümü | Neden |
|---|---|---|---|
| B1 | **Test prior'una göre eşik + `Class1_F1_TestPrior` metriği.** Eşik ve model/ensemble seçimi, test prior'u π=0.20 altında yeniden kurulan class-1 F1'i maksimize eder. TPR/FPR prior-bağımsız olduğundan precision şöyle yeniden kurulur: `precision_π = π·TPR / (π·TPR + (1−π)·FPR)`. Yeni metrik görülmemiş test F1'ini öngörür. | Sınıf Dengesi; Performans Metrikleri (metrik tablosuna ekle); **Özgünlük #2** | **EN ÖNEMLİ.** Raporun kendi Özgünlük #2'sini (asimetrik dağılıma özel kalibrasyon+eşik) fiilen TESLİM eden somut yöntem; rapor bunu yalnız "hedef" olarak anıyor, yöntemi yok. |
| B2 | **Bağımsız paneller + master-panel sızıntı kontrolü.** Her panel yalnız kendi verisiyle eğitilir; pipeline master-panel `Variant_ID` örtüşmesini raporlar (leakage-aware). | Veri ve Yöntem; Öğrenme Süreci (meta-learning'in yerine) | Terk edilen meta-learning'in yerini alır; "4 ayrı model" şartını karşılar. |
| B3 | **Panellere LGBM+XGB ham (kalibrasyonsuz) soft-voting + panel-başına geçiş (gate).** OOF'ta her iki model eğitilir, 0.5/0.5 ham ortalama; ensemble yalnız OOF test-prior F1 tek-LGBM'i **geçerse** tutulur → KANSER & PAH ensemble, CFTR tek-LGBM. | Ensemble Katmanı; Deney Tasarımı | Çeşitlilik ranking'i artırdı (**KANSER +4.83pp, PAH +3.65pp**); gate küçük panelde overfit'i önler. **Panelde kalibrasyon YOK:** OOF-fit izotonik kalibratörü full-data olasılığa uygulamak inference eşiğini bozuyordu (CFTR'de tüm tahminler 0). |
| B4 | **XGBoost da Optuna ile ayarlanıyor + XGB arama uzayı** (max_depth 3–10, learning_rate, min_child_weight 1–10, reg_lambda/alpha 0–5, subsample/colsample 0.6–1.0). Ayrıca LGBM aramasında **reg_alpha** da var (rapor listesi atlamış). | Parametre Seçimi (arama uzayı) | Rapor XGB'yi "ikincil model" diyor ama yalnız LGBM uzayını veriyor ve XGB'nin ayarlandığını söylemiyor. |
| B5 | **Soft-voting ağırlıkları OOF'ta optimize ediliyor** (LGBM/XGB grid; test-prior F1 maks; `ensemble_meta.joblib`'e yazılıp inference'ta okunur). | Ensemble Katmanı | Rapor "ağırlıklı ortalama" diyor ama ağırlıkların öğrenildiğini/sabit olmadığını söylemiyor. |
| B6 | **`missing_concentration`** (satır-düzeyi eksiklik oranı) özelliği. | Veri Ön İşleme | Rapor kolon-bazlı eksiklik bayrağını anıyor (✓); satır-düzeyi toplam eksiklik özelliği yok. |
| B7 | **Gerçek deterministik sonuçlar** (rapor plan raporu, sonuç tablosu yok). Beklenen test class-1 F1: Master **0.6043** · KANSER **0.7138** · PAH **0.5601** · CFTR **0.7671** · **ortalama 0.6613**. ~7 dk (SHAP dahil), 4 model inference dağılımı sağlıklı. | Deney Sonuçları (yeni bölüm) | Faz-2 raporu somut sonuç bekliyor; güncel rakamlar bunlar. |
