# PathGuard — Sonraki Adımlar (İlerleme Planı)

Bu dosya, final aşaması hazırlığında **nereden devam edileceğini** özetler. Karar geçmişi
ve gerekçeler için `docs/rapor_guncellemeleri.md`'ye, proje kurallarına `CLAUDE.md`'ye bakın.

## Şu ana kadar yapılanlar (branch: `feat/independent-panels-and-recall-removal`)

1. **Panel modelleri master'dan bağımsızlaştırıldı** — `master_soft_prediction` sızıntısı
   kaldırıldı; her panel yalnızca kendi verisiyle eğitiliyor (Q&A: 4 bağımsız model).
2. **Recall ≥ 0.90 kısıtı kaldırıldı** (`CLINICAL_RECALL_TARGET = 0.0`) — eşik doğrudan
   class 1 F1'i maksimize ediyor.
3. **Test prior'una göre eşik** (`TEST_PATHOGENIC_PRIOR = 0.20`) — eşik, gerçek test
   dağılımında (~%20 patojenik) class 1 F1'i maksimize ediyor; raporlara `Class1_F1_TestPrior`
   eklendi (gerçek yarışma skoru öngörüsü).
4. **Repo sağlamlaştırma** — sürümler sabitlendi, `.gitignore`'a `venv/`, README sadeleştirildi.

**Güncel beklenen test F1 (gerçek sıralama metriği öngörüsü):**
Master 0.595 · KANSER 0.645 · PAH 0.497 · CFTR 0.741 (ortalama ~0.62).

> Bu branch henüz `main`'e merge edilmedi. İlk iş: PR açıp gözden geçirip merge etmek.
> PR: https://github.com/MEN-INA/pathGuard/pull/new/feat/independent-panels-and-recall-removal

## Açık hedef

Görülmemiş test verisinde **patojenik sınıf (class 1) F1'ini** maksimize etmek. Eşik artık
optimal noktada olduğundan, bundan sonraki kazanım **modelin ayırt etme gücünü (ranking:
ROC/PR-AUC)** artırmaktan gelir.

## Öncelikli yol haritası (etki sırasına göre)

### A. Performans — en yüksek getiri
1. **PAH panelini güçlendir (en zayıf, ROC-AUC 0.76).** Beklenen test F1'i (0.50) en çok
   bu çekiyor. Denenecekler: panel-özel özellik mühendisliği, daha güçlü düzenlileştirme
   araması, sentetik veri (yarışma izin veriyor), gerekirse panel için farklı model.
2. **Eksiklik bayrağı özelliği** (`src/preprocessing.py`). Yüksek-eksiklikli kolonlar için
   ikili "eksik mi" özelliği ekle. Rapor "eksikliğin kendisi bilgilendiricidir" diyor →
   hem rapora uyum hem olası F1 kazancı. **Hızlı ve düşük riskli; ilk buradan başla.**
3. **Panellere kalibrasyon/ensemble.** Şu an her panel tek LightGBM. Master'daki kalibre
   LGBM+XGB soft-voting'i panellere de uygulamak ranking'i iyileştirebilir (küçük örneklemde
   overfit'e dikkat).

### B. Rapor-kod uyumu (kozmetik → düşük öncelik)
4. **Aykırı değer taraması + label smoothing** (`src/data_loader.py` / eğitim) — gürültülü
   etiketler için. Rapor bahsediyor; orta etki.
5. **Optuna 50–100 deneme** (`--trials`) — master ranking'i hafif iyileştirebilir; süre bütçesine
   göre artır.
6. **Çok-seed CV** — sağlamlık raporu için CV'yi 3 seed ile tekrarla. ~0 F1 etkisi, çoğunlukla
   rapor uyumu; istenirse `docs/rapor_guncellemeleri.md`'de sapma olarak da bırakılabilir.
7. **Soft-voting'e kalibre LR** — rapor LGBM+XGB+LR diyor; kod LGBM+XGB. Küçük çeşitlilik etkisi.

### C. İsteğe bağlı
8. **Olasılık prior-kalibrasyonu** — olasılıkların da test prior'una göre kalibrasyonu
   (reliability diagram / Brier kalitesi için; F1 için gerekli değil).

## Kalıcı kısıtlar / notlar
- **Doküman önceliği:** `soru-cevap.md` > `yarisma-raporu.md` > `yarisma-sartnamesi.md`.
- **`yarisma-raporu.md` dondurulmuştur** (düzenlenmez); sapmalar `rapor_guncellemeleri.md`'ye.
- Şartnamedeki veri dağılımı **hatalı**; Q&A esas (train ~%80, test ~%20 patojenik).
- "Değişimsiz varyant (ref==alt)" tespiti kolon adları gizli olduğu için yapılamıyor (yapısal kısıt).

## Nasıl çalıştırılır (hatırlatma)
```powershell
.\venv\Scripts\Activate.ps1
py scripts/train.py --trials 30          # tam eğitim
py scripts/predict.py <test.csv> --panel CFTR --submission-only --output cftr.csv
```
Metrikler: `outputs/master_metrics.json`, `outputs/panel_*_metrics.json` →
`Class1_F1_TestPrior` alanı gerçek hedef metriği gösterir.
