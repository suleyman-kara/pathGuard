# PathGuard — Sonraki Adımlar (İlerleme Planı)

Bu dosya, final aşaması hazırlığında **nereden devam edileceğini** özetler. Karar geçmişi
ve gerekçeler için `docs/rapor_guncellemeleri.md`'ye, proje kurallarına `CLAUDE.md`'ye bakın.

## Şu ana kadar yapılanlar

**Önceki tur (merge edildi, PR #3):**
1. **Panel modelleri master'dan bağımsızlaştırıldı** — `master_soft_prediction` sızıntısı
   kaldırıldı; her panel yalnızca kendi verisiyle eğitiliyor (Q&A: 4 bağımsız model).
2. **Recall ≥ 0.90 kısıtı kaldırıldı** (`CLINICAL_RECALL_TARGET = 0.0`).
3. **Test prior'una göre eşik** (`TEST_PATHOGENIC_PRIOR = 0.20`) — `Class1_F1_TestPrior` raporlanıyor.

**Bu tur (2026-06-22, F1 maksimizasyon turu — detay `rapor_guncellemeleri.md` Değişiklik 5–7):**
4. **Eksiklik bayrağı özellikleri** (`src/preprocessing.py`) — yüksek-eksiklikli kolonlara
   `{col}_is_missing` + `missing_concentration`. Ortalama +0.52pp.
5. **Master XGBoost Optuna ayarı + soft-voting ağırlık optimizasyonu** (`src/pipeline.py`) —
   XGB artık ayarlı (önceden default), ağırlık OOF'ta optimize ediliyor (`predict.py` meta'dan okur).
6. **Panellere ham (kalibrasyonsuz) soft-voting ensemble + panel-başına geçiş** — KANSER & PAH
   ensemble kazandı, CFTR tek-LGBM kaldı (gate kararı). CFTR inference all-zero hatası düzeltildi.

**Güncel beklenen test F1 (baseline → güncel):**
Master 0.590→**0.594** · KANSER 0.666→**0.714** · PAH 0.524→**0.560** · CFTR 0.784→0.767 ·
**ortalama 0.641→0.659 (+1.81pp)**. Çalışma ~6.6 dk. Tüm inference dağılımları sağlıklı.

## Açık hedef

Görülmemiş test verisinde **patojenik sınıf (class 1) F1'ini** maksimize etmek. Eşik artık
optimal noktada olduğundan, bundan sonraki kazanım **modelin ayırt etme gücünü (ranking:
ROC/PR-AUC)** artırmaktan gelir.

## Öncelikli yol haritası (etki sırasına göre)

### A. Performans — en yüksek getiri
1. ~~**Eksiklik bayrağı özelliği**~~ → **YAPILDI** (bu tur, +0.52pp ort.).
2. ~~**Panellere ensemble**~~ → **YAPILDI** (bu tur, ham soft-voting + panel-başına geçiş;
   KANSER +4.83pp, PAH +3.65pp). **NOT/DÜZELTME:** Önceki "panellere **kalibrasyon** →
   ranking iyileşir" maddesi F1 için yanlıştı — izotonik kalibrasyon monotoniktir, eşik
   sonradan optimize edildiğinden F1'i/AUC'yi değiştirmez; kazanç **ensemble çeşitliliğinden**
   gelir. Üstelik OOF-fit kalibratör full-data modele uygulanınca inference eşiği bozuluyordu
   (CFTR all-zero) → paneller **kalibrasyonsuz** ham ortalama kullanır.
3. **PAH'ı daha da güçlendir (hâlâ en zayıf, 0.560).** Ensemble +3.65pp verdi ama PAH dipte.
   Denenecekler: PAH-özel hiperparametre tuning (paneller şu an sabit param), panel-özel özellik
   mühendisliği, sentetik veri (yarışma izin veriyor — ama AUC taban-orandan bağımsız olduğu
   için beklenen getiri düşük). Küçük sette overfit'e dikkat; OOF test-prior F1 ile doğrula.
4. **CFTR missing-flag hassasiyeti.** Eksiklik bayrakları CFTR'yi ~−1.7pp etkiledi (recall
   granülaritesi; ranking sağlam). Panel-bazlı özellik dışlama ~+0.4pp kazandırabilir (belirsiz).

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
