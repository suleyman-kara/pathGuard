# PathGuard

Missense genetik varyantları **Patojenik / Benign** olarak sınıflandıran bir makine öğrenmesi pipeline'ı. Dört bağımsız model üretir: **Master (Genel)**, **KANSER**, **PAH**, **CFTR**. Her panel master modelinden tamamen bağımsız, yalnızca kendi verisiyle eğitilir.

## Kurulum (Windows / PowerShell)

**Gereksinim:**

```powershell
# 1) Repoyu klonla ve içine gir
git clone https://github.com/MEN-INA/pathGuard.git
cd pathGuard

# 2) Sanal ortam (venv) oluştur
py -m venv venv

# 3) venv'i aktive et (her yeni PowerShell oturumunda tekrar gerekir)
.\venv\Scripts\Activate.ps1

# 4) Bağımlılıkları kur
pip install -r requirements.txt
```

Aktivasyon başarılıysa satır başında `(venv)` görünür. `venv` klasörü Git'e gönderilmez; herkes kendi bilgisayarında bu adımları çalıştırarak kendi ortamını kurar — bu yüzden arkadaşın da aynı `requirements.txt` ile birebir aynı sürümleri elde eder.

### Sorun Giderme

- **`source: command not found` / `venv/bin/activate` çalışmıyor:** Bunlar Linux/macOS komutudur. Windows PowerShell'de aktivasyon: `.\venv\Scripts\Activate.ps1`.
- **`python` tanınmıyor ama `py` çalışıyor:** Windows'ta normaldir; Python launcher'ı `py`'dir. Bu repodaki tüm komutlar `py` kullanır. (venv aktifken `py` otomatik olarak venv'i kullanır.)
- **`Activate.ps1 ... çalıştırılamıyor (execution policy)`:** Bir kez şunu çalıştır: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` ve tekrar dene.
- **Alternatif (her zaman çalışır):** Aktivasyonla uğraşmadan venv python'unu doğrudan çağır: `.\venv\Scripts\python.exe scripts/train.py --trials 1 --cv-repeats 1 --skip-shap`

## Eğitim

```powershell
# Hızlı smoke test
py scripts/train.py --trials 1 --cv-repeats 1 --skip-shap

# Tam eğitim
py scripts/train.py --trials 30
```

## Tahmin

```powershell
# Master (Genel) model
py scripts/predict.py data/raw/TEST_FILE.csv --panel MASTER --submission-only --output submission.csv

# Panel modeli (KANSER | PAH | CFTR) — master ağırlığı gerektirmez
py scripts/predict.py data/raw/CFTR_TEST.csv --panel CFTR --submission-only --output cftr.csv
```

Girdi CSV'si eğitimdeki feature şemasıyla uyumlu olmalıdır; eksik/fazla kolonda script açık hata verir.

## Mimari

```mermaid
flowchart TD
    subgraph Data_Prep ["1. Veri Hazırlama & Kalite Raporu"]
        A[Ham CSV Verileri] --> B(Deduplication & Kalite Kontrol)
        B --> C{Etiket Çelişkisi Var mı?}
        C -- Evet --> D[Tüm Çelişkili Kayıtları Sil]
        C -- Hayır --> E[İlk Eşleşen Kaydı Tut]
        D --> F[Sızıntı Kontrolü & Veri Kalite Raporu]
        E --> F
    end

    subgraph Preprocessing ["2. Ön İşleme & Ölçekleme"]
        F --> G(VariantFeatureEncoder)
        G --> H[Kategorik Encoding & Frekans Özellikleri]
        G --> I["Eksiklik Bayrağı: is_missing + missing_concentration"]
        G --> J[Eksiklik >%60: Kolon Düşürme]
        G --> K[Eksiklik %30-%60: IterativeImputer]
        G --> L[Eksiklik <%30: Medyan Imputer]
        G --> M[RobustScaler Ölçekleme]
        H & I & J & K & L & M --> N[Ön İşlemeden Geçmiş Veri]
    end

    subgraph Master_Pipeline ["3. Genel (Master) Model Süreci"]
        N --> O("Optuna Ayarı: LightGBM + XGBoost")
        N --> Q(5-Fold Stratified CV)
        Q --> R[LightGBM & XGBoost OOF Tahminleri]
        R --> S(Isotonic Olasılık Kalibrasyonu)
        S --> T{"Ensemble Seçimi: ağırlık-opt soft-voting vs stacking"}
        O & T --> V[Eğitilmiş Master Modelleri & Kalibratörler]
    end

    subgraph Panel_Pipeline ["4. Bağımsız Panel Modelleri"]
        N --> X[Panel Verisiyle Bağımsız LGBM + XGBoost Eğitimi]
        X --> Y(Repeated Stratified 5-Fold CV)
        Y --> Z["LGBM & Ham Soft-Voting Ensemble OOF"]
        Z --> GATE{"Panel-başına Geçiş: ensemble vs tek-LGBM (OOF test-prior F1)"}
        GATE --> AA(Test Prior'una Göre Eşik Optimizasyonu)
        AA --> CC[Eğitilmiş Panel Modelleri & Panel Eşikleri]
    end
```

## Sonuçlar

`py scripts/train.py --trials 30` çalıştırması.

> **"Beklenen Test F1" nedir?** İki sütun da **aynı metriği** — patojenik sınıfın (class 1) F1'ini — gösterir; tek fark hangi sınıf dağılımında ölçüldüğüdür. Eğitim/OOF dağılımı ~%80 patojenik, **final test seti ise ~%20 patojeniktir** (Q&A). Yarışma skoru test setinde ölçüleceği için gerçek sıralama metriği **Beklenen Test F1** sütunudur. Recall ve specificity sınıf oranından bağımsız olduğundan bunları sabit tutup precision'ı test prior'unda (%20) yeniden hesaplarız. OOF F1 yalnızca referanstır (yanlış dağılımda olduğu için yüksek görünür).

| Model | OOF Class 1 F1 | **Beklenen Test F1** | Recall | Specificity | ROC-AUC | Model tipi |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Master | 0.8336 | **0.5939** | 0.7666 | 0.7962 | 0.8593 | kalibre LGBM+XGB soft-voting |
| KANSER | 0.8428 | **0.7138** | 0.7585 | 0.9083 | 0.9164 | LGBM+XGB ham ensemble |
| PAH | 0.7886 | **0.5601** | 0.6743 | 0.8167 | 0.8031 | LGBM+XGB ham ensemble |
| CFTR | 0.7671 | **0.7671** | 0.6222 | 1.0000 | 0.8905 | tek LGBM (gate kararı) |

Eşik, test prior'u (%20) altında class 1 F1'i maksimize edecek şekilde seçilir. Paneller, OOF
test-prior F1'i tek-LGBM'i geçerse LGBM+XGB **ham soft-voting ensemble** kullanır (panel-başına
geçiş/gate); KANSER ve PAH ensemble seçti, CFTR tek modelde kaldı. PAH en zor panel (düşük
ROC-AUC); CFTR'de yanlış-pozitif yok → specificity 1.0, bu yüzden iki F1 eşit. Ayrıntı ve
baseline'a göre kazanç (ortalama +1.81pp): `docs/rapor_guncellemeleri.md`.

## Notlar

- `models/` ve `outputs/` Git dışındadır (eğitimle üretilir).
- Dış veri kaynağından etiket sorgulanmaz; yalnızca verilen varyant profilleri kullanılır.
- Paneller master'dan bağımsızdır → master-panel örtüşmesinden kaynaklı yanıltıcı (sızıntılı) OOF F1 ortadan kalkar.

---

**Takım:** MEN-INA | **Proje:** PathGuard
