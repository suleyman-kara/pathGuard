import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import CAT_COLS, AA_COLS

class VariantFeatureEncoder:
    """
    Missense genetik varyant özellikleri için kodlayıcı ve ön işlemci.
    - Kategorik özellikleri (CAT_1..CAT_6, AA_1, AA_2) sayısal etiketlere ve frekans kodlamalarına dönüştürür.
    - Sürekli (continuous) sayısal özellikler için:
      * %60'tan fazla eksik değer içeren özellikleri düşürür.
      * %30 ile %60 arasında eksikliği olanlara IterativeImputer uygular.
      * %30'dan az eksikliği olanlara SimpleImputer(median) uygular.
      * Tüm sürekli özelliklere RobustScaler ölçekleme adımını uygular.
    """
    def __init__(self, rare_threshold: int = 3, **kwargs):
        self.rare_threshold = rare_threshold
        self.cat_cols = CAT_COLS + AA_COLS
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.label_maps: Dict[str, Dict[str, int]] = {}
        self.feature_columns_: List[str] = []
        self.output_columns_: List[str] = []
        
        # Sürekli özellik ön işleme öznitelikleri
        self.dropped_cols: List[str] = []
        self.low_missing_cols: List[str] = []
        self.mid_missing_cols: List[str] = []
        self.kept_continuous_cols: List[str] = []
        # Eksiklik bayrağı için: yüksek-eksiklikli kolonlar ve tüm sürekli kolon listesi
        self.missing_flag_cols: List[str] = []
        self.all_continuous_cols_: List[str] = []
        self.median_imputer: Optional[SimpleImputer] = None
        self.iterative_imputer: Optional[IterativeImputer] = None
        self.scaler: Optional[RobustScaler] = None
        
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "VariantFeatureEncoder":
        if X.empty:
            raise ValueError("Boş veri kümesi üzerinde fit işlemi yapılamaz.")

        self.feature_columns_ = list(X.columns)

        # 1. Kategorik değişkenler için kodlama haritalarını oluştur
        for col in self.cat_cols:
            if col in X.columns:
                # Eksik kategorileri 'Missing' ile doldur
                filled = X[col].fillna("Missing").astype(str)
                counts = filled.value_counts()
                freqs = filled.value_counts(normalize=True).to_dict()
                self.freq_maps[col] = freqs

                # Nadir kategorileri bul
                rare_cats = set(counts[counts < self.rare_threshold].index)

                # Sayısal kodlama haritasını oluştur (0: Missing, 1: Rare/Unknown, 2+: Kategoriler)
                unique_valid = [c for c in counts.index if c not in rare_cats and c != "Missing"]
                l_map = {"Missing": 0, "Rare/Unknown": 1}
                for idx, val in enumerate(unique_valid, start=2):
                    l_map[val] = idx
                self.label_maps[col] = l_map

        # 2. Sürekli özellikleri eksiklik oranlarına göre grupla
        continuous_cols = [c for c in X.columns if c not in self.cat_cols]
        self.dropped_cols = []
        self.low_missing_cols = []
        self.mid_missing_cols = []
        self.kept_continuous_cols = []

        for col in continuous_cols:
            missing_rate = X[col].isna().mean()
            if missing_rate > 0.60:
                self.dropped_cols.append(col)
            elif missing_rate >= 0.30:
                self.mid_missing_cols.append(col)
                self.kept_continuous_cols.append(col)
            else:
                self.low_missing_cols.append(col)
                self.kept_continuous_cols.append(col)

        # 2b. Eksiklik bayrağı özellikleri için kolonları belirle.
        # Rapor: "eksikliğin kendisi bilgilendiricidir". Yalnızca yüksek-eksiklikli (>=%30)
        # kolonlara ikili "eksik mi?" bayrağı ekleriz; düşük-eksiklikli kolonların bayrağı
        # neredeyse sabit (gürültü) olacağı için dışarıda bırakılır. Düşürülen (>%60) kolonların
        # değerleri atılsa da eksiklik sinyali korunur. all_continuous_cols_ satır-düzeyi
        # eksiklik yoğunluğu (missing_concentration) için kullanılır.
        self.missing_flag_cols = list(self.mid_missing_cols) + list(self.dropped_cols)
        self.all_continuous_cols_ = list(continuous_cols)

        # 3. Sürekli özellikler için imputasyon nesnelerini eğit
        X_cont = X[self.kept_continuous_cols].copy()
        X_cont = X_cont.replace([np.inf, -np.inf], np.nan)

        if self.low_missing_cols:
            self.median_imputer = SimpleImputer(strategy="median")
            self.median_imputer.fit(X_cont[self.low_missing_cols])
            X_cont[self.low_missing_cols] = self.median_imputer.transform(X_cont[self.low_missing_cols])

        if self.mid_missing_cols:
            # Hatalı n_jobs kaldırıldı, takılmayı önleyen güvenli ayarlar korundu
            self.iterative_imputer = IterativeImputer(
                max_iter=20,                 
                tol=1e-2,                 
                n_nearest_features=10,    
                imputation_order='ascending',
                random_state=42
            )
            self.iterative_imputer.fit(X_cont)
            X_cont_imputed = self.iterative_imputer.transform(X_cont)
            X_cont = pd.DataFrame(X_cont_imputed, columns=self.kept_continuous_cols, index=X.index)

        # 4. Ölçekleyiciyi (RobustScaler) eğit
        if self.kept_continuous_cols:
            self.scaler = RobustScaler()
            self.scaler.fit(X_cont)

        self.is_fitted = True
        self.output_columns_ = list(self._transform_internal(X).columns)
        return self

    def _validate_schema(self, X: pd.DataFrame) -> None:
        incoming = set(X.columns)
        expected = set(self.feature_columns_)
        missing = sorted(expected - incoming)
        extra = sorted(incoming - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"eksik kolonlar: {missing[:20]}")
            if extra:
                parts.append(f"fazla kolonlar: {extra[:20]}")
            raise ValueError("Girdi veri şeması uyumsuz (" + "; ".join(parts) + ").")

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        out_df = X.copy()
        
        # Sonsuz değerleri NaN ile değiştir
        numeric_cols = out_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            out_df[numeric_cols] = out_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Eksiklik bayrağı özellikleri: imputasyon ÖNCESİ NaN maskesinden üretilir.
        # (out_df'in sürekli kolonları aşağıda impute edilince üzerine yazılır; bu yüzden
        #  maskeyi burada yakalayıp en sona ekliyoruz.)
        missing_flag_features: Dict[str, pd.Series] = {}
        for col in self.missing_flag_cols:
            if col in out_df.columns:
                missing_flag_features[f"{col}_is_missing"] = out_df[col].isna().astype(int)
        present_cont = [c for c in self.all_continuous_cols_ if c in out_df.columns]
        if present_cont:
            missing_flag_features["missing_concentration"] = (
                out_df[present_cont].isna().mean(axis=1).astype(float)
            )

        freq_features: Dict[str, pd.Series] = {}
        
        for col in self.cat_cols:
            if col in out_df.columns:
                filled = out_df[col].fillna("Missing").astype(str)
                
                # 1. Frekans kodlama özelliği ekle
                freq_map = self.freq_maps.get(col, {})
                freq_features[f"{col}_freq"] = filled.map(freq_map).fillna(0.0).astype(float)
                
                # 2. Kategorik etiket kodlaması
                l_map = self.label_maps.get(col, {})
                def map_label(val: str) -> int:
                    if val == "Missing":
                        return 0
                    return l_map.get(val, 1) # 1: Nadir/Bilinmeyen
                
                out_df[col] = filled.map(map_label).astype("category")

        # Frekans kolonlarını ekle
        if freq_features:
            out_df = pd.concat([out_df, pd.DataFrame(freq_features, index=out_df.index)], axis=1)

        # Eksiklik bayrağı kolonlarını ekle (imputasyondan önce yakalanan maskeden)
        if missing_flag_features:
            out_df = pd.concat([out_df, pd.DataFrame(missing_flag_features, index=out_df.index)], axis=1)

        # 3. Sürekli özellikleri doldur ve ölçekle
        if self.kept_continuous_cols:
            X_cont = out_df[self.kept_continuous_cols].copy()
            
            # Medyan doldurma
            if self.low_missing_cols and self.median_imputer is not None:
                X_cont[self.low_missing_cols] = self.median_imputer.transform(X_cont[self.low_missing_cols])
                
            # Çok değişkenli doldurma (IterativeImputer)
            if self.mid_missing_cols and self.iterative_imputer is not None:
                X_cont_imputed = self.iterative_imputer.transform(X_cont)
                X_cont = pd.DataFrame(X_cont_imputed, columns=self.kept_continuous_cols, index=out_df.index)
                
            # RobustScaler ile ölçekleme
            if self.scaler is not None:
                X_cont_scaled = self.scaler.transform(X_cont)
                X_cont = pd.DataFrame(X_cont_scaled, columns=self.kept_continuous_cols, index=out_df.index)

            # Ön işleme geçmiş sürekli sütunları güncelle
            for col in self.kept_continuous_cols:
                out_df[col] = X_cont[col]

        # 4. Çok fazla eksiklik içeren kolonları düşür
        if self.dropped_cols:
            out_df = out_df.drop(columns=self.dropped_cols)

        return out_df

    def transform(self, X: pd.DataFrame, for_tree: bool = True) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder fit edilmeden transform çağrılamaz.")
        self._validate_schema(X)
        out_df = self._transform_internal(X)
        if self.output_columns_:
            out_df = out_df[self.output_columns_]
        return out_df

    def fit_transform(self, X: pd.DataFrame, for_tree: bool = True) -> pd.DataFrame:
        return self.fit(X).transform(X, for_tree=for_tree)
