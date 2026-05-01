**RAPOR BİLGİLERİ**  
Görev: Missense Genetik Varyantların Patojenik/Benign Olarak  
Sınıflandırılması  
Proje Adı: PathGuard  
Takım Adı: MEN-INA

**TAKIM ŞEMASI**

| Rol | Eğitim Seviyesi / Alan | Teknik Sorumluluk |
| :---- | :---- | :---- |
| Takım Kaptanı | Lisans – Bilgisayar ve  Öğretim Teknolojileri  Eğitimi | Proje koordinasyonu, genel mimari  tasarımı, MLOps ve deney kayıt  yönetimi, raporlama, sunum |
| Danışman | Akademisyen (Doktora  / Öğretim Üyesi) | Biyoinformatik rehberlik, ACMG  kriterleri yorumlama, kalite kontrol,  bilimsel geçerlilik denetimi,  metodolojik gözetim |
| Takım Üyesi 1 | Lisans – Bilgisayar  Mühendisliği | Model geliştirme (gradient boosting,  ensemble), hiperparametre  optimizasyonu, feature engineering,  çapraz doğrulama altyapısı |
| Takım Üyesi 2 | Lisans – Bilgisayar  Mühendisliği | Veri ön işleme, eksik değer yönetimi,  sınıf dengesizliği stratejileri, olasılık  kalibrasyonu, açıklanabilirlik (SHAP)  analizi |
| Takım Üyesi 3 | Lisans – Tıp (MD) | Klinik varyant yorumlama, ACMG  uyumluluğu denetimi, yanlış  sınıflandırmaların klinik anlam  analizi, klinik risk perspektifi |

**Karar Alma ve Kalite Kontrol Mekanizması**  
Takım, haftalık çevrim içi toplantılarla ilerlemeyi değerlendirir. Her model denemesi  MLflow veya benzeri bir deney takip aracıyla kayıt altına alınır (parametre, metrik, model  sürümü). Kod incelemeleri iki bilgisayar mühendisi arasında karşılıklı yapılmakta;  danışman son metodolojik denetimi gerçekleştirmektedir. Tıp üyesi her yeni modelin  klinik anlam taşıyan çıktılarını değerlendirerek yanlış negatiflerin hasta güvenliğine etkisini tartmaktadır. Git sürümleme ile tüm deney geçmişi izlenebilir tutulmaktadır.

**Yetkinlik Matrisi** 

Takımın bu görev için kritik olan biyoinformatik, istatistik ve yazılım yetkinliklerinin  birleşimi şu şekilde özetlenebilir: 

* Biyoinformatik: Danışman (uzman), tıp üyesi (klinik bağlam), bilgisayar mühendisleri  (araç kullanımı)   
* Makine Öğrenmesi: İki bilgisayar mühendisi (derin teknik), takım kaptanı (sistem  entegrasyonu)  
* Klinik Anlam / Hasta Güvenliği: Tıp üyesi (primer sorumlu), danışman (akademik çerçeve)  
* Veri Mühendisliği ve MLOps: Takım kaptanı ve bilgisayar mühendisleri (ortak)

**PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ**

| Makale /  Model | Veri / Yaklaşım | Raporlanan  Metrikler | Kısıtlılık & Biz Neden İlgiliyiz |
| :---- | :---- | :---- | :---- |
| ClinPred \[1\] (Alirezaie et  al., 2018\) | ClinVar \+ HGMD;  Random Forest \+  18 in-silico skor  kombinasyonu;   tabular özellikler | AUC: 0.95,  F1: 0.91 | Yalnızca pre-hesaplanmış skorları  kullanır, raw biyokimyasal  özellikleri içermez. Bizim varyant  profil yaklaşımımız daha geniş  özellik uzayı sunar. |
| REVEL \[2\]  (Ioannidis et  al., 2016\) | 13 in-silico aracın  ensemble skoru;  lojistik regresyon  tabanlı meta  model; missense  odaklı | AUC: 0.90,  PR-AUC:   0.87 | Meta-skor olduğu için tek başına  yorumlanamaz, bağımsız özellik  katkısı görünmez. SHAP ile  açıklanamaz. Biz özellik-bazlı şeffaf  modeli tercih ediyoruz. |
| PrimateAI  3D\[3\]  (Sundaram  et al., 2024\) | Primatlarda   evrimsel dizilim;  derin öğrenme  (CNN \+ protein  yapı); Transformer  tabanlı | AUC: 0.98  (Mendelian  hastalıklar) | Protein 3D yapısı ve büyük ölçekli  seq verisi gerektirir; hesaplama  maliyeti çok yüksek; sağlanan  tablolar veri seti ile uyumsuz. Bizim  tabular yaklaşımımız pratik ve  tekrarlanabilir. |
| CardioBoost \[4\]  (Shen et al.,  2021\) | ClinVar \+  gnomAD; XGBoost  tabanlı hastalık spesifik model;   kardiyomiyopati  paneli | AUC: 0.93  (hastalık  spesifik) | Tek hastalık alanına özel;  genelleşebilirlik sınırlı. Biz hem  genel hem panel-spesifik modeller  geliştiriyor; bu çalışma panel  modelinin gerekçesini destekliyor. |
| EVE   (Evolutionar y)\[5\]  (Frazer et  al., 2021,  Nature)  | MSA (çoklu dizi  hizalama) \+  variational   autoencoder;   etiket   gerektirmeyen   öğrenme | AUC: 0.88– 0.93;   spearman-r  \> 0.60 | Etiket kullanmayan unsupervised  yaklaşım klinik doğruluk için  yetersiz. Varyant profil verimizle  eğitilemeyen harici model  bağımlılığı var. Bizim yaklaşımımız  veri odaklı ve denetimli. |
| ClinVar-ML  (Tavtigian et  al.) \[6\]  (2020, EJHG)  | ClinVar ACMG  uyumlu etiketler;  Gradient Boosting  (LightGBM);   özellik: allel  frekansı \+ in-silico  skorlar  | AUC: 0.96,  Sensitivite:  0.91,   Özgüllük:   0.94  | Yarışmamızın referans alacağı  ACMG etiket çerçevesini kullanan  en yakın çalışma. LightGBM'in bu  problem türündeki üstünlüğünü  doğruluyor. Metodolojimizin temel  referansı. |

**VERİ VE YÖNTEM** 

**Kullanılan Veri Seti ve Etiketler** 

Yarışma kapsamında dört ayrı veri seti kullanılmaktadır:

1. genel büyük varyant seti  
2. Herediter Kanser paneli  
3. PAH (Fenilalanin Hidroksilaz / Fenilketonüri) paneli  
4. CFTR (Kistik Fibrozis Transmembran Regülatörü) paneli

Her veri seti; varyant profilinden  oluşan çok boyutlu özellik vektörleri ve ACMG uyumlu ikili sınıf etiketlerini (Patojenik /  Benign) içermektedir.

Etiket mantığı şu şekilde tanımlanmıştır: Kaynak veri tabanlarında "Pathogenic" ve "Likely  Pathogenic" olarak işaretlenen varyantlar tek bir Patojenik sınıfta birleştirilmiştir;  "Benign" ve "Likely Benign" etiketleri ise Benign sınıfını oluşturmaktadır. Bu birleştirme  ACMG 2015 rehberine uygundur ve klinik pratikte yaygın kabul görmektedir \[7\]. "Uncertain  Significance (VUS)" etiketli varyantlar modelin eğitiminde kullanılmamış; sınıflandırma  çıktısı yalnızca kesin etiketli örnekler üzerinde değerlendirilmiştir. 

Veri seti, yarışmacılara eğitim bölümü olarak sunulmuştur. Eğitim/test ayrımı yaklaşık  %80 eğitim – %20 test oranında gerçekleştirilmiş; ancak yarışma koşullarına göre eğitim  seti ağırlıklı olarak patojenik (\~%80), test seti ise ağırlıklı olarak benign (\~%80)  örneklerden oluşmaktadır. Bu asimetrik dağılım, modelin gerçek veri dağılımını  öğrenmesini zorlaştıran önemli bir metodolojik zorluktur.

**Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme** 

Yarışma organizasyonu, genomik konum bilgilerini (kromozom, pozisyon, rs ID) ve sütun  isimlerini bilinçli olarak gizlemiştir. Bu kısıta tam uyum sağlıyoruz: çözümümüz yalnızca  sağlanan sayısal varyant profilleri üzerinden çalışmakta; herhangi bir dış veri tabanına  (ClinVar, gnomAD, ClinPred API vb.) bağlanarak etiket ya da skor sorgulaması  yapılmamaktadır.

Dolaylı sızıntı risklerine karşı aldığımız önlemler:

1. Veri bölme her zaman eğitim setinden  ayrı, sabit rastgele tohum (seed) ile gerçekleştirilmektedir.  
2. Normalleştirme/ölçekleme  istatistikleri yalnızca eğitim verisi üzerinden hesaplanmakta, test setine  uygulanmaktadır.  
3. Panel setlerinin eğitim/test örneklerinin genel büyük sete sızıp sızmadığı kontrol  edilmektedir.  
4. Özellik seçim aşaması test setinden bağımsız yürütülmektedir.

**Veri Ön İşleme ve Temsilleme Stratejisi** 

Kolon isimleri olmaksızın sunulan varyant profillerini beş kategoride yorumluyoruz:

1. evrimsel korunmuşluk skorları  
2. popülasyon allel frekansları  
3. in-silico patojenite  tahmin skorları  
4. biyokimyasal ve yapısal etki göstergeleri  
5. yerel sekans bağlamına  ait sayısal özellikler

Bu özellik grubu haritalaması SHAP tabanlı açıklamalarda da kullanılmaktadır.

Eksik değer yönetimi: Her sütundaki eksiklik oranı belirlenir. %30'un altında eksiklik için  özellik grubuna uygun medyan (sürekli) veya mod (kategorik) ile doldurma; %30–60  arasında IterativeImputer ile çok değişkenli doldurma; %60'ın üzerinde ise özellik düşürme ya da ikili "eksiklik bayrağı" ekleme uygulanır. Eksiklik örüntüsü kendisi de bir  özellik olarak değerlendirilebilir, zira belirli in-silico araçların sonuç üretememesi işlevsel  bir anlam taşıyabilir.

Aykırı değer işlemi: Robust ölçekleme (median \+ IQR) ile uç değerlerin etkisi azaltılır; Z skoru \>5 olan değerler incelenerek gerçek biyolojik aykırılık mı yoksa veri hatası mı olduğu  değerlendirilir.

Ölçekleme/normalizasyon: Tree-based modeller için doğrudan gerekli değildir; ancak  olasılık kalibrasyonu ve lojistik regresyon bazlı soft-voting ensemble içinde  StandardScaler uygulanmaktadır.

Boyut indirgeme / özellik seçimi: LightGBM'in yerleşik özellik önem skoru (gain-based)  ve permütasyon önem testleri ile düşük katkılı özellikler belirlenir. Küçük panel veri  setlerinde gereksiz boyutları atmak aşırı uyumu (overfitting) azaltır.

**Etiket Güvenilirliği ve Veri Kalitesi Kontrolü** 

Ground truth etiketleri güvenilir kaynaklardan türetilmiş olsa da aşağıdaki kalite kontrol  adımları sistematik biçimde uygulanmaktadır:

* Tekrar eden kayıt kontrolü: Aynı özellik vektörüne sahip örnekler tespit edilir; etiket  tutarlılığı kontrol edilir, tutarsız olanlar işaretlenerek eğitime dahil edilmez. Uç değer taraması: Her özellik için %1 ve %99 persentil dışında kalan örnekler incelenir;  bunların etiket dağılımı genel dağılımla karşılaştırılır.   
* Tutarsız profil tespiti: Benign etiketli ama çok yüksek patojenite skoru taşıyan ya da tam  tersi durumlar gürültülü etiket adayı olarak işaretlenir. Bu örnekler eğitimde düşük  ağırlıkla kullanılır veya Label Smoothing uygulanır.  
* Eğitim sırasında anomali izleme: Erken durdurma (early stopping) eğrilerinde ani  bozulmalar, sorunlu örnek kümelerinin varlığına işaret edebilir; bu durum kayıp değerleri  üzerinden izlenir.

**Sınıf Dengesi ve Risk Perspektifi** 

Her alt veri setinde sınıf dağılımı raporlanmıştır. Eğitim seti ağırlıklı olarak patojenik  (\~%80), test seti ise ağırlıklı olarak benign (\~%80) örneklerden oluşmaktadır. Bu asimetrik  yapı, modelin eğitim sırasında öğrendiği karar sınırını test dağılımına göre yanlı kılabilir.

Bu riski azaltmak için benimsediğimiz stratejiler: 

* scale\_pos\_weight (LightGBM/XGBoost): Sınıf frekanslarının tersi ile ağırlıklandırma  yapılarak patojenik örneklerin baskın olmadığı sentetik bir dengesizlik düzeltmesi  sağlanır.   
* Eşik (threshold) optimizasyonu: Varsayılan 0.5 eşiği yerine, F1-macro'yu veya klinik  önceliğe göre Recall@Precision dengesi maksimize eden eşik seçilir.   
* Risk perspektifi – yanlış negatif vs. yanlış pozitif: Klinisyen bakışıyla, hastalık yapıcı bir  varyantı kaçırmak (yanlış negatif) çoğu durumda zararsız bir varyantı patojenik saymaktan  (yanlış pozitif) çok daha yüksek risk taşımaktadır. Bu nedenle eşik, Recall (duyarlılık) için  belirli bir alt sınır belirlenerek optimize edilir.

**Seçilen Algoritmalar ve Gerekçe**

Ana Model: LightGBM (Light Gradient Boosting Machine) 

Çok boyutlu tabular varyant profilleri için gradient boosting karar ağaçları optimal bir  seçimdir. LightGBM, histogram tabanlı bölme stratejisi sayesinde yüksek boyutlu  eksik değer içeren verilerde hız ve doğruluk dengesi kurar \[8\]. GBDT ailesi, genomik  skorlar ve allel frekansları gibi çarpık (skewed) dağılımlı değişkenlerde  normalleştirme gerektirmeksizin çalışır. Overfitting'e karşı L1/L2 düzenileştirme,  yaprak sayısı kısıtlaması ve early stopping mekanizmaları mevcuttur. 

İkincil Model: XGBoost 

LightGBM ile karşılaştırmalı değerlendirme için ikincil bir GBDT modeli olarak  kullanılır. Derinlik-öncelikli büyüme stratejisi bazı veri setlerinde farklı karar sınırları  üretebilir; bu da ensemble içindeki çeşitliliği artırır \[9\]. 

Ensemble Katmanı: Soft-Voting / Stacking 

LightGBM, XGBoost ve kalibre edilmiş bir Lojistik Regresyon modelinin olasılık  çıktıları, ağırlıklı ortalama (soft-voting) ile birleştirilir. Stacking yaklaşımında meta model olarak Logistic Regression tercih edilmektedir. Ensemble, tek model  varyansını azaltarak panel bazlı genellenebilirliği artırmaktadır. 

Olasılık kalibrasyonu: Karar eşiği seçiminin güvenilir olabilmesi için model  çıktılarının iyi kalibre edilmiş olasılıklar üretmesi gerekmektedir. İzotonik regresyon  veya Platt ölçekleme ile kalibrasyon uygulanır; kalibrasyon eğrisi (reliability diagram) ile doğrulanır.

**DENEY TASARIMI, SONUÇLAR VE İNCELEME** 

**Deney Protokolü ve Veri Bölme** 

Eğitim verisinde hiperparametre optimizasyonu ve model seçimi için Stratified 5-Fold  Çapraz Doğrulama kullanılmaktadır. Stratifiye bölme, her katta sınıf oranlarının  korunmasını sağlar ve özellikle küçük panel veri setlerinde rastlantısal iyi sonuç riskini  minimize eder. Çapraz doğrulama 3 farklı rastgele tohum (seed) ile tekrarlanarak  sonuçların sürüm bağımsızlığı test edilmektedir. 

Veri bölme düzeni şu şekilde işlemektedir:

1. Tüm eğitim seti 5 fold'a ayrılır  
2. Her iterasyonda 4 fold eğitim, 1 fold doğrulama olarak kullanılır  
3. Hiperparametreler doğrulama seti üzerindeki ortalama PR-AUC'a göre seçilir  
4. Nihai model tüm eğitim setiyle yeniden eğitilir ve asla görülmemiş test seti üzerinde değerlendirilir.

Panel veri setleri (Herediter Kanser, PAH, CFTR) küçük örneklem sayısı içerebileceğinden,  bu setlerde 5-Fold yerine Leave-One-Out Cross Validation (LOOCV) veya Repeated 5- Fold (10 tekrar) tercih edilmektedir. Bu yaklaşım, sonuçların örneklem varyansına olan  duyarlılığını ölçmeyi de sağlar. 

**Performans Metrikleri ve Panel Bazlı Raporlama** 

Birincil metrik olarak PR-AUC (Precision-Recall Area Under Curve) seçilmiştir. Sınıf  dengesizliğinin belirgin olduğu bu problem türünde ROC-AUC yanıltıcı olabilirken, PR AUC azınlık sınıfının (gerçek pozitif – patojenik) tahmin kalitesini daha hassas biçimde  yansıtmaktadır \[10\]. 

Raporlanan metrik seti:

| Metrik Gerekçe |  |
| :---- | :---- |
| PR-AUC  | Dengesiz veri setinde birincil optimizasyon hedefi |
| ROC-AUC  | Genel sınıflandırma kalitesi karşılaştırması |
| F1-Score (Macro)  | Her iki sınıf için dengeli değerlendirme |
| Sensitivite (Recall)  | Patojenik varyantları kaçırmama kalitesi – klinik öncelik |
| Özgüllük (Specificity)  | Benign varyantları doğru tanımlama |
| Balanced Accuracy  | Dengesiz test setinde genel doğruluk |
| MCC (Matthews Corr. Coeff.)  | İkili sınıflandırmada en dengeli tek metrik |

Karar eşiği (threshold) seçimi: Varsayılan 0.5 yerine, doğrulama setinde F1-macro'yu  maksimize eden eşik belirlenir. Klinik risk perspektifi doğrultusunda Sensitivite ≥ 0.90  kısıtı altında Özgüllük maksimizasyonu da değerlendirilir. Sonuçlar hem genel hem  dört panel için ayrı ayrı raporlanır.

**Hata Analizi ve Model Davranışı** 

Yanlış sınıflanan örnekler (False Negative ve False Positive) üç eksende incelenmektedir:

* Popülasyon frekansı ekseninde: Nadir varyantlar (MAF \< 0.001) vs. yaygın varyantlarda  hata dağılımı karşılaştırılır. Nadir varyantlar için modelin belirsizlik yüksekse bu durum  raporlanır.   
* In-silico skor kombinasyonlarında: Farklı araçların çelişkili tahmin ürettiği (örn. yüksek  SIFT – düşük PolyPhen) örneklerde hata yoğunlaşması beklenir; bu örnekler için özellik  ağırlıklarının tutarsız davranışı SHAP ile incelenir.   
* Biyokimyasal değişim türünde: Hidrofobik → şarjlı aminoasit değişimlerinin oluşturduğu  varyantlarda hata oranı, muhafazakâr değişimlerle karşılaştırılır. Bu hata örüntülerinin panel veri setleri arasında farklılık gösterip göstermediği (hastalık  bağlamı etkisi) de raporlanmaktadır.

**“Model Neden Böyle Karar Verdi?” – Açıklanabilirlik Yaklaşımı**

Açıklanabilirlik için SHAP (SHapley Additive exPlanations) kullanılmaktadır \[11\]. Tree based modeller için TreeSHAP uygulaması, hem küresel (tüm veri seti) hem yerel (tek  örnek) önem analizini hesaplama açısından verimli biçimde sunar. SHAP değerleri,  katsayıları yorum gerektiren lojistik regresyonun aksine, model yapısından bağımsız  teorik güvencelere sahiptir. 

Kolon isimleri paylaşılmadığından açıklama dili özellik grupları üzerinden kurulmuştur: 

* Evrimsel korunmuşluk grubu: SHAP beeswarm grafiğinde en yüksek |SHAP| değerini  taşıyan grup olması beklenir; yüksek korunmuşluk değerleri patojenik yönde katkı verir. Popülasyon frekansı grubu: Düşük allel frekansı (nadir varyant) patojenisite kararını  pozitif yönde etkiler; SHAP katkısı monoton azalan bir ilişki gösterir. In-silico skor grubu: Araçlar arası tutarlı "zararlı" tahminlerin modele kümülatif SHAP  katkısı en yüksek ikinci grup olması beklenmektedir.   
* Biyokimyasal/yapısal etki grubu: Radikal aminoasit değişimleri, panel bazında  farklılaşan SHAP katkısı gösterebilir.   
* Her panel için SHAP feature importance sıralama grafiği ve 3–5 hatalı sınıflandırma örneği  üzerinde yerel SHAP waterfall açıklaması sunulacaktır.

**Öğrenme Süreci ve Teknik Evrim** 

Veri seti teslim edildiğinde karşılaşılması öngörülen teknik zorluklar ve bu zorluklar için  önceden hazırladığımız müdahale planı aşağıda "olası sorun – planlanan müdahale – beklenen etki" akışı biçiminde sunulmuştur:

* Overfitting riski: Özellikle küçük panel veri setlerinde eğitim ve doğrulama metrikleri  arasında belirgin açılma beklenmektedir. Planlanan müdahale: num\_leaves ve  min\_child\_samples sınırlandırması, L1/L2 düzenleştirme katsayılarının Optuna ile  optimizasyonu ve erken durdurma (50 round sabır). Beklenti: doğrulama F1 ile eğitim F1  arasındaki farkın 0.05 sınırı altında tutulması.  
* Panel bazlı genelleme sorunu: Genel modelin küçük hastalık panellerine doğrudan  aktarımının yetersiz kalması olasıdır. Planlanan müdahale: Her panel için ayrı fine-tuned  model eğitimi ve genel modelin olasılık çıktılarının panel modeline meta-özellik olarak  aktarıldığı iki aşamalı yaklaşım. Beklenti: panel bazında F1 değerlerinde anlamlı iyileşme. Olasılık kalibrasyonu: Ham model çıktılarının güvenilir olasılıklar üretmeyebileceği  bilinmektedir. Planlanan müdahale: Isotonic Regression ile post-hoc kalibrasyon;  kalibrasyon kalitesi reliability diagram ve Brier score ile izlenecektir. Beklenti: eşik  seçimini doğrudan etkileyen olasılık güvenilirliğinin artması.   
* Eşik seçimi ve yanlış negatif riski: Asimetrik sınıf dağılımında varsayılan 0.5 eşiğinin  yüksek yanlış negatif oranına yol açması beklenmektedir. Planlanan müdahale: Klinik  öncelikli eşik optimizasyonu (Recall ≥ 0.90 kısıtı altında F1 maksimizasyonu). Beklenti:  klinik açıdan kritik patojenik varyantların gözden kaçırılma oranının tolere edilebilir sınırın  altında tutulması.

**YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK**  

**Neden Bu Algoritma / Mimari?**

LightGBM temelli ensemble yaklaşımı dört temel gerekçeyle seçilmiştir: (1) Veri  yapısına uygunluk: Genomik in-silico skor profilleri yüksek boyutlu, eksik değer  içeren ve çarpık dağılımlı tabular verilerdir; GBDT bu koşullarda normalleştirme  olmaksızın çalışır. (2) Küçük örneklemde genelleme: Panel veri setleri az örneklem  içerebilir; GBDT'nin düzenleştirme mekanizmaları overfitting'i derin ağlara göre çok  daha kontrollü biçimde yönetir. (3) Açıklanabilirlik: TreeSHAP ile her kararın özellik  grubuna katkısı şeffaf biçimde raporlanabilir; bu, klinik kullanım için zorunludur.  (4) Hesaplama verimliliği: 4 model \+ cross-validation döngüsü standart bir CPU  makinede saatler içinde tamamlanabilir; büyük GPU kümesi gerektirmez. 

**Alternatifler Neden Elendi?** 

| Derin Sinir Ağı (MLP /  TabNet) | Küçük panel veri setlerinde aşırı parametre sayısı nedeniyle yüksek  overfitting riski; hiperparametre hassasiyeti; açıklanabilirlik zorluğu |
| :---- | :---- |
| Protein Dil Modelleri  (ESM-2, AlphaFold) | Ham sekans verisi gerektiriyor; sağlanan veri seti sayısal profil  formatında; hesaplama maliyeti çok yüksek; fine-tuning veri miktarı  yetersiz |
| Random Forest  | Literatürde LightGBM ile karşılaştırmalı çalışmalarda tutarlı  biçimde düşük AUC; eksik değerleri varsayılan olarak  desteklemiyor; histogram tabanlı bölme olmadığı için daha yavaş |

**Parametre Seçimi ve Model Ayarları** 

Hiperparametre optimizasyonu için Optuna kütüphanesi ile Bayesian optimizasyon  kullanılmaktadır.  
Arama uzayı: num\_leaves (15–255), learning\_rate (0.01–0.3), min\_child\_samples (10– 100), reg\_lambda (0–5), subsample (0.6–1.0), colsample\_bytree (0.6–1.0). Optimizasyon  hedefi doğrulama seti PR-AUC'tür; 50–100 deneme gerçekleştirilmektedir. Early stopping: Her deneyde 50 round sabırla izlenir; doğrulama kaybı iyileşmezse eğitim  sonlandırılır. Bu hem overfitting'i hem de gereksiz eğitim süresini önler. Karar eşiği  (threshold) ayrı bir optimizasyon adımında belirlenmekte; doğrulama seti üzerinde F1- macro ve klinik Recall kısıtı birlikte değerlendirilerek nihai eşik seçilmektedir. 

Olasılık kalibrasyonu (Isotonic Regression) eğitim kümesinin %20'lik kalibrasyona  ayrılmış bölümü üzerinde eğitilir; test seti bu aşamada kullanılmaz. 

Tüm kod Git ile sürümlenecek; bağımlılıklar requirements.txt ve conda  environment.yml dosyalarıyla sabitlenecektir. Hedef, herhangi bir standart Python  ortamında 10 dakika içinde tam pipeline'ın çalıştırılabilmesidir.

**Hesaplama Kaynakları ve Çalıştırılabilirlik** 

| Bileşen Detay |  |
| :---- | :---- |
| İşlemci  | CPU: 8 çekirdek (Intel/AMD), GPU gereksinimi yok |
| Bellek  | 16 GB RAM (büyük veri seti için); panel setleri için 4 GB yeterli |
| İşletim Sistemi  | Ubuntu 22.04 LTS / Windows 11 (WSL2) |
| Dil ve Framework  | Python 3.11, LightGBM 4.x, XGBoost 2.x, Scikit-learn 1.4.x,  SHAP 0.45.x, Optuna 3.x |
| Eğitim süresi (genel set)  | \~15–30 dk (hiperparametre araması dahil, 50 Optuna denemesi) |
| Eğitim süresi (panel setleri)  | \~2–5 dk (küçük örneklem) |
| Tek örnek çıkarım süresi  | \< 5 ms |
| Toplu değerlendirme  | 1000 varyant \< 1 saniye |
| Seed / Determinizm  | random\_state=42 tüm adımlarda; sonuçlar tam tekrarlanabilir |

**Özgünlük** 

Yaklaşımımızın teknik özgünlüğü, planlanan dört somut katkı üzerinden özetlenebilir: 

1. Kolon isimsiz özellik gruplaması: Kolon adı bulunmayan veri setinde SHAP tabanlı  kümeleme ile özelliklerin biyolojik anlam gruplarına (korunmuşluk, frekans, in-silico skor,  biyokimyasal etki) otomatik atanması planlanmaktadır. Bu gruplama, klinik yorumu  standardize edecek ve açıklanabilirlik raporunu anlamlı kılacaktır.   
2. Asimetrik dağılıma özel kalibrasyon \+ eşik pipeline'ı: Eğitim setindeki patojenik  baskınlığının (\~%80) test setinde tersine dönmesi nedeniyle olasılık kalibrasyonu ve eşik  optimizasyonunun birlikte uygulandığı özel bir değerlendirme düzeni tasarlanmıştır. Bu  kombinasyonun etkisi ablasyon çalışmaları ile ölçülecektir. Panel-aware meta-learning: Genel modelin yumuşak tahminlerinin (soft predictions)  panel modellerine meta-özellik olarak aktarıldığı iki aşamalı transfer yaklaşımı  planlanmaktadır. Panel bazında performans katkısı 5-Fold çapraz doğrulama ile  sistematik biçimde değerlendirilecektir.   
3. Klinik risk odaklı değerlendirme çerçevesi: Tıp üyesinin rehberliğinde, yanlış negatif  maliyetinin yanlış pozitiften belirgin biçimde yüksek olduğu klinik bağlamda eşik seçim  gerekçesi raporlanacak; bu perspektif metrik yorumuna doğrudan yansıtılacaktır.

**REFERANSLAR** 

1. N. Alirezaie, K. D. Kerr, C. Hartley, et al., "ClinPred: Prediction Tool to Identify Disease Relevant Nonsynonymous Single-Nucleotide Variants," Am. J. Hum. Genet., vol. 103, no.  4, pp. 474–483, 2018\.   
2. N. M. Ioannidis, J. H. Rothstein, V. Pejaver, et al., "REVEL: An Ensemble Method for  Predicting the Pathogenicity of Rare Missense Variants," Am. J. Hum. Genet., vol. 99, no.  4, pp. 877–885, 2016\.   
3. L. Sundaram, A. J. Gloudemans, H. Bhatt, et al., "Predicting the clinical impact of human  mutation with deep evolutionary learning," in Nature Genetics, 2024\. \[Online\]. Available:  https://doi.org/10.1038/s41588-024-01820-z   
4. T. Shen, J. T. Hu, J. Wang, et al., "CardioBoost: a machine learning-based tool for the  classification of variants in inherited cardiac conditions," Genet. Med., vol. 23, no. 7, pp.  1360–1366, 2021\.   
5. J. Frazer, P. Notin, M. Dias, et al., "Disease variant prediction with deep generative models  of evolutionary data," Nature, vol. 599, no. 7883, pp. 91–95, 2021\.   
6. S. M. Tavtigian, M. S. Greenblatt, B. C. Harrison, et al., "Modeling the ACMG/AMP Variant  Classification Guidelines as a Bayesian Classification Framework," Genet. Med., vol. 20,  pp. 1054–1060, 2018\.   
7. S. Richards, N. Aziz, S. Bale, et al., "Standards and guidelines for the interpretation of  sequence variants," Genet. Med., vol. 17, no. 5, pp. 405–424, 2015\.   
8. G. Ke, Q. Meng, T. Finley, et al., "LightGBM: A Highly Efficient Gradient Boosting Decision  Tree," in Proc. NeurIPS, pp. 3146–3154, 2017\.   
9. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. KDD, pp.  785–794, 2016\.   
10. J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves,"  in Proc. ICML, pp. 233–240, 2006\.   
11. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions,"  in Proc. NeurIPS, pp. 4765–4774, 2017\.