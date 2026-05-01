# PathGuard Yarışma Şartnamesi ve Rapor Karşılaştırma Analizi

Bu rapor, `yarisma-sartnamesi.md` ve takımınızın hazırladığı `yarisma-raporu.md` belgelerinin detaylı karşılaştırması sonucunda hazırlanmıştır. Takımınız (MEN-INA) teknik olarak çok güçlü ve klinik farkındalığı yüksek bir yaklaşım sergilemiş olsa da, **şartnamenin veri dağılımıyla ilgili kritik bir bölümü yanlış yorumlanmıştır**. 

Aşağıda neleri doğru yaptığınızı ve neleri acilen düzeltmeniz gerektiğini bulabilirsiniz.

## ✅ NELERİ DOĞRU YAPTINIZ? (Artılar ve Uyumluluk)

**1. Kısıtlamalara ve Veri Gizliliğine Tam Uyum:**
* Kromozom, pozisyon gibi genomik adreslerin verilmeyeceğini ve dış kaynaklardan etiket çekilmesinin yasak olduğunu doğru anlamışsınız. 
* Özelliklerin (kolon isimlerinin) gizli olacağını bilerek, 5 ana kategori (evrimsel, popülasyon, in-silico vb.) üzerinden özellikleri haritalama stratejiniz şartnameye mükemmel uyuyor.

**2. Etiketlerin Birleştirilmesi (Sınıflandırma Mantığı):**
* ACMG kriterlerine göre "Pathogenic" ve "Likely Pathogenic" sınıflarını **Patojenik**; "Benign" ve "Likely Benign" sınıflarını **Benign** olarak birleştirmeniz ve VUS (Uncertain Significance) varyantları eğitim dışında tutmanız şartnamede belirtilen "Doğru Sınıf (Ground Truth)" mantığı ile %100 örtüşüyor.

**3. Model ve Algoritma Seçimi (LightGBM & XGBoost):**
* Şartnamede tarif edilen veri tipi (yüksek boyutlu in-silico skorlar, eksik değer barındırabilen tabular veri) için Derin Öğrenme (Deep Learning) yerine GBDT (Gradient Boosting) algoritmalarını tercih etmeniz veri mühendisliği açısından en doğru teknik kararlardan biri olmuş.

**4. Panel Setlerine Özel Doğrulama (Cross-Validation):**
* Kistik Fibrozis (CFTR) gibi sadece 140 (70 Patojenik / 70 Benign) satırdan oluşan küçük panel veri setleri için Leave-One-Out (LOOCV) veya Repeated k-Fold gibi özel stratejiler kullanmanız aşırı öğrenmeyi (overfitting) engelleyecektir.

**5. Klinik Yaklaşım ve Açıklanabilirlik (SHAP):**
* Tıp üyesinin de katkısıyla SHAP değerleri üzerinden biyolojik açıklanabilirlik sunmanız, yarışma şartnamesinde doğrudan zorunlu kılınmasa da projenizin bilimsel değerini ve jüri gözündeki etkisini inanılmaz derecede artıracak bir hamledir.

---

## ❌ NELERİ YANLIŞ YAPTINIZ? (Hatalar ve Çelişkiler)

Raporunuzda yarışma performansınızı ciddi derecede etkileyebilecek bazı varsayım hataları bulunmaktadır. Bunları yarışma öncesi modelinize yansıtmanız gerekmektedir:

**1. Veri Dengesi ve Sınıf Dağılımı Yanılgısı (KRİTİK HATA!)**
* **Raporda Söylenen:** *"Eğitim seti ağırlıklı olarak patojenik (~%80), test seti ise ağırlıklı olarak benign (~%80) örneklerden oluşmaktadır. Bu asimetrik dağılım..."*
* **Şartnamede Yazan:** *"Yapay zekâ modellerinin eğitim süreçlerinde yanlılığı engellemek adına veri dengesi gözetilmiştir... her bir veri setindeki Benign varyant sayısı, Patojenik varyant sayısına eşit/yakın tutularak **dengeli bir yapı oluşturulmuştur**."*
* **Gerçek Dağılım:** Şartnamedeki rakamlar incelendiğinde Genel Set, Kanser Paneli, PAH ve CFTR setlerinin hem eğitim hem de test verilerinde sınıf oranları tam olarak **%50 Patojenik - %50 Benign** şeklindedir (Örn: 1500 P / 1500 B).
* **Neden Tehlikeli?:** Verinin dengesiz (%80'e %20) olduğunu zannederek kurduğunuz `scale_pos_weight` gibi dengeleme ayarları ve asimetrik kalibrasyon stratejileri, halihazırda mükemmel dengelenmiş (%50-%50) yarışma verisinde modelinizin başarısını (F1 skorunu) ciddi şekilde bozacaktır. Bu stratejileri iptal etmeli veya sadece gerçek veriyi gördükten sonra uygulamalısınız.

**2. Eğitim / Test Ayrımı Oranları:**
* **Raporda Söylenen:** *"Eğitim/test ayrımı yaklaşık %80 eğitim – %20 test oranında gerçekleştirilmiş"*
* **Gerçek Durum:** Şartnamedeki sayılara göre yarışmanın size vereceği Genel Set'te 3000 eğitim, 2000 test verisi var (Oran: **%60 Eğitim - %40 Test**). Bu oranı sizin değil organizasyonun önceden belirleyip böldüğünü unutmamalısınız.

**3. Birincil Optimizasyon Metriği Hedefi:**
* **Raporda Söylenen:** *"Birincil metrik olarak PR-AUC seçilmiştir."* ve klinik kaygılarla *"Sensitivite (Recall) >= 0.90"* kısıtı getirilerek eşik optimizasyonu yapılması hedeflenmiştir.
* **Şartnamede Yazan:** *"Yarışma sıralamasını belirleyecek temel metrik... F1 Skoru olacaktır."*
* **Neden Tehlikeli?:** Tıp perspektifinden yanlış negatif (hastalığı kaçırma) çok daha riskli olsa da, yarışma platformu algoritmaları bu klinik riskleri önemsemeden doğrudan test verisindeki **F1 Skoruna** göre takımları sıralayacaktır. Recall'ı (Duyarlılık) yapay olarak yüksek tutmak için eşiği düşürürseniz, FP (Yanlış Pozitif) sayınız artar ve F1 skorunuz düşebilir. Birincil ve tek yarışma hedefi olarak **F1 (Macro/Binary)** skoru maksimizasyonuna odaklanmalısınız.

### Özet ve Tavsiye
Metodolojiniz, takım dağılımınız ve algoritma tercihleriniz harika. Ancak, modellerinizi kurarken kodlarınızdaki veri dengesizliği (`class imbalance`) çözümlerini ve klinik Recall kısıtlamalarını **kaldırmalısınız**. Şartname size tamamen dengeli bir veri seti sunacağını ve yarışmayı yalnızca genel F1 skoru üzerinden değerlendireceğini taahhüt etmektedir.
