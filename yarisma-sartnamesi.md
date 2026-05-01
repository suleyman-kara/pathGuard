**YARIŞMA**

Bilinen az sayıdaki varyantın biyolojik ve hesaplamalı özelliklerini kullanarak, klinik  durumu bilinmeyen varyantların "Patojenik (hastalık yapıcı)" veya "Benign (iyi huylu)"  olma durumuna yönelik tahmin modelleri geliştirecektir. 

Yarışmacıların geliştirecekleri modeller ile bilinen az sayıdaki varyantın biyolojik  özelliklerinden yola çıkıp bilinmeyenler üzerinde tahmin yürüterek literatüre katkı  sunulması amaçlanmaktadır.

TEKNOFEST 2026 kapsamında, takımlar modellerini, belirlenen problemler ile ilgili  verilen veri setleri ile birlikte açık veri kaynaklarını da kullanarak geliştirecektir. Geliştirilen  modeller, yarışma sırasında yeni verilerle test edilerek eksternal validasyon (dış  doğrulama) süreci gerçekleştirilecektir. Klinik uygulamalara geçişte kritik bir aşama olan  eksternal validasyon, takımların çeşitli veriler kullanarak geliştirdikleri modellerin yeni verilerde ne kadar etkili olduğunu ortaya koyması için önemlidir.

**KAPSAM**  

Veri setlerinde yer alan varyantların Patojenik veya Benign olarak sınıflandırılmasında,  Amerikan Tıbbi Genetik ve Genomik Koleji (ACMG) rehberleri ve kriterleri referans  alınmıştır. Sınıflandırma sürecinde, kaynak veri tabanlarında bulunan ACMG uyumlu  mevcut etiketler referans kabul edilerek veri setine dahil edilmiştir. Bu etiketler, yarışma  kapsamında 'Doğru Sınıf' (Ground Truth) olarak kullanılacaktır. Yarışmacılara sağlanacak  veri setleri aşağıdaki standartlarda hazırlanmıştır:


Patojenik Sınıf: ClinVar ve ClinGen veri tabanlarından, "Expert Panel" ve güvenilir  "Practice Guideline" inceleme statüsüne sahip, 3 ve 4 yıldız güvenilirlik seviyesindeki  missense varyantlar seçilmiştir. Veri setlerindeki 'Pathogenic' (Patojenik) ve 'Likely  Pathogenic' (Olası Patojenik) olarak tanımlanan varyantlar tek bir Patojenik sınıfı altında  birleştirilmiştir (2909 varyant).

Benign Sınıf: Sınıf dengesizliğini gidermek amacıyla ClinVar (1381 varyant) verilerine ek  olarak gnomAD veri tabanından ClinVar datasındaki genlerin sık görülen sağlıklı  popülasyon varyantları eklenecektir (∼ 1500 varyant). Veri setlerindeki 'Benign' (İyi Huylu)  ve 'Likely Benign' (Olası İyi Huylu) olarak tanımlanan varyantlar tek bir Benign sınıfı altında  birleştirilmiştir.

Yarışmacılara dört ana başlık altında veri setleri sunulacaktır. Veri setlerinin  oluşturulmasında, yapay zekâ modellerinin eğitim süreçlerinde yanlılığı engellemek adına veri dengesi gözetilmiştir. Bu kapsamda, her bir veri setindeki Benign varyant sayısı,  Patojenik varyant sayısına yakın tutularak dengeli bir yapı oluşturulmuştur.


Sağlanacak eğitim veri setlerinin içerdiği varyant sayıları aşağıdaki şekildedir:

* Genel Veri Seti: 1500 adet Patojenik varyant, 1500 adet Benign Varyant  
* Kalıtsal (Herediter) Kanser Paneli: 200 adet Patojenik varyant, 200 adet Benign varyant  
* Fenilketonüri (PAH) Gen Paneli: 200 adet Patojenik varyant, 200 adet Benign varyant  
* Kistik Fibrozis (CFTR) Gen Paneli: 70 adet Patojenik varyant, 70 adet Benign varyant.


Sağlanacak test veri setlerinin içerdiği varyant sayıları aşağıdaki şekildedir:

* Genel Veri Seti: 1000 adet Patojenik varyant, 1000 adet Benign varyant,  
* Kalıtsal (Herediter) Kanser Paneli: 100 adet Patojenik varyant, 100 adet Benign  varyant  
* Fenilketonüri (PAH) Gen Paneli: 100 adet Patojenik varyant, 100 adet Benign varyant  
* Kistik Fibrozis (CFTR) Gen Paneli: 30 adet Patojenik varyant, 30 adet Benign varyant.

Toplam veri havuzu, model geliştirme süreçlerinin sağlıklı yürütülmesi adına ikiye  ayrılmıştır: 

Eğitim Seti: Proje Sunuş Raporu aşamasını geçen yarışmacılara, modellerini eğitmeleri  için etiketli olarak verilecek olan eğitim setidir. 

Test Seti: Geliştirilen modellerin nihai başarımını değerlendirmek amacıyla ayrılan bu set,  yarışma esnasında sınıf etiketleri gizlenmiş (etiketsiz) olarak paylaşılacaktır.   

Yarışma veri setinde varyantların genomik adres (kromozom ve pozisyon) bilgileri,  katılımcıların dış veri kaynaklarına başvurarak etiketi doğrudan bulmalarını engellemek  amacıyla tamamen gizlenmiştir. Bu kısıtlamanın amacı; yarışmacıların patojenite  tahminlerini harici veri kaynaklarına başvurmaksızın, yalnızca yarışma komitesi  tarafından sağlanan varyant profilleri üzerinden yapmalarını sağlamak ve kamuya açık  veri tabanlarından elde edilebilecek hazır etiket bilgisinin kullanımını engellemektir. 

Modellerin yüksek başarımla eğitilebilmesi amacıyla, genomik adres (kromozom ve  pozisyon) bilgisi yerine biyoinformatik araçlarla zenginleştirilmiş, kapsamlı ve çok boyutlu  varyant profilleri sunulacaktır. Her bir varyant için sağlanan öznitelik seti aşağıdaki  detayları kapsamaktadır, veri paylaşılırken öznitelik kolon isimleri verilmeyecektir:

* Sekans ve Değişim Bilgisi: Varyantın gerçekleştiği noktadaki referans ve alternatif  nükleotid bilgisi, kodon değişimi ve bu değişimin yol açtığı amino asit dönüşümü (örneğin; Alanin'den Valin'e dönüşüm).  
* Yerel Sekans ve Çevresel Bağlam Bilgisi: Varyantın bulunduğu bölgenin yapısal  özelliklerinin ve yerel motiflerin model tarafından öğrenilebilmesi amacıyla;  varyant noktasının öncesindeki ve sonrasındaki 5 nükleotid (genomik komşuluk)  ile protein düzeyinde ilgili amino asidin öncesindeki ve sonrasındaki 5 amino asit  (proteomik komşuluk) bilgisi sağlanacaktır.  
* Biyokimyasal ve Yapısal Etkiler: Meydana gelen amino asit değişiminin proteinin  fizikokimyasal özelliklerine (hidrofobiklik, polarite, moleküler ağırlık değişimi vb.)  ve 3 boyutlu yapısına olası etkileri.  
* Evrimsel Korunmuşluk: Varyantın bulunduğu gen bölgesinin, farklı canlı türleri  (filogenetik çeşitlilik) ve insan toplulukları arasındaki genomik benzerliği; bu  bölgenin evrimsel süreçte ne kadar korunduğuna dair skorlar.  
* Popülasyon Verileri: Varyantın farklı insan popülasyonlarında görülme sıklıkları  (Minör Allel Frekansı vb.).  
* In Silico Risk Skorları: Farklı algoritmalar tarafından hesaplanmış, varyantın zararlı olma olasılığına dair hesaplamalı risk skorları.

Modellerin başarısı, test verisi üzerindeki tahminlerin gerçek etiketlerle  karşılaştırılmasıyla ölçülecektir. Yarışma sıralamasını belirleyecek temel metrik, TP  (Doğru Pozitif), FP (Yanlış Pozitif) ve FN (Yanlış Negatif) değerleri üzerinden hesaplanan  F1 Skoru olacaktır.