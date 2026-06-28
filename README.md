# papilledema classification using radiomic features
Papilledema classification using radiomic features and a machine learning pipeline. It includes MRMR feature selection, Optuna hyperparameter optimization, patient-level cross-validation, and a soft voting ensemble.

Not: Bu repo şu anda düzenleme aşamasındadır. İçerik yakında eklenecektir.

## GİRİŞ

Papilödem, optik disk ödeminin şişmesi olarak tanımlanan kritik bir klinik bulgudur (Rinehart ve ark., 2010). Rinehart ve ark. (2010) tarafından yazılan eserde vurgulandığı gibi, bu bulgu artan kafa içi basıncının önemli bir göstergesi olarak hizmet eder: "Papilödem, artan kafa içi basıncının ayırt edici bir belirtisidir."

Radyomik, tıbbi görüntüleme analizinde hızla gelişen ve radyoloji ile onkoloji arasındaki boşluğu dolduran bir alandır (The Fast Future Executive ve ark., 2024). Standart tıbbi görüntülerden (BT, MR veya PET taramaları gibi) çok sayıda kantitatif özelliğin yüksek verimli bir şekilde çıkarılmasını içerir ve bu görüntüleri işlenebilir, yüksek boyutlu verilere dönüştürür (Neustein ve Christen, 2022).

Önemli bir gelişim alanı, radyomiklerin genomik verilerle entegrasyonudur; bu hibrit alan radyogenomik olarak bilinmektedir (Neustein ve Christen, 2022; Lilhore ve ark., 2025). Araştırmacılar, bu yöntemlerin birleştirilmesinin kişiselleştirilmiş tıbbı nasıl ilerlettiğini vurgulamaktadır:

> "Bu yaklaşım, her iki veri türünün güçlü yönlerinden yararlanır: görüntüleme verilerinin uzamsal ve zamansal çözünürlüğü ile genomik verilerin moleküler özgüllüğü. Entegre veri kümeleri üzerinde modeller eğitilerek, araştırmacılar bu veri kümelerinden herhangi birine dayalı olanları aşan tanı araçları geliştirmiştir; bu durum, erken teşhis ve kişiselleştirilmiş tedavi planları üzerinde büyük bir etkiye sahiptir" (Lilhore ve ark., 2025).

Bu çalışmada, optik disk görüntülerinden çıkarılan radyomik özellikleri kullanarak papilödem ile normal vakaları sınıflandırmak için kapsamlı bir makine öğrenimi işlem hattı geliştirilmiştir. İşlem hattı, veri ön işleme, MRMR (Maksimum İlgi Minimum Fazlalık) tabanlı özellik seçimi, Optuna hiperparametre optimizasyonu, hasta düzeyinde 20 tekrarlı bölünmüş çapraz doğrulama, sigmoid olasılık kalibrasyonu ve yumuşak oylama topluluk modelini içermektedir. Altı sınıflandırıcı değerlendirilmiştir: Lojistik Regresyon (LR), RBF çekirdekli Destek Vektör Makinesi (SVM), Rastgele Orman (RF), Ekstra Ağaçlar (ET), Gradyan Artırma (GB) ve K-En Yakın Komşu (KNN).

Bu çalışmanın temel tasarım ilkesi, veri sızıntısının kesinlikle önlenmesidir — tüm ön işleme ve özellik seçimi adımları yalnızca eğitim verileri üzerinde uygulanır ve hasta düzeyindeki bölünme, aynı hastaya ait örneklerin farklı alt kümelerde aynı anda yer almamasını sağlar.

## VERİ SETİ

Bu çalışma için iki adet CSV dosyası sağlanmıştır: normal bireylerden alınan örnekleri içeren normal_radiomics.csv ve papilödem tanısı konmuş hastalardan alınan örnekleri içeren papilodem_radiomics.csv dosyaları. Her örnek, optik disk görüntülemesinden çıkarılan tek bir ROI (İlgi Bölgesi - Region of Interest) çerçeve bileşenini temsil etmektedir.

Veri seti, sağlıklı gözler ile optik sinir başında şişmeye neden olan durumları ayırt etmek için makine öğrenimi modellerini eğitmek ve değerlendirmek üzere özel olarak tasarlanmıştır. Bu veri setleri birleştirilmiş ve sonsuz (infinity) değerler işlenmiştir.

**Veri yükleme**

```python
# Veri yükleme
normal = pd.read_csv('normal_radiomics.csv')
papilodem = pd.read_csv('papilodem_radiomics.csv')

normal['label'] = 0
papilodem['label'] = 1

# Birleştirme
df = pd.concat([normal, papilodem], ignore_index=True)

# Sonsuz değerleri düzeltme
feature_cols = [col for col in df.columns if col.startswith('Feature_')]
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Hasta listesi
patients = df['PatientIndex'].unique()

print("Veri seti boyutu:", df.shape)
print("Toplam hasta sayısı:", len(patients))
print("Özellik sayısı:", len(feature_cols))
print("Sınıf dağılımı:\n", df['label'].value_counts())
```

**Çıktı**
```text
Veri seti boyutu: (966, 749)
Toplam hasta sayısı: 48
Özellik sayısı: 746
Sınıf dağılımı:
label
0    672
1    294
Name: count, dtype: int64
```

Veri seti, her biri 746 radyomik özelliğe sahip 966 örnekten oluşmaktadır. Veri setinde 48 farklı hasta bulunmakta olup, her hasta hem sağ hem de sol gözden birden fazla örnek katkısında bulunmaktadır. Tek bir hastanın düzinelerce satıra sahip olabileceği bu çoklu örnek yapısı, standart rastgele bölünmeyi uygunsuz kılmakta ve hasta düzeyinde bölünme stratejisini zorunlu hale getirmektedir.

Şekil: Veri seti önizlemesi

<img width="1719" height="675" alt="image" src="https://github.com/user-attachments/assets/89e5677e-e686-4587-a780-d6d6c7a6881c" />

Sınıf dağılımı orta düzeyde bir dengesizlik göstermektedir: 672 örnek Normal sınıfına (%69,6) ve 294 örnek Papilödem sınıfına (%30,4) aittir. Bu dengesizlik, her iki sınıfı da frekanslarından bağımsız olarak eşit şekilde değerlendiren Macro-F1'in birincil değerlendirme metriği olarak kullanılmasıyla ele alınmıştır.

Bu veri setinin keşfedilen önemli bir özelliği, tek bir hastanın her iki sınıftan da örneklere sahip olabilmesidir; örneğin, bir göz normal iken diğer göz papilödem belirtileri gösterebilir. Bu durum, örnek düzeyinde bölünme yerine hasta düzeyinde veri bölümlendirmesini daha da güçlendirmektedir.

## YÖNTEM

Veri sızıntısını önlemek için tasarlanmış titiz bir makine öğrenimi işlem hattı, test verilerinden gelen bilgilerin eğitim sürecini istemeden etkilememesini sağlayan sistematik bir yaklaşımdır. Cuantum Technologies LLC (2025) tarafından belirtildiği gibi: "Veri Sızıntısının Önlenmesi: İşlem hatları, veri dönüşümlerinin eğitim ve test veri kümelerine tutarlı bir şekilde uygulanmasını sağlayarak test setindeki bilgilerin eğitim sürecini istemeden etkilemesini önler."

Temel ilke, tüm iş akışı boyunca eğitim ve test verileri arasında katı bir ayrımın korunmasını içerir. Gonzalez ve Stubberfield (2024) tarafından belirtildiği gibi: "Bu, birçok acemi veri bilimcinin eğitim setini test setinden ayırmadan önce tüm veri kümesine veri dönüşümleri ve ön işleme uygulamasıyla yaptığı bir hatadır. Bu, yüksek yanlılığa ve aşırı iyimser model performansına yol açabilir."

Tekrarlanan hasta düzeyinde dış bölünmeler kavramı, rastgele veri bölümlendirmesinin doğasında bulunan değişkenliği ele almaktadır. Fenner (2019) tarafından belirtildiği gibi: "Rastgeleliğe güvendiğimiz her zaman, değişkenliğe tabi oluruz: birkaç farklı eğitim-test bölünmesi farklı sonuçlar verebilir. Bunlardan bazıları harika, bazıları ise felaket olabilir."

Çözüm, temel eğitim-test bölünmesi sürecinin farklı rastgele tohumlarla birden çok kez (bu çalışmada 20 kez) tekrarlanmasını içerir. Bu yaklaşım, birçok bölünme yaparak ve farklı sonuçları inceleyerek eğitim-test bölünmelerinden kaynaklanan değişimi araştırmaya yardımcı olur ve daha sağlam bir değerlendirme çerçevesi sağlar (Fenner, 2019).

**Dış döngü**

```python
outer_splits = []
for seed in range(20):
    p_trainval, p_test = train_test_split(patients, test_size=0.2, random_state=seed)
    p_train, p_val = train_test_split(p_trainval, test_size=0.125, random_state=seed)
    outer_splits.append({
        'train patients': p_train,
        'val patients': p_val,
        'test patients': p_test,
        'seed': seed
    })
```

Çapraz doğrulama katmanları içinde, bağımsız model seçimi ve hiperparametre optimizasyonu birkaç kritik amaca hizmet eder. Raschka, Liu ve Mirjalili (2022) tarafından açıklandığı gibi: "Tipik olarak, model ayarlaması için k-katlı çapraz doğrulama kullanırız; yani, test katmanlarında model performansının değerlendirilmesinden tahmin edilen, tatmin edici genelleme performansı sağlayan optimal hiperparametre değerlerini bulmak için."

İç içe çapraz doğrulama yaklaşımı şunları içerir:

1. Verileri eğitim ve test katmanlarına bölmek için bir dış k-katlı çapraz doğrulama döngüsü
2. Eğitim katmanı üzerinde k-katlı çapraz doğrulama kullanarak model seçimi için bir iç döngü
3. Seçimden sonra model performansını değerlendirmek için test katmanını kullanma (Raschka, Liu ve Mirjalili, 2022)

Bu yöntem, hiperparametre seçiminin yalnızca eğitim verilerine dayanmasını sağlayarak test setlerinden bilgi sızıntısını önler. Witten ve ark. (2016) tarafından belirtildiği gibi: "Hiperparametre seçimi yalnızca eğitim setine dayanmalıdır. Yukarıdaki parametre seçim sürecini çapraz doğrulama içinde, her katman için bir kez olmak üzere birden çok kez uygularken, hiperparametre değerlerinin katmandan katmana biraz farklı olması tamamen olasıdır."

Modern makine öğrenimi işlem hatları, temel veri sızıntısı önlemenin ötesinde ek avantajlar sunar. Cuantum Technologies LLC (2025) tarafından belirtildiği gibi, bu işlem hatları içindeki çapraz doğrulama şunları sağlamaktadır: "Aşırı Uyumun Azaltılması: Modelin verilerin farklı alt kümelerinde değerlendirilmesiyle, çapraz doğrulama eğitim verilerinin belirli bir alt kümesine aşırı uyum riskini azaltmaya yardımcı olur. Güvenilir Performans Tahminleri: Tüm katmanlardaki ortalama performans, modelin görülmemiş veriler üzerinde nasıl performans gösterme olasılığının daha istikrarlı ve güvenilir bir tahminini sağlar."

Veri sızıntısı, doğrulama veya test veri kümelerinden gelen bilgilerin bir modelin eğitim aşamasında istemeden kullanılabilir hale gelmesi durumunda ortaya çıkar (Hall, Curtis ve Pandey, 2023; Huyen, 2022). Tıbbi veri kümelerinde, tek bir hastanın birden fazla veri noktasına sahip olması oldukça yaygındır; örneğin, birkaç röntgen görüntüsü, BT dilimi veya boylamsal klinik kayıt gibi.

Bu tür bir veri setini hasta yerine rastgele örnek bazında (örneğin, görüntü veya kayıt numarasına göre) bölerseniz, aynı hastadan gelen farklı örnekler kaçınılmaz olarak eğitim, doğrulama ve test kümelerine dağılacaktır (Kuligin, 2026; Hall, Curtis ve Pandey, 2023). Kuligin (2026) tarafından belirtildiği gibi:

"Aynı hastanın aynı taraması hem eğitim veri setinde hem de değerlendirme veri setinde bulunduğundan, model eğitim aşamasında belirli bir hastanın gerçek teşhisini 'gördü'. Bu, modelin hangi tanının hangi kalıplarla ilgili olduğunu gerçekten öğrenmek yerine her hasta için doğru tanıyı ezberlemesine izin verdi. Sonuç olarak, araştırmacılar değerlendirme sırasında iyi bir performans elde etti ancak model üretimde başarısız oldu."

İşlem hattı aşağıdaki sıralı aşamalardan oluşmaktadır:

1. Veri yükleme ve ikili etiket ataması (Normal=0, Papilödem=1)
2. 20 tekrarlı hasta düzeyinde bölünme (%70 eğitim / %10 doğrulama / %20 test)
3. Her bir dış bölünme içinde - her model için:
 *  Optuna hiperparametre optimizasyonu ile iç çapraz doğrulama
 *  Yalnızca iç eğitim katmanında uygulanan ön işleme
 *  Yalnızca iç eğitim katmanında MRMR özellik seçimi
 *  Model eğitimi ve iç doğrulama değerlendirmesi (Macro-F1)
4. En iyi konfigürasyonun tam eğitim seti üzerinde yeniden eğitilmesi
5. Eğitim seti üzerinde sigmoid kalibrasyonu
6. Doğrulama setinin eşik optimizasyonu ve toplama stratejisi seçimi için kullanılması
7. Eğitim + doğrulama birleştirilmesi, son modelin yeniden eğitilmesi
8. Bağımsız test seti değerlendirmesi
9. 20 bölünme boyunca sonuçların toplanması
10. Yumuşak oylama topluluğu (RF + ET + GB)
11. Özellik kararlılık analizi
12. İstatistiksel karşılaştırma (Friedman, Wilcoxon, Bonferroni)

Doğrulama seti (%10 hasta), model seçiminin ötesinde iki amaca hizmet etmektedir. İlk olarak, sınıfa özgü olasılık eşikleri, Macro-F1'i maksimize ederek 0,05-0,95 aralığında 0,01 adımlarla taranarak doğrulama seti üzerinde optimize edilir. İkinci olarak, hasta düzeyinde toplama stratejileri değerlendirilir ve en iyi strateji, birincil kriter hasta düzeyinde Macro-F1 ve ikincil kriter dengeli doğruluk esas alınarak seçilir. Değerlendirilen toplama stratejileri arasında ortalama olasılık, maksimum olasılık, çoğunluk oylaması, ilk üç ortalaması, P90, güven ağırlıklı ve entropi ağırlıklı toplama yer almaktadır.

Doğrulama setinde eşik ve toplama stratejisi seçiminin ardından eğitim ve doğrulama setleri birleştirilir. Nihai model, optimizasyon sırasında belirlenen en iyi hiperparametreler ve MRMR konfigürasyonu kullanılarak bu birleşik set üzerinde yeniden eğitilir. Test seti değerlendirmesinden önce nihai modele sigmoid kalibrasyonu uygulanır.
