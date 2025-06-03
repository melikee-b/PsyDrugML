# Kişilik ve Madde Kullanımı: Veri Madenciliğiyle Açığa Çıkan İlişkiler

Bu proje, bireylerin demografik ve psikolojik özelliklerine dayanarak 18 farklı yasal ve yasadışı maddenin kullanım durumunu tahmin etmeyi amaçlamaktadır. Veri madenciliği teknikleriyle desteklenen modelleme sürecinde veri dengesizliği sorunlarına yönelik çözümler ve kapsamlı analizler yer almaktadır.

---

## 🔎 Proje Hakkında

- **Veri seti**: UCI Machine Learning Repository’den alınan *Drug Consumption (quantified)* veri seti.  
- **Kapsam**: 1885 birey, 12 psikolojik ve demografik özellik, 18 madde kullanım bilgisi.  
- **Amaç**: Çoklu madde kullanımı tahmini yapmak ve hangi özniteliklerin (kişilik özellikleri, demografi vb.) madde kullanımıyla en çok ilişkili olduğunu anlamak.

---

## ⚙️ Veri Ön İşleme

- Sütun adları Türkçe’ye çevrildi.  
- 7 sınıflı madde kullanım bilgileri (CL0–CL6), ikili (binary) sınıflara dönüştürüldü.  
- Cinsiyet ve kategorik değişkenler sayısal kodlara dönüştürüldü (örneğin, Cinsiyet: Erkek=1, Kadın=0).

---

## 📊 Keşifsel Veri Analizi

- Tüm maddeler için kullanım dağılımları grafikleri oluşturuldu.  
- Korelasyon matrisi çıkarılarak psikolojik ve demografik özelliklerin birbiriyle ilişkisi incelendi.

---

## 🌲 Modelleme ve Veri Dengesizlik Çözümleri

- Ana sınıflandırıcı olarak **Random Forest** (MultiOutputClassifier yapısıyla) kullanıldı.  
- Veri dengesizliğini azaltmak için **SMOTE (Synthetic Minority Over-sampling Technique)** uygulandı.  
- Ayrıca **class_weight='balanced'** parametresi ile modelin azınlık sınıflara daha fazla odaklanması sağlandı.

---

## 📈 Model Performansı

Model performansı, **SMOTE öncesi ve sonrası** ayrı ayrı değerlendirildi.  
Her madde için **precision, recall, F1-score ve accuracy** metrikleri hesaplandı.

|                                | Accuracy | Precision | Recall  | F1-Score |
|--------------------------------|----------|-----------|---------|----------|
| **SMOTE Öncesi**               | 0.8100   | 0.6725    | 0.5554  | 0.5717   |
| **SMOTE + class_weight Sonrası**| 0.7994   | 0.6298    | 0.6141  | 0.6195   |

- **En yüksek başarı**: Alkol, esrar ve kokain gibi yaygın kullanılan maddelerde (F1-score ≈ 0.99).  
- **Sınırlı başarı**: Krack, eroin, uçucu madde gibi kullanıcı sayısı az olan maddelerde.  
- **Psikolojik öznitelikler**: Duyum arayışı ve dürtüsellik, en önemli tahminleyici değişkenler olarak öne çıktı.

---
## 📝 Sonuç ve Literatür Karşılaştırması

- Model, özellikle yaygın kullanılan maddelerde yüksek performans göstermiştir.  
- Literatürde yer alan Fehrman et al. (2015) çalışmasıyla bulgular güçlü biçimde örtüşmektedir.  
- SMOTE ve class_weight ayarı, recall ve F1-score gibi dengeli metriklerde iyileşme sağlamıştır.

---

## 🚀 Projeyi Çalıştırmak

Bu projeyi bir **Jupyter Notebook** veya **Google Colab** ortamında kolayca çalıştırabilirsiniz.

### Örnek Çalıştırma Adımları

1. 📦 **Gerekli kütüphaneleri yükleyin**:
   ```python
   pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
## 📁 CSV Veri Setini Yükleyin

`drug_consumption.csv` dosyasını projenizin çalışma dizinine ekleyin.

---

## 📝 Notebook Dosyasını Adım Adım Çalıştırın

- Veri yükleme ve ön işleme  
- Keşifsel analiz  
- Model eğitimi ve değerlendirmesi  
- Performans karşılaştırmaları ve sonuç değerlendirmeleri

## 📚 Kaynaklar

- Fehrman, E., Muhammad, A. K., Mirkes, E. M., Egan, V., & Gorban, A. N. (2015). *The Five Factor Model of Personality and Evaluation of Drug Consumption Risk*. arXiv preprint [arXiv:1503.01769](https://arxiv.org/abs/1503.01769).
- UCI Machine Learning Repository: *Drug Consumption (quantified)*. [UCI Dataset Link](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
## ✏️ Lisans

Bu proje **açık kaynaklı** olup, yalnızca akademik ve araştırma amaçlı kullanılabilir. Kullanım sırasında ilgili veri seti lisanslarına ve atıf gerekliliklerine dikkat edilmelidir.
