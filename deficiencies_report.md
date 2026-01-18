# Simülasyon Eksiklik Analizi Raporu

"Gerçeğe en yakın simülasyon" hedefi doğrultusunda yapılan kod incelemesi sonucunda aşağıdaki eksiklikler tespit edilmiştir.

## 1. Kritik Eksiklik: Ortam Uyuşmazlığı (Environment Mismatch)
**Durum:** Simülasyonun fizik motoru (`src/simulation/environment.py`) ve görsel motoru (`src/simulation/terrain.py`) birbirinden tamamen bağımsız iki farklı dünya oluşturmaktadır.
- **Fizik Dünyası:** Rastgele "hills" veya "valley" algoritmaları ile oluşturulan, rastgele binalar içeren bir yükseklik haritası (heightmap) kullanıyor.
- **Görsel Dünya:** Perlin noise ile oluşturulan, ortasında düz bir pist alanı olan, etrafında dağlar bulunan ve belirli konumlarda (Control Tower, Hangar vb.) binalar içeren bir dünya çiziyor.
**Sonuç:** İHA görsel olarak bir dağın içinden geçebilir (çünkü fizik dünyasında orada dağ yok) veya görsel olarak boş bir alanda görünmez bir engele çarpabilir (çünkü fizik dünyasında orada tepe veya bina var). Bu durum simülasyonun güvenilirliğini tamamen yok etmektedir.

## 2. Sensör Modellemeleri
**Durum:** İHA sensörleri "truth data" (gerçek veri) döndürmektedir.
**Eksiklik:**
- GPS gürültüsü (drift, random walk) yok.
- IMU (İvmeölçer/Jiroskop) bias ve gürültüleri modellenmemiş.
- Kamera görüntüleri mükemmel netlikte, lens distorsiyonu veya sensör gürültüsü (noise) yok.

## 3. Atmosfer ve Hava Durumu
**Durum:** Basit bir rüzgar vektör hesabı (`src/utils/physics.py`) mevcut ancak görsel karşılığı yok.
**Eksiklik:**
- Görsel olarak yağmur, kar, sis, bulut katmanları yok.
- Rüzgarın görsel etkileri (ağaçların sallanması vb.) yok.
- Gelişmiş aerodinamik etkiler (Türbülans bölgeleri, termaller, yer etkisi - ground effect detayları) sınırlı.

## 4. Ses
**Durum:** Simülasyonda hiç ses yok.
**Eksiklik:** Motor sesi, rüzgar sesi, çarpışma efektleri eksik.

## 5. Yapay Zeka ve Trafik
**Durum:** Düşman İHA'ları basit davranışlar sergiliyor.
**Eksiklik:** Karmaşık senaryolar (formasyon uçuşu, koordineli saldırı) için daha gelişmiş AI gerekli olabilir.

---
**Önerilen Çözüm Planı:**
Öncelikli olarak **1. Madde (Ortam Uyuşmazlığı)** giderilmelidir. Fizik ve görsel dünyanın tek bir "Gerçeklik Kaynağı"ndan (Single Source of Truth) beslenmesi sağlanacaktır.
