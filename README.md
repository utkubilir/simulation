# Teknofest Savaşan İHA Simülasyonu (vNext)

Bu repo, Teknofest Savaşan İHA yarışması için simülasyon ortamını sağlar. UI (görsel) veya headless (arayüzsüz) modlarda çalıştırılabilir; senaryo tabanlı koşular üretir, tespit/takip/kilitlenme süreçlerini simüle eder ve sonuçları kaydeder.

## Özellikler

- **Senaryo tabanlı simülasyon**: Farklı uçuş ve hedef davranışları için YAML senaryoları.
- **UI ve headless modları**: Görselleştirme ile veya arayüzsüz çalıştırma seçenekleri.
- **Deterministik çalışma**: Aynı seed ile tekrarlanabilir koşular.
- **Kamera ve algılama simülasyonu**: Görüş açısı, çözünürlük, gürültü, distorsiyon gibi parametreler.
- **Takip ve kilitlenme mantığı**: Takipçi ve kilitlenme durum makinesi ile yarışma mantığı.
- **Çıktı kayıtları**: Frame ve run seviyesinde sonuçların `results/` altında saklanması.

## Gereksinimler

- Python 3.10+
- Gerekli bağımlılıklar (ör. `numpy`, `opencv`, `pygame` vb.)

> Not: Kurulum yöntemi repo içindeki paketleme yapısına göre değişebilir. Basit kurulum için `pip install -r requirements.txt` (varsa) kullanılabilir.

## Hızlı Başlangıç

### 1) UI modu (görsel)

```bash
python main.py
```

Parametrelerle çalıştırma:

```bash
python main.py --scenario easy_lock --uav-count 3 --run-id demo_run
```

### 2) Headless modu

```bash
python -m scripts.run --mode headless --scenario easy_lock --seed 42 --duration 30
```

### 3) UI modu (scripts.run)

```bash
python -m scripts.run --mode ui --scenario easy_lock --seed 42
```

## Konfigürasyon

### Genel ayarlar

`config/settings.yaml` içinde simülasyon genel ayarları yer alır:

- Simülasyon FPS ve fizik güncelleme hızı
- Dünya boyutları
- İHA parametreleri
- Kamera parametreleri (FOV, çözünürlük, gürültü vs.)
- Algılama ve takip parametreleri
- Kilitlenme mantığı

### Senaryolar

Senaryolar YAML dosyaları ile tanımlıdır:

- `scenarios/` dizini simülasyon tarafından kullanılan senaryoları içerir.
- `config/scenarios/` dizini alternatif/ek senaryolar barındırır.

Örnek senaryo adları:

- `default`, `easy_lock`, `zigzag`, `multi_target_3`, `high_speed`

Senaryo adını CLI ile seçebilirsiniz:

```bash
python -m scripts.run --mode headless --scenario zigzag --seed 7
```

## Çıktılar

- Simülasyon çıktıları ve loglar varsayılan olarak `results/` dizinine yazılır.
- `run_id` kullanarak koşuların çıktıları birbirinden ayrıştırılabilir.

## Testler

Projede test altyapısı `pytest` ile yapılandırılmıştır:

```bash
pytest
```

Belirli bir test grubunu çalıştırma örneği:

```bash
pytest -m unit
```

## Dizin Yapısı

- `main.py`: Geriye uyumluluk için wrapper giriş noktası.
- `scripts/run.py`: Esas CLI girişi (UI/headless seçimleri).
- `src/`: Simülasyon, render, UAV, algılama ve takip modülleri.
- `scenarios/`: Simülasyon senaryoları.
- `config/`: Genel ayarlar ve ek senaryolar.
- `tests/`: Testler.
- `results/`: Simülasyon çıktıları.

## İpuçları

- Aynı senaryo ve seed ile tekrar çalıştırarak deterministik çıktı alabilirsiniz.
- Headless modu, CI veya hızlı regresyon koşuları için uygundur.
- UI modu, kamera ve HUD davranışlarını görsel olarak doğrulamak için önerilir.
