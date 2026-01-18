# Teknofest Savaşan İHA Simülasyonu (vNext)

Bu repo, Teknofest Savaşan İHA yarışması için simülasyon ortamını sağlar. UI (görsel) veya headless (arayüzsüz) modlarda çalıştırılabilir; senaryo tabanlı koşular üretir, tespit/takip/kilitlenme süreçlerini simüle eder ve sonuçları kaydeder.

## Özellikler

- **Senaryo tabanlı simülasyon**: Farklı uçuş ve hedef davranışları için YAML senaryoları.
- **UI ve headless modları**: Görselleştirme ile veya arayüzsüz çalıştırma seçenekleri.
- **3D Görselleştirme**: Panda3D tabanlı gerçekçi 3D simülasyon ortamı.
- **Deterministik çalışma**: Aynı seed ile tekrarlanabilir koşular.
- **Kamera ve algılama simülasyonu**: Görüş açısı, çözünürlük, gürültü, distorsiyon gibi parametreler.
- **Takip ve kilitlenme mantığı**: Takipçi ve kilitlenme durum makinesi ile yarışma mantığı.
- **Çıktı kayıtları**: Frame ve run seviyesinde sonuçların `results/` altında saklanması.

## Gereksinimler

Proje Python 3.10+ gerektirir. Bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
```

## Hızlı Başlangıç

### 1) UI modu (2D Görsel)

```bash
python main.py
```

Parametrelerle çalıştırma:

```bash
python main.py --scenario easy_lock --uav-count 3 --run-id demo_run
```

### 2) 3D Simülasyon

Panda3D tabanlı 3D simülasyonu başlatmak için:

```bash
python main_3d.py
```

Kontroller:
- **W/S**: Pitch (Burun yukarı/aşağı)
- **A/D**: Roll (Sola/sağa yatır)
- **Q/E**: Yaw (Sola/sağa dön)
- **Shift/Ctrl**: Gaz Artır/Azalt
- **C**: Kamera değiştir (Takip/Kokpit/Orbit)
- **1/2/3**: Kamera modları doğrudan seçim
- **P**: Otopilot (Combat modu) aç/kapa
- **ESC**: Çıkış

### 3) Headless modu (Arayüzsüz)

Toplu koşular veya CI ortamları için:

```bash
python -m scripts.run --mode headless --scenario easy_lock --seed 42 --duration 30
```

### 4) UI modu (scripts.run üzerinden)

```bash
python -m scripts.run --mode ui --scenario easy_lock --seed 42
```

## Benchmark ve Raporlama

Birden fazla senaryoyu toplu olarak çalıştırmak ve performans ölçümü yapmak için:

```bash
python -m scripts.run_all_scenarios --seeds 42 123 --output results/benchmark_v1
```

Oluşan sonuçlardan detaylı rapor üretmek için:

```bash
python -m scripts.generate_report --runs-dir results/benchmark_v1
```

Bu işlem `results/benchmark_v1/REPORT.md` dosyasını oluşturacaktır.

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

- `main.py`: 2D arayüzlü simülasyon giriş noktası.
- `main_3d.py`: 3D simülasyon giriş noktası.
- `scripts/run.py`: Esas CLI girişi (UI/headless seçimleri).
- `scripts/run_all_scenarios.py`: Benchmark çalıştırıcı.
- `scripts/generate_report.py`: Rapor oluşturucu.
- `src/`: Simülasyon, render, UAV, algılama ve takip modülleri.
- `scenarios/`: Simülasyon senaryoları.
- `config/`: Genel ayarlar ve ek senaryolar.
- `tests/`: Testler.
- `results/`: Simülasyon çıktıları.

## İpuçları

- Aynı senaryo ve seed ile tekrar çalıştırarak deterministik çıktı alabilirsiniz.
- Headless modu, CI veya hızlı regresyon koşuları için uygundur.
- UI modu, kamera ve HUD davranışlarını görsel olarak doğrulamak için önerilir.
- 3D modu, senaryoları gerçekçi bir ortamda gözlemlemek için idealdir.
