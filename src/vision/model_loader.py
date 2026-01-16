"""
Harici Model ve Veri Yükleme Yöneticisi

YOLO modellerini ve dataset'leri projeye aktarmak için kullanılır.
"""

from pathlib import Path
import shutil
import json
from typing import List, Dict, Optional
from datetime import datetime


class ModelLoader:
    """
    Harici YOLO modelleri ve dataset'leri yönetir
    
    Kullanım:
        loader = ModelLoader()
        
        # Kendi modelinizi import edin
        loader.import_model('/path/to/your/best.pt', 'my_uav_model')
        
        # Dataset import edin
        loader.import_dataset('/path/to/dataset/', 'uav_dataset_v1')
        
        # Mevcut modelleri listele
        models = loader.list_models()
        
        # Model yolunu al
        path = loader.get_model_path('my_uav_model')
    """
    
    # Varsayılan dizinler
    MODELS_DIR = Path('models/custom')
    DATASETS_DIR = Path('data/custom')
    PRETRAINED_DIR = Path('models/pretrained')
    
    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: Proje kök dizini (None ise mevcut dizin)
        """
        if base_path:
            base = Path(base_path)
            self.MODELS_DIR = base / 'models' / 'custom'
            self.DATASETS_DIR = base / 'data' / 'custom'
            self.PRETRAINED_DIR = base / 'models' / 'pretrained'
            
        # Dizinleri oluştur
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model metadata dosyası
        self.metadata_file = self.MODELS_DIR / 'metadata.json'
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Metadata dosyasını yükle"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'models': {}, 'datasets': {}}
        
    def _save_metadata(self):
        """Metadata dosyasını kaydet"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
    def import_model(self, source_path: str, model_name: str, 
                     description: str = None, classes: List[str] = None) -> Path:
        """
        Harici YOLO modelini projeye aktar
        
        Args:
            source_path: Kaynak .pt dosyası
            model_name: Model için isim (boşluksuz)
            description: Model açıklaması
            classes: Model sınıfları listesi
            
        Returns:
            Kopyalanan model dosyasının yolu
        """
        source = Path(source_path)
        
        if not source.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {source}")
            
        if not source.suffix == '.pt':
            raise ValueError("Model dosyası .pt uzantılı olmalıdır")
            
        # Geçersiz karakterleri temizle
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-'))
        
        # Hedef yol
        dest = self.MODELS_DIR / f"{safe_name}.pt"
        
        # Kopyala
        shutil.copy(source, dest)
        
        # Metadata güncelle
        self.metadata['models'][safe_name] = {
            'path': str(dest),
            'original_path': str(source),
            'description': description or '',
            'classes': classes or ['uav'],
            'imported_at': datetime.now().isoformat(),
            'size_mb': dest.stat().st_size / (1024 * 1024)
        }
        self._save_metadata()
        
        print(f"✓ Model aktarıldı: {dest}")
        print(f"  Boyut: {self.metadata['models'][safe_name]['size_mb']:.2f} MB")
        
        return dest
        
    def import_dataset(self, source_dir: str, dataset_name: str,
                       description: str = None) -> Path:
        """
        Eğitim/test dataset'ini projeye aktar
        
        Beklenen YOLO format yapısı:
        source_dir/
        ├── images/
        │   ├── train/
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── val/
        │       ├── image1.jpg
        │       └── ...
        ├── labels/
        │   ├── train/
        │   │   ├── image1.txt
        │   │   └── ...
        │   └── val/
        │       ├── image1.txt
        │       └── ...
        └── data.yaml (opsiyonel)
        
        Args:
            source_dir: Kaynak dataset dizini
            dataset_name: Dataset için isim
            description: Dataset açıklaması
            
        Returns:
            Kopyalanan dataset dizininin yolu
        """
        source = Path(source_dir)
        
        if not source.exists():
            raise FileNotFoundError(f"Dataset dizini bulunamadı: {source}")
            
        if not source.is_dir():
            raise ValueError("source_dir bir dizin olmalıdır")
            
        # Yapı kontrolü
        images_dir = source / 'images'
        labels_dir = source / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️  Uyarı: 'images' dizini bulunamadı")
        if not labels_dir.exists():
            print(f"⚠️  Uyarı: 'labels' dizini bulunamadı")
            
        # Geçersiz karakterleri temizle
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-'))
        
        # Hedef yol
        dest = self.DATASETS_DIR / safe_name
        
        # Kopyala
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        
        # İstatistikler
        train_images = len(list((dest / 'images' / 'train').glob('*'))) if (dest / 'images' / 'train').exists() else 0
        val_images = len(list((dest / 'images' / 'val').glob('*'))) if (dest / 'images' / 'val').exists() else 0
        
        # Metadata güncelle
        self.metadata['datasets'][safe_name] = {
            'path': str(dest),
            'original_path': str(source),
            'description': description or '',
            'imported_at': datetime.now().isoformat(),
            'train_images': train_images,
            'val_images': val_images
        }
        self._save_metadata()
        
        print(f"✓ Dataset aktarıldı: {dest}")
        print(f"  Eğitim görüntüleri: {train_images}")
        print(f"  Doğrulama görüntüleri: {val_images}")
        
        return dest
        
    def list_models(self) -> List[Dict]:
        """
        Mevcut modelleri listele
        
        Returns:
            Model bilgileri listesi
        """
        models = []
        
        for name, info in self.metadata.get('models', {}).items():
            path = Path(info['path'])
            if path.exists():
                models.append({
                    'name': name,
                    'path': str(path),
                    'description': info.get('description', ''),
                    'classes': info.get('classes', []),
                    'size_mb': info.get('size_mb', 0),
                    'imported_at': info.get('imported_at', '')
                })
                
        return models
        
    def list_datasets(self) -> List[Dict]:
        """
        Mevcut dataset'leri listele
        
        Returns:
            Dataset bilgileri listesi
        """
        datasets = []
        
        for name, info in self.metadata.get('datasets', {}).items():
            path = Path(info['path'])
            if path.exists():
                datasets.append({
                    'name': name,
                    'path': str(path),
                    'description': info.get('description', ''),
                    'train_images': info.get('train_images', 0),
                    'val_images': info.get('val_images', 0),
                    'imported_at': info.get('imported_at', '')
                })
                
        return datasets
        
    def get_model_path(self, model_name: str) -> Path:
        """
        Model dosya yolunu al
        
        Args:
            model_name: Model adı
            
        Returns:
            Model dosyasının yolu
        """
        if model_name in self.metadata.get('models', {}):
            path = Path(self.metadata['models'][model_name]['path'])
            if path.exists():
                return path
                
        # Doğrudan dosya kontrolü
        direct_path = self.MODELS_DIR / f"{model_name}.pt"
        if direct_path.exists():
            return direct_path
            
        raise FileNotFoundError(f"Model bulunamadı: {model_name}")
        
    def get_dataset_path(self, dataset_name: str) -> Path:
        """
        Dataset dizin yolunu al
        
        Args:
            dataset_name: Dataset adı
            
        Returns:
            Dataset dizininin yolu
        """
        if dataset_name in self.metadata.get('datasets', {}):
            path = Path(self.metadata['datasets'][dataset_name]['path'])
            if path.exists():
                return path
                
        # Doğrudan dizin kontrolü
        direct_path = self.DATASETS_DIR / dataset_name
        if direct_path.exists():
            return direct_path
            
        raise FileNotFoundError(f"Dataset bulunamadı: {dataset_name}")
        
    def delete_model(self, model_name: str):
        """Model sil"""
        if model_name in self.metadata.get('models', {}):
            path = Path(self.metadata['models'][model_name]['path'])
            if path.exists():
                path.unlink()
            del self.metadata['models'][model_name]
            self._save_metadata()
            print(f"✓ Model silindi: {model_name}")
        else:
            print(f"⚠️  Model bulunamadı: {model_name}")
            
    def delete_dataset(self, dataset_name: str):
        """Dataset sil"""
        if dataset_name in self.metadata.get('datasets', {}):
            path = Path(self.metadata['datasets'][dataset_name]['path'])
            if path.exists():
                shutil.rmtree(path)
            del self.metadata['datasets'][dataset_name]
            self._save_metadata()
            print(f"✓ Dataset silindi: {dataset_name}")
        else:
            print(f"⚠️  Dataset bulunamadı: {dataset_name}")
            
    def create_dataset_yaml(self, dataset_name: str, classes: List[str] = None) -> Path:
        """
        YOLO format data.yaml dosyası oluştur
        
        Args:
            dataset_name: Dataset adı
            classes: Sınıf isimleri listesi
            
        Returns:
            Oluşturulan yaml dosyasının yolu
        """
        dataset_path = self.get_dataset_path(dataset_name)
        yaml_path = dataset_path / 'data.yaml'
        
        classes = classes or ['uav']
        
        content = f"""# YOLO Dataset Configuration
# Otomatik oluşturuldu

path: {dataset_path.absolute()}
train: images/train
val: images/val

# Sınıflar
nc: {len(classes)}
names: {classes}
"""
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✓ data.yaml oluşturuldu: {yaml_path}")
        return yaml_path
        
    def print_summary(self):
        """Özet bilgi yazdır"""
        models = self.list_models()
        datasets = self.list_datasets()
        
        print("\n" + "="*50)
        print("📦 Model ve Dataset Özeti")
        print("="*50)
        
        print(f"\n🤖 Modeller ({len(models)} adet):")
        if models:
            for m in models:
                print(f"   • {m['name']}: {m['size_mb']:.1f} MB")
        else:
            print("   (Model bulunamadı)")
            
        print(f"\n📁 Dataset'ler ({len(datasets)} adet):")
        if datasets:
            for d in datasets:
                total = d['train_images'] + d['val_images']
                print(f"   • {d['name']}: {total} görüntü")
        else:
            print("   (Dataset bulunamadı)")
            
        print("="*50 + "\n")
