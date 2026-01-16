#!/usr/bin/env python3
"""
Teknofest Savaşan İHA - 3D Simülasyon

Sadece kendi İHA'nızı 3D ortamda görün.
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.world import SimulationWorld
from src.simulation.renderer_3d import Renderer3D


def main():
    print("\n" + "="*50)
    print("🛩️  3D Savaşan İHA Simülasyonu")
    print("="*50)
    
    # Dünya oluştur
    world = SimulationWorld()
    
    # Sadece oyuncu İHA
    world.spawn_uav(
        uav_id='player',
        team='blue',
        position=[500, 500, 100],
        heading=45,
        is_player=True
    )
    
    # Düşman İHA
    world.spawn_uav(
        uav_id='enemy_1',
        team='red',
        position=[800, 800, 150],
        heading=225,
        is_player=False
    )
    
    print("\n✓ Oyuncu İHA oluşturuldu")
    print("\nKontroller:")
    print("  W/S     : Pitch (burun yukarı/aşağı)")
    print("  A/D     : Roll (sola/sağa yatır)")
    print("  Q/E     : Yaw (sola/sağa dön)")
    print("  Shift   : Gaz artır")
    print("  Ctrl    : Gaz azalt")
    print("  C       : Kamera değiştir")
    print("  1/2/3   : Takip/Kokpit/Orbit kamera")
    print("  ESC     : Çıkış")
    print("="*50 + "\n")
    
    # 3D renderer başlat
    app = Renderer3D(world=world)
    app.run()


if __name__ == '__main__':
    main()
