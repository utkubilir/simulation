"""
Kamikaze Görevi Modülü

Şartname 6.2'ye uygun yer hedefi dalış sistemi.

Görev Akışı:
1. APPROACH: Hedef koordinatlarına yaklaş
2. CLIMB: Minimum dalış irtifasının üzerine çık (>100m)
3. ALIGN: Hedef üzerinde pozisyon al
4. DIVE: Dik dalış başlat, QR kodu oku
5. PULLUP: Güvenli toparlanma manevrası
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict


class KamikazePhase(Enum):
    """Dalış fazları"""
    IDLE = "idle"               # Görev başlamadı
    APPROACH = "approach"       # Hedefe yaklaşma
    CLIMB = "climb"             # Dalış irtifasına tırmanma
    ALIGN = "align"             # Hedef üzerinde hizalanma
    DIVE = "dive"               # Dalış
    PULLUP = "pullup"           # Toparlanma manevrası
    COMPLETE = "complete"       # Görev tamamlandı
    FAILED = "failed"           # Başarısız


@dataclass
class KamikazeConfig:
    """Kamikaze görev parametreleri (Şartname 6.2 uyumlu)"""
    min_dive_altitude: float = 100.0      # Minimum dalış başlangıç irtifası (şartname: 100m)
    approach_altitude: float = 120.0      # Yaklaşma irtifası
    dive_angle: float = -60.0             # Dalış açısı (derece)
    pullup_altitude: float = 30.0         # Toparlanma başlangıç irtifası
    pullup_target_altitude: float = 80.0  # Toparlanma hedef irtifası
    max_dive_speed: float = 50.0          # Maksimum dalış hızı
    approach_distance: float = 200.0      # Tırmanma başlangıç mesafesi
    align_distance: float = 50.0          # Dalış başlangıç mesafesi
    qr_detection_timeout: float = 2.0     # QR okuma timeout
    target_area_margin: float = 0.3       # Hedef vuruş alanı (şartname: %30)


@dataclass
class GroundTarget:
    """Yer hedefi bilgisi (Şartname 6.2)"""
    position: np.ndarray      # [x, y, z] - z=0 yer seviyesi
    qr_content: str           # QR kod içeriği
    size: float = 2.0         # QR kod boyutu (şartname: 2m x 2m)
    wall_height: float = 3.0  # Çevre duvar yüksekliği (şartname: 3m)
    wall_angle: float = 45.0  # Duvar açısı (şartname: 45 derece)


class KamikazeController:
    """
    Kamikaze dalış kontrolcüsü
    
    Şartname 6.2 uyumlu yer hedefi dalış sistemi.
    QR kod okuma ve güvenli toparlanma manevrası sağlar.
    
    Görev akışı:
    1. APPROACH: Hedef koordinatlarına yaklaş
    2. CLIMB: Minimum dalış irtifasının üzerine çık (>100m)
    3. ALIGN: Hedef üzerinde pozisyon al
    4. DIVE: Dik dalış başlat, QR kodu oku
    5. PULLUP: Güvenli toparlanma manevrası
    """
    
    def __init__(self, target: GroundTarget, config: KamikazeConfig = None):
        self.target = target
        self.config = config or KamikazeConfig()
        self.phase = KamikazePhase.IDLE
        self.phase_start_time = 0.0
        
        # Dalış verileri
        self.dive_start_altitude = 0.0
        self.dive_start_time = 0.0
        self.dive_end_time = None
        
        # QR tespit
        self.qr_detected = False
        self.qr_read_content = None
        self.qr_detection_time = None
        
        # Sunucu paketi gönderildi mi?
        self.packet_sent = False
        
    def start(self, sim_time: float):
        """Görevi başlat"""
        if self.phase == KamikazePhase.IDLE:
            self.phase = KamikazePhase.APPROACH
            self.phase_start_time = sim_time
            print(f"🎯 Kamikaze görevi başlatıldı - Hedef: {self.target.position[:2]}")
            
    def reset(self):
        """Görevi sıfırla"""
        self.phase = KamikazePhase.IDLE
        self.phase_start_time = 0.0
        self.dive_start_altitude = 0.0
        self.dive_start_time = 0.0
        self.dive_end_time = None
        self.qr_detected = False
        self.qr_read_content = None
        self.qr_detection_time = None
        self.packet_sent = False
        
    def update(self, uav_state: dict, camera_data: dict, sim_time: float) -> dict:
        """
        Kamikaze görev güncellemesi
        
        Args:
            uav_state: İHA durum bilgisi {position, altitude, heading, speed, velocity}
            camera_data: Kamera görüntüsü ve QR tespit bilgisi {qr_detected, qr_content, qr_bbox}
            sim_time: Simülasyon zamanı
            
        Returns:
            {
                'phase': KamikazePhase,
                'autopilot_commands': dict,  # heading, altitude, speed, pitch
                'server_packet': optional dict,  # Kamikaze paketi
                'mission_complete': bool,
                'mission_success': bool
            }
        """
        if self.phase == KamikazePhase.IDLE:
            return self._idle_result()
            
        pos = np.array(uav_state['position'])
        alt = uav_state.get('altitude', pos[2])
        heading = uav_state.get('heading', 0)
        speed = uav_state.get('speed', 25.0)
        
        if self.phase == KamikazePhase.APPROACH:
            return self._approach(pos, alt, heading, sim_time)
            
        elif self.phase == KamikazePhase.CLIMB:
            return self._climb(pos, alt, heading, sim_time)
            
        elif self.phase == KamikazePhase.ALIGN:
            return self._align(pos, alt, heading, sim_time)
            
        elif self.phase == KamikazePhase.DIVE:
            return self._dive(pos, alt, heading, camera_data, sim_time)
            
        elif self.phase == KamikazePhase.PULLUP:
            return self._pullup(pos, alt, heading, sim_time)
            
        elif self.phase == KamikazePhase.COMPLETE:
            return self._complete_result(True)
            
        elif self.phase == KamikazePhase.FAILED:
            return self._complete_result(False)
            
        return self._idle_result()
    
    def _idle_result(self) -> dict:
        """Boşta sonucu"""
        return {
            'phase': self.phase,
            'autopilot_commands': {},
            'server_packet': None,
            'mission_complete': False,
            'mission_success': False
        }
        
    def _complete_result(self, success: bool) -> dict:
        """Tamamlanma sonucu"""
        return {
            'phase': self.phase,
            'autopilot_commands': {'altitude': self.config.approach_altitude, 'speed': 25.0},
            'server_packet': None,
            'mission_complete': True,
            'mission_success': success
        }
    
    def _heading_to_target(self, pos: np.ndarray) -> float:
        """Hedefe yönü hesapla (derece)"""
        target_pos = self.target.position[:2]
        diff = target_pos - pos[:2]
        heading_rad = np.arctan2(diff[1], diff[0])
        return np.degrees(heading_rad)
    
    def _distance_to_target(self, pos: np.ndarray) -> float:
        """Hedefe mesafeyi hesapla (metre, 2D)"""
        target_pos = self.target.position[:2]
        return np.linalg.norm(target_pos - pos[:2])
    
    def _approach(self, pos: np.ndarray, alt: float, heading: float, sim_time: float) -> dict:
        """Hedefe yaklaşma fazı"""
        commands = {}
        
        dist = self._distance_to_target(pos)
        target_heading = self._heading_to_target(pos)
        
        commands['heading'] = target_heading
        commands['altitude'] = self.config.approach_altitude
        commands['speed'] = 30.0
        commands['throttle'] = 0.8
        
        # Hedefe yeterince yakınsa tırmanmaya başla
        if dist < self.config.approach_distance:
            self.phase = KamikazePhase.CLIMB
            self.phase_start_time = sim_time
            print(f"📈 CLIMB fazına geçiliyor - İrtifa: {alt:.1f}m, Hedef: >{self.config.min_dive_altitude}m")
            
        return {
            'phase': self.phase,
            'autopilot_commands': commands,
            'server_packet': None,
            'mission_complete': False,
            'mission_success': False
        }
    
    def _climb(self, pos: np.ndarray, alt: float, heading: float, sim_time: float) -> dict:
        """Dalış irtifasına tırmanma"""
        commands = {}
        
        target_alt = self.config.min_dive_altitude + 20  # 120m hedef
        target_heading = self._heading_to_target(pos)
        
        commands['altitude'] = target_alt
        commands['heading'] = target_heading
        commands['speed'] = 25.0
        commands['throttle'] = 0.9
        
        # Yeterli irtifaya ulaştıysa hizalanmaya geç
        if alt >= self.config.min_dive_altitude:
            self.phase = KamikazePhase.ALIGN
            self.phase_start_time = sim_time
            self.dive_start_altitude = alt
            print(f"🎯 ALIGN fazına geçiliyor - İrtifa: {alt:.1f}m")
            
        return {
            'phase': self.phase,
            'autopilot_commands': commands,
            'server_packet': None,
            'mission_complete': False,
            'mission_success': False
        }
    
    def _align(self, pos: np.ndarray, alt: float, heading: float, sim_time: float) -> dict:
        """Hedef üzerinde hizalanma"""
        commands = {}
        
        dist = self._distance_to_target(pos)
        target_heading = self._heading_to_target(pos)
        
        commands['heading'] = target_heading
        commands['altitude'] = self.config.min_dive_altitude + 10
        commands['speed'] = 20.0
        commands['throttle'] = 0.6
        
        # Hedef üzerinde ve hizalıysa dalışa başla
        if dist < self.config.align_distance:
            self.phase = KamikazePhase.DIVE
            self.phase_start_time = sim_time
            self.dive_start_time = sim_time
            self.dive_start_altitude = alt
            print(f"⬇️ DIVE fazına geçiliyor - İrtifa: {alt:.1f}m, Mesafe: {dist:.1f}m")
            
        return {
            'phase': self.phase,
            'autopilot_commands': commands,
            'server_packet': None,
            'mission_complete': False,
            'mission_success': False
        }
    
    def _dive(self, pos: np.ndarray, alt: float, heading: float, 
              camera_data: dict, sim_time: float) -> dict:
        """Dalış fazı - QR kod okuma"""
        commands = {}
        
        target_heading = self._heading_to_target(pos)
        
        commands['pitch'] = self.config.dive_angle  # Dik dalış
        commands['heading'] = target_heading
        commands['throttle'] = 0.3  # Düşük gaz
        
        server_packet = None
        
        # QR kod tespiti
        if camera_data and not self.qr_detected:
            if camera_data.get('qr_detected'):
                self.qr_detected = True
                self.qr_read_content = camera_data.get('qr_content')
                self.qr_detection_time = sim_time
                self.dive_end_time = sim_time
                
                print(f"✅ QR kod okundu: {self.qr_read_content}")
                
                # Sunucuya kamikaze paketi hazırla (şartname format)
                server_packet = {
                    'type': 'kamikaze',
                    'dive_end_time': self.dive_end_time,
                    'qr_content': self.qr_read_content,
                    'dive_start_altitude': self.dive_start_altitude,
                    'position': pos.tolist(),
                    'timestamp': sim_time
                }
                self.packet_sent = True
        
        # Toparlanma irtifasına ulaştıysa veya QR okuduysa
        if alt <= self.config.pullup_altitude:
            self.phase = KamikazePhase.PULLUP
            self.phase_start_time = sim_time
            print(f"⬆️ PULLUP fazına geçiliyor - İrtifa: {alt:.1f}m, QR: {self.qr_detected}")
        elif self.qr_detected and alt <= self.config.pullup_altitude + 20:
            # QR okunduysa biraz daha erken toparlanmaya başla
            self.phase = KamikazePhase.PULLUP
            self.phase_start_time = sim_time
            print(f"⬆️ PULLUP fazına geçiliyor (QR okundu) - İrtifa: {alt:.1f}m")
            
        return {
            'phase': self.phase,
            'autopilot_commands': commands,
            'server_packet': server_packet,
            'mission_complete': False,
            'mission_success': False
        }
    
    def _pullup(self, pos: np.ndarray, alt: float, heading: float, sim_time: float) -> dict:
        """Toparlanma manevrası"""
        commands = {}
        
        commands['pitch'] = 20.0  # Yukarı çek
        commands['throttle'] = 1.0  # Tam gaz
        commands['altitude'] = self.config.pullup_target_altitude
        commands['speed'] = 30.0
        
        # Güvenli irtifaya ulaştıysa görev tamamlandı
        if alt >= self.config.pullup_target_altitude:
            if self.qr_detected:
                self.phase = KamikazePhase.COMPLETE
                print(f"🎉 Kamikaze görevi BAŞARILI! QR: {self.qr_read_content}")
            else:
                self.phase = KamikazePhase.FAILED
                print(f"❌ Kamikaze görevi BAŞARISIZ - QR okunamadı")
            
        return {
            'phase': self.phase,
            'autopilot_commands': commands,
            'server_packet': None,
            'mission_complete': self.phase in [KamikazePhase.COMPLETE, KamikazePhase.FAILED],
            'mission_success': self.phase == KamikazePhase.COMPLETE
        }
    
    def get_status(self) -> dict:
        """Görev durumu"""
        return {
            'phase': self.phase.value,
            'qr_detected': self.qr_detected,
            'qr_content': self.qr_read_content,
            'dive_start_altitude': self.dive_start_altitude,
            'dive_start_time': self.dive_start_time,
            'dive_end_time': self.dive_end_time,
            'packet_sent': self.packet_sent
        }
    
    def is_active(self) -> bool:
        """Görev aktif mi?"""
        return self.phase not in [KamikazePhase.IDLE, KamikazePhase.COMPLETE, KamikazePhase.FAILED]
    
    def is_complete(self) -> bool:
        """Görev tamamlandı mı?"""
        return self.phase in [KamikazePhase.COMPLETE, KamikazePhase.FAILED]
    
    def is_successful(self) -> bool:
        """Görev başarılı mı?"""
        return self.phase == KamikazePhase.COMPLETE
