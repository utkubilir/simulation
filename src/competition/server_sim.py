"""
Yarışma Sunucusu Simülasyonu
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TeamScore:
    """Takım puanı"""
    team_id: str
    total_score: int = 0
    correct_locks: int = 0
    incorrect_locks: int = 0
    lock_history: list = field(default_factory=list)


class CompetitionServerSimulator:
    """
    Teknofest yarışma sunucusunu simüle eder
    
    - Telemetri verileri sağlar
    - Kilitlenme bildirimlerini değerlendirir
    - Puanlama yapar
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Puanlama
        self.correct_lock_points = config.get('correct_lock_points', 100)
        self.incorrect_penalty = config.get('incorrect_lock_penalty', -50)
        self.min_lock_duration = config.get('min_lock_duration', 1.0)
        
        # Takımlar
        self.teams: Dict[str, TeamScore] = {}
        
        # İHA pozisyonları (simülasyondan güncellenir)
        self.uav_positions: Dict[str, Dict] = {}
        
        # Geçerli kilitlenmeler
        self.valid_locks: List[Dict] = []
        
        # Sunucu zamanı
        self.start_time = time.time()
        
    def register_team(self, team_id: str):
        """Takım kaydet"""
        if team_id not in self.teams:
            self.teams[team_id] = TeamScore(team_id=team_id)
            logger.info(f"✓ Takım kaydedildi: {team_id}")
            
    def update_uav_positions(self, positions: Dict[str, Dict]):
        """İHA pozisyonlarını güncelle"""
        self.uav_positions = positions
        
    def get_telemetry(self, team_id: str = None, sim_time: float = None) -> Dict:
        """
        Telemetri verilerini al
        
        Gerçek yarışmada sunucu tüm İHA pozisyonlarını paylaşır.
        
        Args:
            team_id: Takım ID (opsiyonel)
            sim_time: Simülasyon zamanı (opsiyonel, wall-clock yerine kullanılır)
        """
        timestamp = sim_time if sim_time is not None else (time.time() - self.start_time)
        return {
            'timestamp': timestamp,
            'uavs': self.uav_positions.copy()
        }
        
    def report_lock_on(self, team_id: str, lock_data: Dict, sim_time: float = None) -> Dict:
        """
        Kilitlenme bildirimi
        
        Args:
            team_id: Bildiren takım
            lock_data: {
                'target_id': hedef İHA ID,
                'lock_quadrangle': (x1, y1, x2, y2),
                'duration': kilitlenme süresi,
                'timestamp': zaman
            }
            sim_time: Simülasyon zamanı (opsiyonel, determinizm için)
            
        Returns:
            Sonuç ve puan
        """
        if team_id not in self.teams:
            self.register_team(team_id)
            
        target_id = lock_data.get('target_id')
        duration = lock_data.get('duration', 0)
        
        result = {
            'accepted': False,
            'points': 0,
            'message': '',
            'total_score': 0
        }
        
        # Süre kontrolü
        if duration < self.min_lock_duration:
            result['message'] = f'Yetersiz kilitlenme süresi: {duration:.2f}s < {self.min_lock_duration}s'
            return result
            
        # Hedef kontrolü
        if target_id is None or target_id not in self.uav_positions:
            # Yanlış kilitlenme
            self.teams[team_id].incorrect_locks += 1
            self.teams[team_id].total_score += self.incorrect_penalty
            
            result['points'] = self.incorrect_penalty
            result['message'] = f'Geçersiz hedef: {target_id}'
            result['total_score'] = self.teams[team_id].total_score
            
            self.teams[team_id].lock_history.append({
                'target_id': target_id,
                'valid': False,
                'points': self.incorrect_penalty,
                'timestamp': sim_time if sim_time is not None else time.time()
            })
            
            return result
            
        # Doğru kilitlenme
        self.teams[team_id].correct_locks += 1
        self.teams[team_id].total_score += self.correct_lock_points
        
        result['accepted'] = True
        result['points'] = self.correct_lock_points
        result['message'] = f'Kilitlenme onaylandı: {target_id}'
        result['total_score'] = self.teams[team_id].total_score
        
        self.teams[team_id].lock_history.append({
            'target_id': target_id,
            'valid': True,
            'points': self.correct_lock_points,
            'timestamp': sim_time if sim_time is not None else time.time()
        })
        
        
        self.valid_locks.append({
            'team_id': team_id,
            'target_id': target_id,
            'lock_data': lock_data,
            'timestamp': sim_time if sim_time is not None else time.time()
        })
        
        logger.info(f"🎯 {team_id}: {target_id}'e kilitlenme! +{self.correct_lock_points} puan")
        
        return result
        
    def get_score(self, team_id: str) -> Optional[Dict]:
        """Takım puanını al"""
        if team_id not in self.teams:
            return None
            
        team = self.teams[team_id]
        return {
            'team_id': team_id,
            'total_score': team.total_score,
            'correct_locks': team.correct_locks,
            'incorrect_locks': team.incorrect_locks,
            'lock_history': team.lock_history[-10:]  # Son 10
        }
        
    def get_leaderboard(self) -> List[Dict]:
        """Sıralama tablosu"""
        scores = []
        for team_id, team in self.teams.items():
            scores.append({
                'team_id': team_id,
                'score': team.total_score,
                'locks': team.correct_locks
            })
        return sorted(scores, key=lambda x: x['score'], reverse=True)
        
    def reset(self):
        """Sunucuyu sıfırla"""
        self.teams.clear()
        self.uav_positions.clear()
        self.valid_locks.clear()
        self.start_time = time.time()
