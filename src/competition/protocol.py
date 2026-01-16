"""
İletişim Protokolü
"""

import json
import time
from typing import Dict


class CommunicationProtocol:
    """Yarışma sunucusu iletişim protokolü (V2 - Tam Uyumlu)"""
    
    @staticmethod
    def encode_login(team_id: str, password: str) -> str:
        """Sunucuya giriş paketi"""
        return json.dumps({
            'kullaniciAdi': team_id,
            'sifre': password
        })

    @staticmethod
    def encode_telemetry(team_id: str, uav_state: Dict) -> str:
        """Periyodik telemetri paketi"""
        return json.dumps({
            'takim_numarasi': team_id,
            'iha_enlem': uav_state.get('lat', 0.0),
            'iha_boylam': uav_state.get('lon', 0.0),
            'iha_irtifa': uav_state.get('altitude', 0.0),
            'iha_dikilme': uav_state.get('pitch', 0.0),
            'iha_yonelme': uav_state.get('yaw', 0.0),
            'iha_yatis': uav_state.get('roll', 0.0),
            'iha_hiz': uav_state.get('speed', 0.0),
            'zaman_damgasi': int(time.time() * 1000)
        })
    
    @staticmethod
    def encode_lock_report(team_id: str, target_id: str, 
                           start_time: int, end_time: int) -> str:
        """Kilitlenme raporunu JSON'a encode et"""
        return json.dumps({
            'takim_numarasi': team_id,
            'kilitlenmeBaslangicZamani': start_time,
            'kilitlenmeBitisZamani': end_time,
            'hedef_id': target_id
        })
        
    @staticmethod
    def decode_message(message: str) -> Dict:
        """Gelen sunucu mesajını (rakip telemetrileri vb.) decode et"""
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return {}

