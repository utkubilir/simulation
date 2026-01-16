"""
Yarışma Sunucusu Link Modülü (UDP)
"""

import socket
import threading
import time
import logging
from typing import Optional, Callable, Dict
from .protocol import CommunicationProtocol

logger = logging.getLogger(__name__)

class SocketClient:
    """
    Yarışma sunucusu ile UDP üzerinden haberleşen istemci.
    
    - Periyodik telemetri gönderimi.
    - Rakip telemetrilerini dinleme.
    - Kilitlenme verilerini anlık raporlama.
    """
    
    def __init__(self, host: str, port: int, team_id: str, password: str):
        self.host = host
        self.port = port
        self.team_id = team_id
        self.password = password
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_connected = False
        self.last_telemetry_time = 0
        self.telemetry_interval = 1.0  # 1 Hz saniyede bir
        
        self.on_message_received: Optional[Callable[[Dict], None]] = None
        self._listen_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Haberleşmeyi başlat"""
        self._running = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        # Login denemesi
        self.send_login()
        logger.info(f"SocketClient: {self.host}:{self.port} adresine bağlanıldı.")

    def stop(self):
        """Haberleşmeyi durdur"""
        self._running = False
        if self.sock:
            self.sock.close()
        logger.info("SocketClient: Haberleşme kapatıldı.")

    def _listen_loop(self):
        """Sunucudan gelen paketleri dinle"""
        while self._running:
            try:
                data, addr = self.sock.recvfrom(4096)
                message = data.decode('utf-8')
                decoded = CommunicationProtocol.decode_message(message)
                
                if self.on_message_received:
                    self.on_message_received(decoded)
            except Exception as e:
                if self._running:
                    logger.error(f"SocketClient dinleme hatası: {e}")
                time.sleep(0.1)

    def send_login(self):
        """Sunucuya login paketi gönder"""
        login_pkg = CommunicationProtocol.encode_login(self.team_id, self.password)
        self._send_raw(login_pkg)

    def send_telemetry(self, uav_state: Dict):
        """Uçağın telemetrisini gönder"""
        now = time.time()
        if now - self.last_telemetry_time >= self.telemetry_interval:
            tele_pkg = CommunicationProtocol.encode_telemetry(self.team_id, uav_state)
            self._send_raw(tele_pkg)
            self.last_telemetry_time = now

    def send_lock_report(self, target_id: str, start_time: int, end_time: int):
        """Kilitlenme raporunu gönder"""
        lock_pkg = CommunicationProtocol.encode_lock_report(
            self.team_id, target_id, start_time, end_time
        )
        self._send_raw(lock_pkg)
        logger.info(f"🎯 Kilitlenme sunucuya raporlandı: {target_id}")

    def _send_raw(self, message: str):
        """UDP paketi gönder"""
        try:
            self.sock.sendto(message.encode('utf-8'), (self.host, self.port))
        except Exception as e:
            logger.error(f"SocketClient gönderme hatası: {e}")
