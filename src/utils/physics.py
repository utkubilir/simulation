"""
Fiziksel Sabitler Modulü

Bu modül, tüm simülasyon genelinde kullanılan fiziksel sabitleri içerir.
Merkezi yönetim, farklı ortamların (yerçekimi, hava yoğunluğu vb.) kolayca
test edilmesini sağlar.
"""

import numpy as np

# Yerçekimi ivmesi (m/s^2)
GRAVITY = 9.81

# Hava yoğunluğu (kg/m^3) - Deniz seviyesi, standart atmosfer
AIR_DENSITY = 1.225

# Gaz sabiti (J/(kg*K)) - Kuru hava
GAS_CONSTANT_AIR = 287.05

# Standart sıcaklık (Kelvin) - Deniz seviyesi
STANDARD_TEMPERATURE = 288.15

# Standart basınç (Pa) - Deniz seviyesi
STANDARD_PRESSURE = 101325.0

def get_air_density(altitude: float) -> float:
    """
    Standard Atmosphere Model (Basitleştirilmiş)
    İrtifaya bağlı hava yoğunluğunu döndürür (kg/m^3).
    """
    # Her 1000 metrede yoğunluk yaklaşık %10-12 düşer (0-10km arası)
    # rho = rho0 * e^(-h/H) where H approx 8500m
    h_scale = 8500.0
    return AIR_DENSITY * np.exp(-max(0, altitude) / h_scale)

def get_wind_at_pos(position: np.ndarray, t: float) -> np.ndarray:
    """
    Rüzgar ve Türbülans Modeli
    
    Returns:
        np.ndarray: [vx, vy, vz] rüzgar hızı (m/s)
    """
    # Sabit ana rüzgar (Örn: Kuzeydoğudan 5m/s)
    base_wind = np.array([-3.5, -3.5, 0.0])
    
    # Kararsız rüzgar (Gusts) - Basit sinüzoidal türbülans
    # Gerçek sistemde Dryden/von Karman filtresi kullanılır
    gust_x = 1.5 * np.sin(0.5 * t) + 0.5 * np.cos(2.0 * t)
    gust_y = 1.2 * np.cos(0.4 * t) + 0.4 * np.sin(1.8 * t)
    gust_z = 0.5 * np.sin(1.2 * t) # Dikey türbülans
    
    return base_wind + np.array([gust_x, gust_y, gust_z])



