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
    base_wind = np.array([-2.0, -2.0, 0.0])  # Reduced from -3.5
    
    # İrtifaya bağlı rüzgar kesmesi (shear)
    altitude = max(0.0, float(position[2]))
    shear_factor = np.clip(altitude / 500.0, 0.0, 1.0)
    shear_wind = base_wind * (0.3 + 0.7 * shear_factor)
    
    # Kararsız rüzgar (Gusts) - Reduced intensity for stable flight
    gust_x = 0.3 * np.sin(0.5 * t) + 0.1 * np.cos(2.0 * t)  # Reduced from 1.5/0.5
    gust_y = 0.2 * np.cos(0.4 * t) + 0.1 * np.sin(1.8 * t)  # Reduced from 1.2/0.4
    gust_z = 0.1 * np.sin(1.2 * t)  # Reduced from 0.5
    
    return shear_wind + np.array([gust_x, gust_y, gust_z])


