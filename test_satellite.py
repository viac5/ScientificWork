import math
import pytest
from app import calculate_satellite_position

# Тесты для функции calculate_satellite_position
# pip install pytest
# Запуск: python -m pytest test_satellite.py -v -s


def test_circular_orbit():
    """Тест для круговой экваториальной орбиты"""
    # Параметры круговой орбиты
    semi_major_axis = 7000  # км
    eccentricity = 0
    inclination = 0  # экваториальная орбита
    arg_perigee = 0
    raan = 0  # right ascension of ascending node
    time = 0
    
    x, y, z, r = calculate_satellite_position(
        semi_major_axis, eccentricity, inclination, 
        arg_perigee, raan, time
    )
    
    # Для круговой орбиты радиус должен быть равен полуоси
    assert abs(r - semi_major_axis) < 1e-6
    assert abs(math.sqrt(x*x + y*y + z*z) - semi_major_axis) < 1e-6

def test_elliptical_orbit():
    """Тест для эллиптической орбиты"""
    semi_major_axis = 7000
    eccentricity = 0.1
    inclination = math.pi/4  # 45 градусов
    arg_perigee = 0
    raan = 0
    time = 0
    
    x, y, z, r = calculate_satellite_position(
        semi_major_axis, eccentricity, inclination,
        arg_perigee, raan, time
    )
    
    # В начальный момент времени расстояние должно быть в перигее
    expected_r = semi_major_axis * (1 - eccentricity)
    assert abs(r - expected_r) < 1e-6


