import sqlite3
import json
from flask import Flask, render_template, jsonify, request
import os
import configparser
import math
import time
import numpy as np
import logging
import re
from astropy.coordinates import get_sun, EarthLocation
from astropy.time import Time
import threading

app = Flask(__name__)

# Получаем базовый путь к проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу карты относительно корня проекта
MBTILES_FILE = os.path.join(BASE_DIR, "map", "OAM-World-1-8-min-J80.mbtiles")

if not os.path.exists(MBTILES_FILE):
    raise FileNotFoundError(f"Карта не найдена по пути: {MBTILES_FILE}. "
                          f"Убедитесь, что файл карты находится в папке 'map' "
                          f"в корне проекта.")

@app.route('/3d')
def three_d_tracking():
    return render_template('3d.html')

def calculate_sun_vector(timestamp):
    """
    Расчет единичного вектора на Солнце на заданный момент времени.
    """
    try:
        # Проверка формата времени
        if not isinstance(timestamp, (int, float)):
            raise ValueError("Некорректный формат времени. Ожидается Unix timestamp.")

        # Указание текущего времени
        time = Time(timestamp, format='unix')
        # Получение положения Солнца
        sun_coords = get_sun(time)
        # Преобразование положения Солнца в геоцентрическую систему координат (в а.е.)
        sun_vector = sun_coords.cartesian.xyz.value  # Получение [x, y, z]
        # Нормализация вектора (единичный вектор)
        sun_vector = np.array(sun_vector) / np.linalg.norm(sun_vector)
        return sun_vector
    except Exception as e:
        raise ValueError(f"Ошибка при вычислении вектора Солнца: {e}")


def calculate_visibility_zone(
        roll, pitch, altitude_km, satellite_lat, satellite_lon,
        view_angle, ap_angle, sun_angle, sun_vector, num_points=100
):
    """
    Расчет зоны видимости спутника с учетом ориентации и солнечного угла.
    """
    R_earth = 6371000  # Радиус Земли в метрах
    altitude = altitude_km * 1000  # Высота спутника в метрах

    # Расчет ширины и длины зоны видимости
    half_swath_width = altitude * math.tan(math.radians(view_angle / 2))  # Половина ширины зоны видимости
    swath_length = altitude * math.tan(math.radians(ap_angle / 2))  # Длина зоны вдоль траектории спутника

    # Преобразование углов крена и тангажа в радианы
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    # Генерация сетки зоны видимости в локальной системе координат
    dx = np.linspace(-half_swath_width, half_swath_width, num_points)
    dy = np.linspace(-swath_length / 2, swath_length / 2, num_points)
    dx, dy = np.meshgrid(dx, dy)

    # Поворот сетки по углу крена
    x_rotated = dx * math.cos(roll_rad) - dy * math.sin(roll_rad)
    y_rotated = dx * math.sin(roll_rad) + dy * math.cos(roll_rad)

    # Учет масштаба Земли при смещении центра зоны видимости
    d_lat_center = math.degrees(altitude * math.sin(pitch_rad) / R_earth)
    d_lon_center = math.degrees(
        altitude * math.sin(pitch_rad) / (R_earth * math.cos(math.radians(satellite_lat)))
    )

    visibility_center_lat = satellite_lat + d_lat_center
    visibility_center_lon = satellite_lon + d_lon_center

    # Преобразование сетки в географические координаты
    visibility_coords = []
    for i in range(num_points):
        for j in range(num_points):
            d_lat = math.degrees(y_rotated[i, j] / R_earth)
            d_lon = math.degrees(x_rotated[i, j] / (R_earth * math.cos(math.radians(visibility_center_lat))))
            lat = visibility_center_lat + d_lat
            lon = visibility_center_lon + d_lon

            # Учет минимального угла на Солнце
            surface_point = np.array([
                math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
                math.sin(math.radians(lat))
            ])
            sun_angle_cos = np.dot(surface_point, sun_vector) / (
                    np.linalg.norm(surface_point) * np.linalg.norm(sun_vector))

            # Добавление точки, если угол больше минимального
            if math.degrees(math.acos(sun_angle_cos)) >= sun_angle:
                visibility_coords.append((lat, lon))

    return visibility_coords


@app.route('/get_visibility_zone', methods=['POST'])
def get_visibility_zone():
    try:
        # Получение данных из запроса клиента
        data = request.get_json()
        title = data.get('title')
        roll = data.get('roll')
        pitch = data.get('pitch')

        # Проверка наличия всех обязательных данных
        if not title:
            return jsonify({'error': 'Satellite title is required'}), 400
        if roll is None or pitch is None:
            return jsonify({'error': 'Both roll and pitch parameters are required'}), 400

        # Получение данных о спутнике по title
        satellite_data = getSelectedSatellite(title)
        if not satellite_data:
            return jsonify({'error': 'Satellite not found'}), 404
        satellite = satellite_data[0]

        # Получение орбитальных параметров спутника
        semi_axis = satellite['semi_axis']
        eccentricity = satellite['eccentricity']
        inclination = math.radians(satellite['inclination'])
        periarg = math.radians(satellite['periarg'])
        ascnode = math.radians(satellite['ascnode'])

        # Чтение данных о радаре спутника
        all_satellites = read_satellite_radars()
        selected_unit = next((unit for unit in all_satellites if unit['title'].strip() == title.strip()), None)
        if not selected_unit:
            return jsonify({'error': 'Radar configuration not found'}), 404

        view_angle = selected_unit['view_angle']
        ap_angle = selected_unit['ap_angle']
        sun_angle = selected_unit['sun_angle']

        # Вычисление времени с начала миссии
        current_time = time.time()
        mission_start_time = time.strptime(satellite['mission_begin'], "%d.%m.%Y %H:%M:%S.%f")
        time_passed = current_time - time.mktime(mission_start_time)

        # Вычисление позиции спутника
        x_geo, y_geo, z_geo, r = calculate_satellite_position(semi_axis, eccentricity, inclination, periarg, ascnode,
                                                              time_passed)

        # Преобразование в широту и долготу
        lat = math.degrees(math.asin(z_geo / r))
        lon = math.degrees(math.atan2(y_geo, x_geo))

        # Высота спутника в километрах
        altitude_km = r - 6371
        current_timestamp = time.time()
        sun_vector = calculate_sun_vector(current_timestamp)

        # Вычисление зоны видимости
        coords = calculate_visibility_zone(roll, pitch, altitude_km, lat, lon, view_angle, ap_angle, sun_angle,
                                           sun_vector)

        # Возвращение координат зоны видимости
        return jsonify({'visibility_zone': coords}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
def calculate_satellite_position(semi_axis, eccentricity, inclination, periarg, ascnode, time_passed):
    # Константы
    MU = 398600.4418  # Гравитационный параметр Земли, км^3 / с^2
    EARTH_ROTATION_RATE = 2 * math.pi / 86400  # Угловая скорость вращения Земли, рад/с

    # Средняя аномалия
    n = math.sqrt(MU / (semi_axis ** 3))
    M = n * time_passed

    # Решение уравнения Кеплера
    E = M
    tol = 1e-8
    for _ in range(100):
        delta = (E - eccentricity * math.sin(E) - M) / (1 - eccentricity * math.cos(E))
        E -= delta
        if abs(delta) < tol:
            break

    # Истинная аномалия
    sin_true_anomaly = math.sqrt(1 - eccentricity**2) * math.sin(E) / (1 - eccentricity * math.cos(E))
    cos_true_anomaly = (math.cos(E) - eccentricity) / (1 - eccentricity * math.cos(E))
    true_anomaly = math.atan2(sin_true_anomaly, cos_true_anomaly)

    # Расстояние до спутника
    r = semi_axis * (1 - eccentricity * math.cos(E))

    # Геоцентрические координаты в плоскости орбиты
    x_orbit = r * math.cos(true_anomaly)
    y_orbit = r * math.sin(true_anomaly)

    # Преобразование в 3D-геоцентрическую систему
    x_geo = (math.cos(periarg) * math.cos(ascnode) - math.sin(periarg) * math.sin(ascnode) * math.cos(inclination)) * x_orbit \
          - (math.sin(periarg) * math.cos(ascnode) + math.cos(periarg) * math.sin(ascnode) * math.cos(inclination)) * y_orbit
    y_geo = (math.cos(periarg) * math.sin(ascnode) + math.sin(periarg) * math.cos(ascnode) * math.cos(inclination)) * x_orbit \
          + (math.cos(periarg) * math.cos(ascnode) * math.cos(inclination) - math.sin(periarg) * math.sin(ascnode)) * y_orbit
    z_geo = (math.sin(periarg) * math.sin(inclination)) * x_orbit \
          + (math.cos(periarg) * math.sin(inclination)) * y_orbit

    # Учет вращения Земли
    earth_rotation_angle = EARTH_ROTATION_RATE * time_passed
    x_rot = x_geo * math.cos(earth_rotation_angle) + y_geo * math.sin(earth_rotation_angle)
    y_rot = -x_geo * math.sin(earth_rotation_angle) + y_geo * math.cos(earth_rotation_angle)
    z_rot = z_geo

    return x_rot, y_rot, z_rot, r


@app.route('/get_satellite_positions', methods=['POST'])
def get_satellite_positions():
    try:
        # Получаем title спутника из запроса
        data = request.get_json()
        title = data.get('title')

        if not title:
            return jsonify({'error': 'Satellite title is required'}), 400

        # Получаем данные о спутнике по title
        satellite_data = getSelectedSatellite(title)
        if not satellite_data:
            return jsonify({'error': 'Satellite not found'}), 404

        # Предполагаем, что функция getSelectedSatellite возвращает список, берем первый элемент
        satellite = satellite_data[0]

        # Извлекаем параметры орбиты с проверкой на наличие значений
        try:
            semi_axis = satellite['semi_axis']
            eccentricity = satellite['eccentricity']
            inclination = math.radians(satellite['inclination'])
            periarg = math.radians(satellite['periarg'])
            ascnode = math.radians(satellite['ascnode'])
        except KeyError as e:
            return jsonify({'error': f'Missing satellite parameter: {str(e)}'}), 400

        # Вычисляем текущее положение
        current_time = time.time()

        # Обработка времени запуска миссии
        try:
            mission_start_time = time.strptime(satellite['mission_begin'], "%d.%m.%Y %H:%M:%S.%f")
            time_passed = current_time - time.mktime(mission_start_time)
        except (ValueError, KeyError) as e:
            return jsonify({'error': 'Invalid mission start time format or missing parameter.'}), 400

        # Вызов функции для вычисления позиции спутника
        x_geo, y_geo, z_geo, r = calculate_satellite_position(semi_axis, eccentricity, inclination, periarg, ascnode,
                                                              time_passed)

        # Проверка на нулевое значение r перед преобразованием
        if r == 0:
            return jsonify({'error': 'Calculated radius is zero, cannot compute latitude and longitude.'}), 500

        # Преобразование 3D координат в широту и долготу
        lat = math.degrees(math.asin(z_geo / r))
        lon = math.degrees(math.atan2(y_geo, x_geo))

        # Формируем ответ с данными о положении спутника
        position = {
            'title': satellite['title'],
            'lat': lat,
            'lon': lon,
            'radius': r,
        }

        return jsonify(position)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_satellite_past_trajectory', methods=['POST'])
def get_satellite_past_trajectory():
    try:
        # Получаем title спутника из запроса
        data = request.get_json()
        title = data.get('title')

        if not title:
            return jsonify({'error': 'Satellite title is required'}), 400

        # Получаем данные о спутнике по title
        satellite_data = getSelectedSatellite(title)
        if not satellite_data:
            return jsonify({'error': 'Satellite not found'}), 404

        # Предполагаем, что функция getSelectedSatellite возвращает список, берем первый элемент
        satellite = satellite_data[0]

        # Извлекаем параметры орбиты с проверкой на наличие значений
        try:
            semi_axis = satellite['semi_axis']
            eccentricity = satellite['eccentricity']
            inclination = math.radians(satellite['inclination'])
            periarg = math.radians(satellite['periarg'])
            ascnode = math.radians(satellite['ascnode'])
        except KeyError as e:
            return jsonify({'error': f'Missing satellite parameter: {str(e)}'}), 400

        # Обработка времени запуска миссии
        try:
            mission_start_time = time.strptime(satellite['mission_begin'], "%d.%m.%Y %H:%M:%S.%f")
            mission_start_timestamp = time.mktime(mission_start_time)
        except (ValueError, KeyError) as e:
            return jsonify({'error': 'Invalid mission start time format or missing parameter.'}), 400

        # Вычисляем текущее время
        current_time = time.time()

        # Генерируем временные метки за последний час (каждые 10 секунд, например)
        trajectory_points = []
        for t in range(0, 20000, 10):  # 3600 секунд в часе, шаг 10 секунд
            time_passed = (current_time - mission_start_timestamp) - t
            x_geo, y_geo, z_geo, r = calculate_satellite_position(semi_axis, eccentricity, inclination, periarg, ascnode, time_passed)

            # Проверка на нулевое значение r перед преобразованием
            if r == 0:
                continue  # Можно добавить обработку ошибки

            # Преобразование 3D координат в широту и долготу
            lat = math.degrees(math.asin(z_geo / r))
            lon = math.degrees(math.atan2(y_geo, x_geo))

            # Добавляем точку в траекторию
            trajectory_points.append({
                'time': current_time - t,  # Время точки
                'lat': lat,
                'lon': lon,
                'radius': r,
            })
            #print(f"t: {t}, lat: {lat}, lon: {lon}, r: {r}")  # Вывод для отладки
            

        # Формируем ответ с траекторией спутника
        trajectory = {
            'title': satellite['title'],
            'trajectory': trajectory_points,
        }

        return jsonify(trajectory)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Функция для создания слоя тайлов из .mbtiles файла
def serve_tile(z, x, y):
    con = sqlite3.connect(MBTILES_FILE)
    cur = con.cursor()
    y = (1 << z) - 1 - y  # Изменение координат Y для рендеринга карты
    cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, y))
    row = cur.fetchone()
    con.close()

    if row:
        return row[0]
    else:
        return None

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/ground')
def ground_view():
    return render_template('groundunits.html')

@app.route('/tiles/<int:z>/<int:x>/<int:y>.png')
def tiles(z, x, y):
    tile = serve_tile(z, x, y)
    if tile:
        return tile, 200, {'Content-Type': 'image/png'}
    else:
        return "Tile not found", 404

@app.route('/save_coords', methods=['POST'])
def save_coords():
    data = request.json
    with open('selected_areas.geojson', 'w') as f:
        json.dump(data, f, indent=4)  # Сохраняем в формате GeoJSON
    return jsonify({"status": "success", "message": "Coordinates saved!"})


@app.route('/submit', methods=['POST'])
def submit():
    # Получение данных из формы
    title = request.form['title']
    caption = request.form['caption']
    icon_width = request.form['icon_width']
    icon_style = request.form['icon_style']
    icon_pen_color = request.form['icon_pen_color']
    icon_pen_width = request.form['icon_pen_width']
    icon_pen_style = request.form['icon_pen_style']
    icon_brush_color = request.form['icon_brush_color']
    icon_brush_style = request.form['icon_brush_style']
    radar_color = request.form['radar_color']
    view_angle = request.form['view_angle']

    # Логика сохранения данных в INI файл (например, через библиотеку configparser)
    import configparser
    config = configparser.ConfigParser()

    config['Settings'] = {
        'Title': title,
        'Caption': caption,
        'IconWidth': icon_width,
        'IconStyle': icon_style,
        'IconPenColor': icon_pen_color,
        'IconPenWidth': icon_pen_width,
        'IconPenStyle': icon_pen_style,
        'IconBrushColor': icon_brush_color,
        'IconBrushStyle': icon_brush_style,
        'RadarColor': radar_color,
        'ViewAngle': view_angle,
    }

    # Сохранение INI файла
    with open('output.ini', 'w') as configfile:
        config.write(configfile)

    return "Данные сохранены в INI файл!"


def load_groundunits_ini():
    # Получаем базовый путь к проекту
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к файлу groundunits.ini относительно папки static
    ini_path = os.path.join(BASE_DIR, "static", "data", "groundunits", "groundunits.ini")
    
    # Путь к папке static для замены WORKPATH
    static_path = os.path.join(BASE_DIR, "static")
    
    # Проверка существования файла
    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"Файл groundunits.ini не найден по пути: {ini_path}")
    
    config = configparser.ConfigParser()
    config.read(ini_path, encoding='windows-1251')
    ground_units_paths = []

    for section in config.sections():
        if section.startswith("NKPOR_"):
            for _, path in config.items(section):
                # Заменяем WORKPATH на относительный путь к static
                full_path = path.split()[0].replace("WORKPATH", static_path)
                ground_units_paths.append(full_path)

    return ground_units_paths


def read_ground_units():
    ground_units = []
    object_paths = load_groundunits_ini()

    for path in object_paths:
        ini_file_path = os.path.join(path, 'params.ini')
        if os.path.exists(ini_file_path):
            config = configparser.ConfigParser()
            # Изменяем кодировку на windows-1251 для чтения файла
            config.read(ini_file_path, encoding='windows-1251')
            try:
                title = config.get('params', 'title', fallback='Неизвестно').split('/*')[0].strip()

                lat_str = config.get('mission.initcond', 'position.lat(deg)', fallback='0').split('/*')[0].strip()
                lon_str = config.get('mission.initcond', 'position.long(deg)', fallback='0').split('/*')[0].strip()

                lat = float(lat_str)
                lon = float(lon_str)

                ground_units.append({
                    'title': title,
                    'lat': lat,
                    'lon': lon,
                })
            except Exception as e:
                print(f"Ошибка при чтении {ini_file_path}: {e}")
        else:
            print(f"Файл {ini_file_path} не найден.")

    return ground_units

@app.route('/get_ground_units', methods=['GET'])
def get_ground_units():
        data = read_ground_units()
        return jsonify(data)


@app.route('/get_selected_ground_units', methods=['POST'])
def get_selected_ground_units():
    selected_ids = request.json.get('selected_ids', [])
    all_ground_units = read_ground_units()

    # Фильтруем по выбранным идентификаторам
    selected_units = [unit for unit in all_ground_units if unit['title'] in selected_ids]

    return jsonify(selected_units)

# Глобальная переменная для кэширования данных спутников
satellite_cache = None
cache_lock = threading.Lock()

def load_satellites_ini():
    # Получаем базовый путь к проекту
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к файлу satellites.ini относительно папки static
    ini_path = os.path.join(BASE_DIR, "static", "data", "satellites", "satellites.ini")
    
    # Путь к папке static для замены WORKPATH
    static_path = os.path.join(BASE_DIR, "static")
    
    # Проверка существования файла
    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"Файл satellites.ini не найден по пути: {ini_path}")

    config = configparser.ConfigParser()
    config.read(ini_path, encoding='windows-1251')
    satellites_paths = []

    for section in config.sections():
        if section.startswith("P-V"):
            for _, path in config.items(section):
                # Формируем полный путь относительно static
                full_path = path.split()[0].replace("WORKPATH", static_path)
                satellites_paths.append(full_path)

    return satellites_paths

def read_satellite_paths():
    global satellite_cache
    with cache_lock:
        if satellite_cache is not None:
            return satellite_cache

        satellite_paths = []
        object_paths = load_satellites_ini()

        for path in object_paths:
            ini_file_path = os.path.join(path, 'params.ini')
            if os.path.exists(ini_file_path):
                config = configparser.ConfigParser()
                config.read(ini_file_path, encoding='windows-1251')
                try:
                    title = config.get('params', 'title', fallback='Неизвестно').split('/*')[0].strip()
                    caption = config.get('params', 'caption', fallback='Неизвестно').split('/*')[0].strip()
                    mission_begin = config.get('params', 'mission.begin', fallback='Неизвестно').split('/*')[0].strip()
                    mission_finish = config.get('params', 'mission.finish', fallback='Неизвестно').split('/*')[0].strip()

                    semi_major_axis_str = config.get('mission.initcond', 'orbit.semiaxis(km)', fallback='0').split('/*')[0].strip()
                    eccentricity_str = config.get('mission.initcond', 'orbit.eccentricity', fallback='0').split('/*')[0].strip()
                    latitude_str = config.get('mission.initcond', 'orbit.latarg(deg)', fallback='0').split('/*')[0].strip()
                    periarg_str = config.get('mission.initcond', 'orbit.periarg(deg)', fallback='0').split('/*')[0].strip()
                    inclination_str = config.get('mission.initcond', 'orbit.inclination(deg)', fallback='0').split('/*')[0].strip()
                    ascnode_str = config.get('mission.initcond', 'orbit.ascnode(deg)', fallback='0').split('/*')[0].strip()
                    longitude_str = config.get('mission.initcond', 'orbit.long(deg)', fallback='0').split('/*')[0].strip()

                    Roll_str = config.get('params.shot', 'Roll(deg)', fallback='0').split('/*')[0].strip()
                    Pitch_str = config.get('params.shot', 'Pitch(deg)', fallback='0').split('/*')[0].strip()
                    Wroll_str = config.get('params.shot', 'Wroll(deg/sec)', fallback='0').split('/*')[0].strip()
                    Wpitch_str = config.get('params.shot', 'Wpitch(deg/sec)', fallback='0').split('/*')[0].strip()
                    Rev_time_str = config.get('params.shot', 'Rev_time(sec)', fallback='0').split('/*')[0].strip()
                    Day_time_str = config.get('params.shot', 'Day_time(sec)', fallback='0').split('/*')[0].strip()
                    Route_time_str = config.get('params.shot', 'Route_time(sec)', fallback='0').split('/*')[0].strip()
                    Route_between_time_str = config.get('params.shot', 'Route_between_time(sec)', fallback='0').split('/*')[0].strip()

                    semi_major_axis = float(semi_major_axis_str)
                    eccentricity = float(eccentricity_str)
                    latitude = float(latitude_str)
                    periarg = float(periarg_str) if periarg_str != '-NAN' else -100000000
                    inclination = float(inclination_str)
                    ascnode = float(ascnode_str)
                    longitude = float(longitude_str)
                    Wroll = float(Wroll_str)
                    Wpitch = float(Wpitch_str)
                    Rev_time = float(Rev_time_str)
                    Day_time = float(Day_time_str)
                    Route_between_time = float(Route_between_time_str)

                    satellite_paths.append({
                        'title': title,
                        'caption': caption,
                        'mission_begin': mission_begin,
                        'mission_finish': mission_finish,
                        'eccentricity': eccentricity,
                        'semi_axis': semi_major_axis,
                        'latitude': latitude,
                        'periarg': periarg,
                        'inclination': inclination,
                        'ascnode': ascnode,
                        'longitude': longitude,
                        'Roll_str': Roll_str,
                        'Pitch_str': Pitch_str,
                        'Wroll': Wroll,
                        'Wpitch': Wpitch,
                        'Rev_time': Rev_time,
                        'Day_time': Day_time,
                        'Route_time_str': Route_time_str,
                        'Route_between_time': Route_between_time,
                    })
                except ValueError as e:
                    print(f"Ошибка преобразования в число в файле {ini_file_path}: {e}")
                except Exception as e:
                    print(f"Ошибка при чтении {ini_file_path}: {e}")
            else:
                print(f"Файл {ini_file_path} не найден.")

        satellite_cache = satellite_paths
        return satellite_paths

@app.route('/get_satellites', methods=['GET'])
def get_satellite_paths():
    data = read_satellite_paths()
    return jsonify(data)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_value(value):
    """Удаляет комментарии и ненужные символы из строки."""
    # Убираем все после символа '/*' (включая сам символ) и очищаем пробелы
    value = re.sub(r'\s*/\*.*', '', value).strip()

    # Извлекаем только первое числовое значение (до точки или других символов)
    value = re.match(r'[\d\.]+', value)  # Ищем только первое число с точкой
    if value:
        return value.group(0)
    return '0.0'  # Возвращаем 0.0, если значение не удалось очистить


def load_satellite_data(file_path):
    """Загружает данные из INI файла спутника."""
    config = configparser.ConfigParser()
    config.read(file_path, encoding='windows-1251')

    satellite_data = {}

    try:
        # Считывание параметров из секции params
        satellite_data['title'] = config.get('params', 'title', fallback='Неизвестно').split('/*')[0].strip()
        satellite_data['caption'] = clean_value(config.get('params', 'caption', fallback='Неизвестно'))
        satellite_data['icon_width'] = int(clean_value(config.get('params', 'icon.width', fallback='0')))
        satellite_data['icon_style'] = int(clean_value(config.get('params', 'icon.style', fallback='0')))
        satellite_data['icon_pen'] = clean_value(
            config.get('params', 'icon.pen', fallback='rgb->(0,0,0);width->1;style->0'))
        satellite_data['icon_brush'] = clean_value(
            config.get('params', 'icon.brush', fallback='rgb->(0,0,139);style->0'))
        satellite_data['color_radar'] = clean_value(config.get('params', 'color.radar', fallback='(255,215,0)'))

        # Считывание параметров из секции params.shot
        satellite_data['view_angle'] = float(clean_value(config.get('params.shot', 'ViewAngle(deg)', fallback='0.0')))
        satellite_data['ap_angle'] = float(clean_value(config.get('params.shot', 'APangle(deg)', fallback='0.0')))
        satellite_data['sun_angle'] = float(clean_value(config.get('params.shot', 'SunAngle(deg)', fallback='0.0')))

        # Обработка параметра, который может содержать дополнительные символы
        record_speed = clean_value(config.get('params.shot', 'Record_speed(gbit/sec)', fallback='0.0'))
        satellite_data['record_speed'] = float(record_speed)

    except configparser.Error as e:
        logging.error(f"Ошибка при чтении файла {file_path}: {e}")

    return satellite_data


def read_satellite_radars():
    """Читает данные всех радаров спутников из соответствующих INI файлов."""
    satellite_paths = []
    object_paths = load_satellites_ini()  # Этот метод должен быть реализован для получения путей

    for i, path in enumerate(object_paths, start=1):
        satellite_file = f"00{str(i).zfill(2)}01_radar_P-V_mss.ini"
        satellite_ini_path = os.path.join(path, satellite_file)

        # Логирование поиска файла
        #logging.info(f"Пытаемся загрузить файл: {satellite_ini_path}")

        if os.path.exists(satellite_ini_path):
            # Считываем данные из ini файла спутника
            satellite_data = load_satellite_data(satellite_ini_path)
            satellite_paths.append(satellite_data)
            #logging.info(f"Данные для спутника {i} загружены.")
        else:
            logging.warning(f"Файл {satellite_ini_path} не найден.")

    # Логирование завершения работы функции
    #logging.info(f"Загружено данных для {len(satellite_paths)} спутников.")

    return satellite_paths

@app.route('/get_selected_satellites', methods=['POST'])
def get_selected_satellites():
    selected_ids = request.json.get('selected_paths', [])
    #print("Получены выбранные пути:", selected_ids)  # Для отладки
    all_satellites = read_satellite_paths()
    selected_units = [unit for unit in all_satellites if unit['title'] in selected_ids]
    #print("Отправляем спутники:", selected_units)  # Для отладки

    return jsonify(selected_units)

def getSelectedSatellite(title):
    all_satellites = read_satellite_paths()
    selected_unit = [unit for unit in all_satellites if unit['title'] == title]
    #print("выбраны:", selected_unit)
    return selected_unit

if __name__ == '__main__':
    app.run(debug=True)