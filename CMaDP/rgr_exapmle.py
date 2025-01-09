import gzip
import sys
import struct
import time
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

start_time = time.time()

file_path = 'E:\\f15\\ssies\\2001\\09\\'
file_name = 'f15dm01sep11.dat.gz'
read_file = file_path + file_name

data = gzip.open(read_file, 'rb').read()

amount_of_records = int(sys.getsizeof(data) / 2292)
print(amount_of_records)

template_data = {
    1:['Spececraft ID', 'Номер космического аппарата', 5],
    2:['Data file ID', 'ID файла', 6],
    3:['Integer year', 'Год', 2, 1, 1950],
    4:['Day of year', 'День в году', 2, 1, 0],
    5:['Hour of day', 'Час дня', 1, 1, 0],
    6:['Minute of hour', 'Минута часа', 1, 1, 0],
    7:['Geodetic latitude', 'Геодезическая широта', 2, 10, -90],
    8:['Geographic longitude', 'Географическая долгота', 2, 10, 0],
    9:['Geomagnetic latitude', 'Геомагнитная широта', 2, 10, -90],
    10:['Magnetic local time at 110 km field line intercept', 
        'Местное магнитное время при пересечении линии поля на расстоянии 110 км.', 2, 10, 0],
    11:['Geomagnetic longitude at sub-satellite point', 'Геомагнитная долгота в подспутниковой точке', 2, 10, 0],
    12:['Geographic latitude of sub-solar point', 'Географическая широта подсолнечной точки', 2, 10, -90],
    13:['Geographic longitude of sub-solar poin', 'Географическая долгота подсолнечной точки', 2, 10, 0],
    14:['Geographic latitude DMSP spacecraft', 'Географическая широта космического корабля DMSP', 2, 10, -90],
    15:['Geographic longitude DMSP spacecraft', 'Географическая долгота космического корабля DMSP', 2, 10, 0],
    16:['Corrected geomagnetic latitude at 110 km altitude', 
        'Исправленная геомагнитная широта на высоте 110 км.', 2, 10, -90],
    17:['Corrected geomagnetic longitude at 110 km altitude', 
        'Скорректированная геомагнитная долгота на высоте 110 км.', 2, 10, 0], 
    18:['Invariant latitude', 'Неизменная широта', 2, 10, 0],
    19:['Altitude at the start of the minute', 'Высота над уровнем моря в начале минуты', 2, 1, 0],
    20:['Altitude at the end of the minute', 'Высота над уровнем моря в конце минуты', 2, 1, 0],
    21:['Northward component of model magnetic field at satellite', 
        'Северная составляющая модельного магнитного поля на спутнике', 4, 10, -70000],
    22:['Eastward component of model magnetic field at satellite', 
        'Восточная составляющая модельного магнитного поля на спутнике', 4, 10, -70000],
    23:['Downward component of model magnetic field at satellite', 
        'Нисходящая составляющая модельного магнитного поля на спутнике', 4, 10, -70000],
    24:['Geographic x-component', 'Географическая x-координата', 3, 100000, -1],
    25:['Geographic y-component', 'Географическая y-координата', 3, 100000, -1],
    26:['Geographic z-component', 'Географическая z-координата', 3, 100000, -1],
    27:['Potential control model flag', 'Флаг потенциальной модели управления', 1, 1, 0],
    28:['Potential difference between spacecraft and electron probe ground', 
        'Потенциальная разница между космическим кораблем и электронным зондом', 1, 1, -10],
    29:['Potential difference between ion array and electron probe ground', 
        'Потенциальная разница между ионной решеткой и заземлением электронного зонда', 1, 1, -3],
    30:['Drift meter repeller grid functions', 'Функции сетки отпугивателя дрейфового измерителя', 1, 1, 0],
    31:['Scintillation meter filter range commands', 'Команды диапазона фильтрации сцинтилляционного измерителя', 1, 1, 0],
    32:['No. of seconds of data for this minute', 'Количество секунд данных за эту минуту', 1, 1, 0],
    33:['Second of minute', 'Секунда минуты', 1, 1, 0],
    34:['Vertical speed, 1st sample of sec', 'Вертикальная скорость, 1-я выборка в сек.', 2, 0.1, -3000],
    35:['Vertical speed, 2nd sample of sec', 'Вертикальная скорость, 2-я выборка в сек.', 2, 0.1, -3000],
    36:['Vertical speed, 3rd sample of sec', 'Вертикальная скорость, 3-я выборка в сек.', 2, 0.1, -3000],
    37:['Vertical speed, 4th sample of sec', 'Вертикальная скорость, 4-я выборка в сек.', 2, 0.1, -3000],
    38:['Vertical speed, 5th sample of sec', 'Вертикальная скорость, 5-я выборка в сек.', 2, 0.1, -3000],
    39:['Vertical speed, 6th sample of sec', 'Вертикальная скорость, 6-я выборка в сек.', 2, 0.1, -3000],
    40:['Horizontal speed, 1st sample of sec', 'Горизонтальная скорость, 1-я выборка в сек.', 2, 0.1, -3000],
    41:['Horizontal speed, 2nd sample of sec', 'Горизонтальная скорость, 2-я выборка в сек.', 2, 0.1, -3000],
    42:['Horizontal speed, 3rd sample of sec', 'Горизонтальная скорость, 3-я выборка в сек.', 2, 0.1, -3000],
    43:['Horizontal speed, 4th sample of sec', 'Горизонтальная скорость, 4-я выборка в сек.', 2, 0.1, -3000],
    44:['Horizontal speed, 5th sample of sec', 'Горизонтальная скорость, 5-я выборка в сек.', 2, 0.1, -3000],
    45:['Horizontal speed, 6th sample of sec', 'Горизонтальная скорость, 6-я выборка в сек.', 2, 0.1, -3000],
    46:['Ratio of LLA/LLB', 'Соотношение LLA/LLB', 2, 1, 0],
    47:['Measured aperture potential', 'Измеренный потенциал апертуры', 2, 100, -19],
    48:['Zero fill', 'Нулевая заливка', 8, 1, 0]
    }


# Функции parse_binary_data и process_second_data остаются без изменений
def parse_binary_data(data, template_data, start_index=0):
    offset = start_index
    minute_data = []
    minute_converted = []
    for key, value in template_data.items():
        name_en, name_ru, byte_size = value[:3]
        scale = value[3] if len(value) > 3 else 1
        offset_value = value[4] if len(value) > 4 else 0
        field_bytes = data[offset:offset + byte_size]
        if key <= 2:
            field_value = field_bytes.decode('utf-8').strip()
        else:
            field_value = int.from_bytes(field_bytes, byteorder="big", signed=True)
        minute_data.append(field_value)
        if key > 2:
            physical_value = (float(field_value) / scale) + offset_value
            minute_converted.append(physical_value)
        else:
            minute_converted.append(0)
        offset += byte_size
        if key == 32:
            second_count = field_value
        if key == 33:
            second_data_results, second_data_nonconverted = process_second_data(data, offset, second_count)
            for second in second_data_results:
                minute_converted.extend(second)
            for second in second_data_nonconverted:
                minute_data.extend(second)
            break
    return minute_data, minute_converted


def process_second_data(data, start_index, second_count):
    second_results = []
    second_data_nonconverted = []
    second_template = {
        33: ['Second of minute', 1, 1, 0],
        34: ['Vertical speed, 1st sample of sec', 2, 0.1, -3000],
        35: ['Vertical speed, 2nd sample of sec', 2, 0.1, -3000],
        36: ['Vertical speed, 3rd sample of sec', 2, 0.1, -3000],
        37: ['Vertical speed, 4th sample of sec', 2, 0.1, -3000],
        38: ['Vertical speed, 5th sample of sec', 2, 0.1, -3000],
        39: ['Vertical speed, 6th sample of sec', 2, 0.1, -3000],
        40: ['Horizontal speed, 1st sample of sec', 2, 0.1, -3000],
        41: ['Horizontal speed, 2nd sample of sec', 2, 0.1, -3000],
        42: ['Horizontal speed, 3rd sample of sec', 2, 0.1, -3000],
        43: ['Horizontal speed, 4th sample of sec', 2, 0.1, -3000],
        44: ['Horizontal speed, 5th sample of sec', 2, 0.1, -3000],
        45: ['Horizontal speed, 6th sample of sec', 2, 0.1, -3000],
        46: ['Ratio of LLA/LLB', 2, 1, 0],
        47: ['Measured aperture potential', 2, 100, -19],
        48: ['Zero fill', 8, 1, 0],
    }
    for _ in range(second_count):
        second_data = []
        converted_data = []
        for key, value in second_template.items():
            name_en, byte_size, scale, offset_value = value
            field_bytes = data[start_index:start_index + byte_size]
            field_value = int.from_bytes(field_bytes, byteorder="big", signed=True)
            second_data.append(field_value)
            physical_value = (float(field_value) / scale) + offset_value
            converted_data.append(physical_value)
            start_index += byte_size
        second_results.append(converted_data)
        second_data_nonconverted.append(second_data)
    return second_results, second_data_nonconverted


def preprocess_minute_data(minute_data):
    # Преобразование в массив для работы с NaN и аномалиями
    data_array = np.array(minute_data, dtype=float)
    
    # Замена NaN на 0
    data_array[np.isnan(data_array)] = 0
    
    # Фильтрация аномалий (например, ограничение диапазона [-10^6, 10^6])
    data_array = np.clip(data_array, -1e6, 1e6)
    
    # Усреднение данных по секундам
    averaged_data = np.mean(data_array, axis=0)
    
    return averaged_data


# Разбор данных
parsed_data = []
converted_data = []
offset = 0

for _ in range(amount_of_records):
    minute_data, minute_converted = parse_binary_data(data, template_data, offset)
    parsed_data.append(minute_data)
    converted_data.append(minute_converted)
    offset += 2292

# Создание DataFrame
columns = [template_data[key][0] for key in template_data.keys()]
print(len(columns))
df_parsed = pd.DataFrame(parsed_data) #, columns=columns
print('parsed data')
print(df_parsed)

df_converted = pd.DataFrame(converted_data) #, columns=columns
print('converted data')
print(df_converted)

#print('conv data')
#print(converted_data)

#filtered_converted_data = []
#for minute in converted_data:
#    filtered_data = preprocess_minute_data(minute)
#    filtered_converted_data.append(filtered_data)

df_parsed = df_parsed.fillna(0)
df_converted = df_converted.fillna(0)

for col_index in range(32, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if 1 <= x <= 59 else 0)

for col_index in range(33, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(34, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(35, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(36, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(37, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(38, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(39, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(40, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(41, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(42, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(43, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

for col_index in range(44, len(df_converted.columns), 16):
    df_converted.iloc[:, col_index] = df_converted.iloc[:, col_index].apply(lambda x: x if -3000 <= x <= 3000 else 0)

df_converted.iloc[:, 28] = df_converted.iloc[:, 28].apply(lambda x: x if -3 <= x <= 0 else 0)

df_converted.iloc[:, 27] = df_converted.iloc[:, 27].apply(lambda x: x if -3 <= x <= 28 else 0)

df_converted.iloc[:, 25] = df_converted.iloc[:, 25].apply(lambda x: x if 0.0 <= x <= 1.0 else 0)

df_converted.iloc[:, 24] = df_converted.iloc[:, 24].apply(lambda x: x if 0.0 <= x <= 1.0 else 0)

df_converted.iloc[:, 23] = df_converted.iloc[:, 23].apply(lambda x: x if 0.0 <= x <= 1.0 else 0)

df_converted.iloc[:, 22] = df_converted.iloc[:, 22].apply(lambda x: x if -70000 <= x <= 70000 else 0)

df_converted.iloc[:, 21] = df_converted.iloc[:, 21].apply(lambda x: x if -70000 <= x <= 70000 else 0)

df_converted.iloc[:, 20] = df_converted.iloc[:, 20].apply(lambda x: x if -70000 <= x <= 70000 else 0)

df_converted.iloc[:, 19] = df_converted.iloc[:, 19].apply(lambda x: x if 400 <= x <= 500 else 0)

df_converted.iloc[:, 18] = df_converted.iloc[:, 18].apply(lambda x: x if 400 <= x <= 500 else 0)

df_converted.iloc[:, 17] = df_converted.iloc[:, 17].apply(lambda x: x if 0.0 <= x <= 90.0 else 0)

df_converted.iloc[:, 16] = df_converted.iloc[:, 16].apply(lambda x: x if 0.0 <= x <= 360.0 else 0)

df_converted.iloc[:, 15] = df_converted.iloc[:, 15].apply(lambda x: x if -90.0 <= x <= 90.0 else 0)

df_converted.iloc[:, 14] = df_converted.iloc[:, 14].apply(lambda x: x if 0.0 <= x <= 360.0 else 0)

df_converted.iloc[:, 13] = df_converted.iloc[:, 13].apply(lambda x: x if -90.0 <= x <= 90.0 else 0)

df_converted.iloc[:, 12] = df_converted.iloc[:, 12].apply(lambda x: x if 0.0 <= x <= 360.0 else 0)

df_converted.iloc[:, 11] = df_converted.iloc[:, 11].apply(lambda x: x if -90.0 <= x <= 90.0 else 0)

df_converted.iloc[:, 10] = df_converted.iloc[:, 10].apply(lambda x: x if 0.0 <= x <= 360.0 else 0)

df_converted.iloc[:, 9] = df_converted.iloc[:, 9].apply(lambda x: x if 0.0 <= x <= 24.0 else 0)

df_converted.iloc[:, 8] = df_converted.iloc[:, 8].apply(lambda x: x if -90.0 <= x <= 90.0 else 0)

df_converted.iloc[:, 7] = df_converted.iloc[:, 7].apply(lambda x: x if 0 <= x <= 360 else 0)

df_converted.iloc[:, 6] = df_converted.iloc[:, 6].apply(lambda x: x if -90.0 <= x <= 90.0 else 0)

print('CORRECTED')
print(df_converted)

average_columns = [f"avg_measurement_{i+1}" for i in range(15)]

# Создаем пустой DataFrame с нужными столбцами
averages_df = pd.DataFrame(columns=average_columns)

# Для каждой строки
for row_index, row in df_parsed.iterrows():
    # Извлекаем все данные после 32-го столбца
    measurements = row[33:].values
    
    # Проверяем, делится ли количество данных на 15
    if len(measurements) % 15 == 0:
        # Преобразуем данные в массив, имеющий 15 столбцов
        reshaped_measurements = measurements.reshape(-1, 15)
        
        # Усредняем данные по каждому из 15 замеров
        averages = reshaped_measurements.mean(axis=0)
        
        # Добавляем усредненные значения в новый DataFrame
        averages_df.loc[row_index] = averages

# Убираем столбцы после 33-го в исходном DataFrame
df_parsed_trimmed = df_parsed.iloc[:, :33]

# Склеиваем два DataFrame (обрезанный и усредненный)
result_df = pd.concat([df_parsed_trimmed, averages_df], axis=1)

print(result_df)
# Использование xarray для представления данных
xr_data = xr.DataArray(
    result_df,
    dims=["record", "field"],
    coords={"record": range(len(result_df)), "field": columns}
)

print(xr_data)

# Построение графиков с matplotlib
plt.figure(figsize=(10, 6))
plt.plot(result_df[3], result_df[6], label='Geodetic latitude')
plt.plot(result_df[3], result_df[8], label='Geomagnetic latitude')
plt.xlabel('Day of Year')
plt.ylabel('Latitude')
plt.title('Latitude vs Day of Year')
plt.legend()
plt.grid()
plt.show()

# 1. Временные ряды: Изменение высоты с течением времени
plt.figure(figsize=(10, 6))
plt.plot(result_df[5], result_df[18], label='Altitude')
plt.xlabel('Time (Minute of Hour)')
plt.ylabel('Altitude (km)')
plt.title('Изменение высоты с течением времени')
plt.legend()
plt.grid()
plt.show()

# 2. Географические карты: Карта высот по широте и долготе
plt.figure(figsize=(10, 6))
plt.scatter(result_df[7], result_df[6], c=result_df[18], cmap='viridis')
plt.colorbar(label='Altitude (km)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Карта высот по широте и долготе')
plt.show()

# 3. Векторные поля: Магнитное поле (Северная и Восточная компоненты)
plt.figure(figsize=(10, 6))
plt.quiver(result_df[7], result_df[6], 
           result_df[20], 
           result_df[21], 
           scale=1e5, color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Магнитное поле (Северная и Восточная компоненты)')
plt.show()

# 4. Тепловая карта: Корреляция между магнитными и географическими параметрами
plt.figure(figsize=(10, 8))
cols = [20, 
        21, 
        22, 
        6, 7]
sns.heatmap(result_df[cols].corr(), annot=True, cmap='coolwarm')
plt.title('Корреляция между магнитными и географическими параметрами')
plt.show()

# 5. Вертикальная и горизонтальная скорости
plt.figure(figsize=(10, 6))
plt.plot(result_df[5], result_df['avg_measurement_1'], label='Vertical Speed')
plt.plot(result_df[5], result_df['avg_measurement_7'], label='Horizontal Speed')
plt.xlabel('Time (Second of Minute)')
plt.ylabel('Speed (m/s)')
plt.title('Вертикальная и горизонтальная скорости')
plt.legend()
plt.grid()
plt.show()

# 6. Пространственные 3D-графики: Пространственное распределение высоты
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result_df[7], result_df[6], result_df[18], c=result_df[18], cmap='viridis')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude (km)')
plt.title('Пространственное распределение высоты')
plt.show()

# 7. Гистограмма: Распределение соотношения LLA/LLB
plt.figure(figsize=(10, 6))
plt.hist(result_df['avg_measurement_13'], bins=20, color='orange', alpha=0.7)
plt.xlabel('Ratio of LLA/LLB')
plt.ylabel('Frequency')
plt.title('Распределение соотношения LLA/LLB')
plt.show()


print(f"Время выполнения: {time.time() - start_time:.2f} секунд")

