import gzip
import sys
import struct
import time
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import struct

start_time = time.time()

file_path = 'E:\\f15\\ssies\\2001\\08\\'
file_name = 'f15dm01aug10.dat.gz'
file_name = 'f15dm01aug15.dat.gz'
file_name = 'f15dm01aug26.dat.gz'

read_file = file_path+file_name

data = gzip.open(file_path + file_name, 'rb').read()

amount_of_records = int(sys.getsizeof(data)/2292)
print(amount_of_records)
#print(data)
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

    offset = start_index
    for _ in range(second_count):
        second_data_converted = []
        second_nonconverted = []
        for key, value in second_template.items():
            byte_size, scale, offset_value = value[1], value[2], value[3]
            field_bytes = data[offset:offset + byte_size]
            field_value = int.from_bytes(field_bytes, byteorder="big", signed=True)
            physical_value = (float(field_value) / scale) + offset_value
            real_value = float(field_value)
            second_data_converted.append(physical_value)
            second_nonconverted.append(real_value)
            offset += byte_size
        second_results.append(second_data_converted)
        second_data_nonconverted.append(second_nonconverted)

    return second_results, second_data_nonconverted


nonconverted_data = []
converted_data = []

for i in range(amount_of_records):
    record, converted_record = parse_binary_data(data, template_data, i * 2292)
    nonconverted_data.append(record)
    converted_data.append(converted_record)

columns = []
for key, value in template_data.items():
        columns.append(str(key))

data_table = pd.DataFrame(nonconverted_data)
converted_data_table = pd.DataFrame(converted_data)

data_table[0] = data_table[4] * 60 + data_table[5]
converted_data_table[0] = converted_data_table[4] * 60 + converted_data_table[5]

data_table = data_table.dropna()
converted_data_table = converted_data_table.dropna()

print("Данные 'как есть':")
print(data_table.head())

print("\nПреобразованные данные:")
print(converted_data_table.head())

print(f'Время = {time.time()-start_time}')
file_size = len(data)
expected_size = amount_of_records * 2292
print(f"Размер файла: {file_size}, Ожидаемый размер: {expected_size}")

