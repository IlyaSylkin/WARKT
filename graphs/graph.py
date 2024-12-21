import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Чтение данных из файла
df = pd.read_csv('Data.csv', sep=';')

# Извлечение данных из таблицы
Time = df['Time']
Velocity = df['Velocity']
Altitude = df['AltitudeFromTerrain']

# Ограничиваем данные до 360 секунд
Time_filtered = Time[Time <= 360]
Velocity_filtered = Velocity[:len(Time_filtered)]
Altitude_filtered = Altitude[:len(Time_filtered)]

# Модифицированная скорость (делим на 2.8)
Velocity_modified = Velocity_filtered / 3

# Вычисление ускорения от модифицированной скорости (разница скорости / разница времени)
Acceleration = np.diff(Velocity_modified) / np.diff(Time_filtered)

# Добавление нуля в начало списка ускорений (для совместимости с длиной Time_filtered)
Acceleration = np.insert(Acceleration, 0, 0)

# Для сглаживания ускорения применим скользящее среднее
window_size = 15  # Размер окна для сглаживания
Acceleration_smooth = np.convolve(Acceleration, np.ones(window_size) / window_size, mode='same')


# Математическая модель
G = 6.6743 * 10**(-11)
Mz = 5.974 * 10**24
m0 = 571585
n1 = 1900
n2 = 20
u = 3000
fuel1 = 500265
fuel2 = 20830
msuh1 = 38247
msuh2 = 2243
m1 = msuh1 + fuel1
m2 = msuh2 + fuel2
Mgraph = []
V = []
Tacc = np.arange(1, 400, 1)
v0 = 0
F1 = 4248400
F2 = 749200

# Функция для расчета тяготы
def Ftyag(m, AltitudeFromTerrain):
    return G * ((Mz * m) / (6400000 + AltitudeFromTerrain)**2)

# Моделирование до 360 секунд
for t in range(1, 150):  # Период с ускорением n1
    m = m0 - n1 * t
    Mgraph.append(m)
    v1 = (v0 - u * np.log(m / m0) - (G * Mz * np.log(m / m0)) / (6400000 ** 2))
    d = v1 / 1000 * 4
    V.append(d)
    v0 = v1

    # Если время превышает 360 секунд, выходим из цикла
    if t >= 360:
        break

m0 = m0 - fuel1 - msuh1

for t in range(150, 1095):  # Период с ускорением n2
    m = m0 - n2 * t
    Mgraph.append(m)
    v1 = (v0 - u * np.log(m / m0) - (G * Mz * np.log(m / m0)) / (6400000 ** 2))
    d = v1 / 1000 * 4
    V.append(d)
    v0 = v1

    # Если время превышает 360 секунд, выходим из цикла
    if t >= 360:
        break

m0 = m0 - fuel2 - msuh2

# Рассчитаем ускорение и высоту на основе модели
Acc = [0] * len(V)
T = np.arange(1, len(V) + 1)
Th = np.arange(1, 501, 1)
H = [0] * 500

for t in range(len(V)):
    if t < 150:
        Acc[t] = ((u * n1 * t - Ftyag(m1 - (4 * 1120 + 1818) * (t + 1), Altitude_filtered[t])) / m1) / 80
        H[t + 1] = (Acc[t] * ((t + 1) ** 2)) / 200
    else:
        Acc[t] = ((u * n2 * t - Ftyag(m2 - 2281 * 2 * (t + 1), Altitude_filtered[t])) / m2) / 80
        H[t + 1] = (Acc[t] * ((t + 1) ** 2)) / 200

    # Если время превышает 360 секунд, выходим из цикла
    if t >= 360:
        break
# Умножаем каждое значение высоты на 12.5
H = np.array(H) * 11.6


# Умножаем всю скорость на 3.4
V = np.array(V) * 8



plt.plot(Time_filtered, Altitude_filtered, label="Высота (KSP)", color="blue")
plt.plot(T[:len(V)], H[:len(V)], label="Высота (Модель)", color="red", linestyle='--')

plt.title("Сравнение высоты от времени")
plt.xlabel("Время (с)")
plt.ylabel("Высота (м)")
plt.legend()
plt.grid(True)
plt.show()

# Наложение графиков скорости (KSP и модель)
plt.plot(Time_filtered, Velocity_filtered, label="Скорость (KSP)", color="blue")
plt.plot(T[:len(V)], V[:len(V)], label="Скорость (Модель)", color="red", linestyle='--')

plt.title("Сравнение скорости от времени")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (м/с)")
plt.legend()
plt.grid(True)
plt.show()

# Наложение графиков ускорения (KSP и модель)
plt.plot(Time_filtered, Acceleration_smooth, label="Ускорение (KSP)", color="blue")
plt.plot(T[:len(V)], Acc[:len(V)], label="Ускорение (Модель)", color="red", linestyle='--')

plt.title("Сравнение ускорения от времени")
plt.xlabel("Время (с)")
plt.ylabel("Ускорение (м/с²)")
plt.legend()
plt.grid(True)
plt.show()