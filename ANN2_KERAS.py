import os
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
#import Include.ANN_v2 as A


# ИНС В НОТАЦИИ KERAS
#==========================================================================================

def getArray2(FilePath, Filename):
    _file = open(FilePath + '\\' + Filename)
    line = _file.read()
    vec = line.split(', ')
    vec2 = []
    for i in range(len(vec)):
        vec2.append([float(vec[i])])
    _file.close()
    return vec2


def getArray2_v2(FPath, FX, FY, Num):  #Чтение данных из агрегированного файла
    X_File = open(FPath + '\\' + FX)
    Y_File = open(FPath + '\\' + FY)
    lineX = X_File.read()
    lineY = Y_File.read()
    vecX = lineX.split(', ')
    vecY = lineY.split(', ')
    Arr = []
    for item in range(Num):
        Arr.append([[float(vecY[item])], []])
        for item2 in range(item*400,(item+1)*400):
            Arr[item][1].append([float(vecX[item2])])
    X_File.close()
    Y_File.close()
    return Arr


def getArray2_v3(FilePath, Filename):
    _file = open(FilePath + '\\' + Filename)
    line = _file.read()
    vec = line.split(', ')
    vec2 = []
    for i in range(len(vec)):
        vec2 += [vec[i]]
    _file.close()
    return vec2


def getArray400(FPath, File, Num):  #Чтение данных из агрегированного файла
    _File = open(FPath + '\\' + File)
    Line = _File.read()
    vec = Line.split(', ')
    Arr = []
    for item in range(Num):
        Arr2 = []
        for item2 in range(item*400,(item+1)*400):
            Arr2.append([float(vec[item2])])
        Arr.append(Arr2)
    _File.close()
    return Arr


def getArray3(Beg, End, FilePath, Arr):
    for item in range(Beg, End):
        #Arr[item][0] = float(os.listdir(FilePath)[item][:4])
        #Arr[item][1] = getArray(FilePath, os.listdir(FilePath)[item])
        Arr.append([[float(os.listdir(FilePath)[item][:4])], []])


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


#====================================================================================================

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))


directory = r"C:\ANALYZER\FULL XY"
#directory = r"C:\ANALYZER\train_db"

numproc = 4  # Количество процессов
numf = 52284  # Количество файлов
numf2 = 953

# Восстановление данных из новых файлов========================================
start_time = time.time()
print('Восстанавливаю данные из файлов...')

FArrX = getArray400(directory, 'FULLX.txt', numf)
FArrY = getArray2(directory, 'FULLY.txt')
#FArrY = getArray2_v3(directory, 'FULLY.txt')
#print(FArrX)
#print(FArrY)

#Arr = getArray2_v2(directory, 'FULLX.txt', 'FULLY.txt', numf)
#print(Arr)

print('ВОССТАНОВЛЕНИЕ ДАННЫХ ИЗ ФАЙЛОВ УСПЕШНО!!! --- %s seconds ---' % (time.time() - start_time))
# =============================================================================


# Преобразование в NumPy-массивы======================================
start_time = time.time()
print('Выполняю преобразование списков в массивы NumPy')

#ArrX = np.array([])
#ArrX = np.vstack(FArrX)
#ArrY = np.array([])
#ArrY = np.vstack(FArrY)

#print(ArrX)
#print(ArrY)
#print(len(ArrX[1]))
#print(len(ArrY))

#XY_data = np.array([])  # Для представления в TensorFlow в виде NumPy-массива аргументов
#XY_data = np.vstack(Arr)

#print(Arr2)
#print(Arr2[1][1][6])
#print(Arr2[0][1][0] + Arr2[1][1][0])

X_data = []  # Аргументы функции для обучения
Y_data = []  # Значения функции для обучения
X_data2 = np.asarray(FArrX)
X_data = X_data2.reshape(52284,400)

#Y_data2 = np.asarray(FArrY)
#Y_data = np.reshape(1,52284)

Y_data = np.vstack(FArrY)
#Y_data = np.ravel(FArrY)
#Y_data = np.hstack(FArrY)
#Y_data = np.asarray(FArrY)
#Y_data = list(FArrY)
#Y_data = Y_data2.reshape(1,52284)

#print(X_data)
#print(X_data[0])
#print(X_data.shape)
#print('************************')
#print('=============================================')
#print(Y_data)
#print(Y_data.shape)


print('ПРЕОБРАЗОВАНИЕ ВЫПОЛНЕНО! --- %s seconds ---' % (time.time() - start_time))
# ====================================================================

#print(X_data.max())    # 7.49809 - Максимальный X
#print(X_data.min())     # -6.86429 - Минимальный X
#print(Y_data.max())    # 1100.0 - Максимальный Y

#X_data = (X_data + X_data.min()) / X_data.max()
#Y_data = Y_data / Y_data.max()
#Y_data = Y_data / 194


# ******************НЕЙРОННАЯ СЕТЬ************************

model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(400,)),
    keras.layers.Dense(400, input_shape=(400,), activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1158, activation=tf.nn.softmax)
])

#loss='sparse_categorical_crossentropy',

model.compile(
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              #optimizer=tf.train.AdamOptimizer(0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_data, Y_data, epochs=10)
model.summary()


model.save(directory + '\my_model.h5')


#********************ТЕСТ НЕЙРОННОЙ СЕТИ**********************************************************
'''
#Подгрузка тестовых данных =======================================================================
start_time = time.time()
print('Подгружаю тестовые данные...')
FTArrX = getArray400(directory, 'TEST_X.txt', numf2)
FTArrY = getArray2(directory, 'TEST_Y.txt')
#print(FTArrX)
#print(FTArrY)
print('ПОДГРУЗКА ТЕСТОВЫХ ДАННЫХ ЗАВЕРШЕНА!  --- %s seconds ---' % (time.time() - start_time))

#Преобразование данных в NumPy ===================================================================

start_time = time.time()
print('Преобразовываю данные в массивы NumPy...')
XT_data = []  # Аргументы функции для обучения
YT_data = []  # Значения функции для обучения
XT_data2 = np.asarray(FTArrX)
XT_data = XT_data2.reshape(numf2,400)
YT_data = np.vstack(FTArrY)
#print(XT_data)
#print(YT_data)
print('ПРЕОБРАЗОВАНИЕ ДАННЫХ ЗАВЕРШЕНО!  --- %s seconds ---' % (time.time() - start_time))

# Оценка точности ИНС ========================================

loss, acc = model.evaluate(XT_data, YT_data)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Предсказание результатов ===================================

predictions = model.predict(XT_data)

for i in range(len(predictions)):
    print('Patient №', i, ' is', np.argmax(predictions[i]), '(', YT_data[i], ')')
'''