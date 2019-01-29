import os
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
#import Include.ANN as A
#import Include.ANN2_KERAS as A2

# ИНС В НОТАЦИИ KERAS, ПРЕДСКАЗАНИЕ РЕЗУЛЬТАТОВ
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


def Likeness(vec1, vec2, alike, belongs):
    return 0

def ER():
    return 0
def EL():
    return 0
def EER():
    return 0

#=================================================================================================

directory = r"C:\ANALYZER\FULL XY"
numf = 953      #Количество файлов в папке

#Подгрузка тестовых данных =======================================================================

start_time = time.time()
print('Подгружаю тестовые данные...')
FArrX = getArray400(directory, 'TEST_X.txt', numf)
FArrY = getArray2(directory, 'TEST_Y.txt')
#print(FArrX)
#print(FArrY)
print('ПОДГРУЗКА ТЕСТОВЫХ ДАННЫХ ЗАВЕРШЕНА!  --- %s seconds ---' % (time.time() - start_time))


#Преобразование данных в NumPy ===================================================================

start_time = time.time()
print('Преобразовываю данные в массивы NumPy...')
X_data = []  # Аргументы функции для обучения
Y_data = []  # Значения функции для обучения
X_data2 = np.asarray(FArrX)
X_data = X_data2.reshape(numf,400)
Y_data = np.vstack(FArrY)
#print(X_data)
#print(Y_data)
print('ПРЕОБРАЗОВАНИЕ ДАННЫХ ЗАВЕРШЕНО!  --- %s seconds ---' % (time.time() - start_time))


# Форматирование данных под задачи Keras =========================================================

#print(X_data.max())    # 6.65564 - Максимальный X
#print(Y_data.max())    # 1157.0 - Максимальный Y

#X_data = X_data / X_data.max()
#Y_data = Y_data / Y_data.max()


#Восстановление сессии работы с нейронной сетью ==================================================
new_model = keras.models.load_model(directory + '\my_model.h5')

'''
model.compile(optimizer = tf.train.AdamOptimizer(0.0001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
'''

new_model.summary()

#Оценка точности ИНС =============================================================================
loss, acc = new_model.evaluate(X_data, Y_data)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


#Предсказание результатов ========================================================================
predictions = new_model.predict(X_data)

for i in range(len(predictions)):
    print('Person №', i, ' is', np.argmax(predictions[i]), '(', Y_data[i], ')')

#=================================================================================================

