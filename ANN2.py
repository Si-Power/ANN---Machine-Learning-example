import os
import tensorflow as tf
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import Include.ANN_v2 as A

# ИНС В НОТАЦИИ TENSORFLOW
#=============================================================================================
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

#====================================================================================================

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))


directory = r"C:\ANALYZER\FULL XY"
#directory = r"C:\ANALYZER\train_db"

numproc = 4  # Количество процессов
numf = 52284  # Количество файлов

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


# ******************НЕЙРОННАЯ СЕТЬ************************

print('Нейронная сеть обучается...')
start_time = time.time()

#Всего 194 человека, 52284 файлов, У каждого человека 269-270 файлов биометрических данных
n_samples = 52284
batch_size = 194
num_steps = 1000

# создаем заглушки, задаем размерность
x = tf.placeholder(tf.float32, [None, 400])    #Произвольное количество векторов любого размера
y_ = tf.placeholder(tf.float32, [None, 194])    #Может быть 194 различных вариантов ответа

# инициализируем переменные W и b
W = tf.Variable(tf.zeros([400, 194]))
b = tf.Variable(tf.zeros([194]))

y = tf.nn.softmax(tf.matmul(x, W) + b) # задаем модель

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1])) #строим фукцию ошибки

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss) #задаем оптимизатор

#correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(Y,1))
#accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))
#print ( "Точность: %s" % sess.run(accuracy, feed_dict = {x :mnist.test.images, Y: mnist.test.labels } ))

# запускаем сессию
display_step = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        # берем случайное подмножество из batch_size индексов данных
        indices = np.random.choice(n_samples, batch_size)

        # берем набор данных по выбранным индексам
        X_batch, y_batch = X_data[indices], Y_data[indices]

        print(X_batch)
        print(y_batch)

        X1 = np.vstack(X_batch)
        print(X1.shape)
        Y1 = np.vstack(y_batch)
        print(Y1.shape)

        # подаем в функцию sess.run список переменных, которые нужно подсчитать
        _, loss_val, y_pred_val = sess.run([optimizer, loss, y], feed_dict={x: X_batch, y: y_batch})

        # выводим результат
        if (i + 1) % display_step == 0:
            print('Epoch %d: %.8f, y_pred=%.4f' % (i + 1, loss_val, y_pred_val))

print('НЕЙРОННАЯ СЕТЬ ЗАКОНЧИЛА ОБУЧЕНИЕ! --- %s seconds ---' % (time.time() - start_time))


'''
#Построение графиков функций==================================================================
print('Строю график функции...')
#x = np.linspace(0, 10, 100)
#plt.plot(Ar, Ar2)
#plt.plot(x, BigAr2[x])
Xplot = np.arange(0,num_steps,1)
plt.plot(Xplot, loss_val)
plt.show()
print('ГРАФИК ФУНКЦИЙ ВЫПОЛНЕН!')
#=============================================================================================
'''

#Сохранение модели============================================================================
#saved = saver.save(sess, directory + '\\' + 'model.ckpt’)
#=============================================================================================

