import os
import tensorflow as tf
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# РЕАЛИЗАЦИЯ МНОГОПОТОЧНОГО ЧТЕНИЯ ФАЙЛОВ БЕЗ ГЕНЕРАТОРА СПИСКОВ, ВРУЧНУЮ
# ===========================================================================================

def getArray(FilePath, Filename):
    '''Returns Float array from file with <FilePath>/<File>.<Extension>'''
    _file = open(FilePath + '\\' + Filename)
    line = _file.read()
    vec = line.split(' ')
    vec_sl = vec[2:402]
    vec2 = []
    for i in range(len(vec_sl)):
        vec2.append(float(vec_sl[i]))
    _file.close()
    return vec2


def DbFill(A, B, Arr, Arr2, Path, Files):
    for i in range(A, B):
        Arr += getArray(Path, Files[i])  # Формируем список аргументов
        LArr = len(getArray(Path, Files[i]))  # Оцениваем длину файла с аргументами
        for j in range(LArr):  # Порождаем значения функции в соответствии с количеством аргументов
            Arr2 += [float(Files[i][:4])]


def ReadNames(A, B, Path, Files):
    for item in range(A, B):
        Files.append(os.listdir(Path)[item])


# =======================================================================================

# Реализация с параллелизмом
if __name__ == '__main__':
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    manager = multiprocessing.Manager()

    directory = r"C:\ANALYZER\train_db"
    filenames = []

    # =============================================================================================
    print('Формирую список файлов...')
    F1 = manager.list(); F2 = manager.list(); F3 = manager.list(); F4 = manager.list(); F5 = manager.list()
    F6 = manager.list(); F7 = manager.list(); F8 = manager.list(); F9 = manager.list(); F10 = manager.list()
    F11 = manager.list(); F12 = manager.list(); F13 = manager.list(); F14 = manager.list(); F15 = manager.list()
    F16 = manager.list(); F17 = manager.list(); F18 = manager.list(); F19 = manager.list(); F20 = manager.list()


    p1 = multiprocessing.Process(target=ReadNames, args=(0, 2614, directory, F1))
    p2 = multiprocessing.Process(target=ReadNames, args=(2614, 5228, directory, F2))
    p3 = multiprocessing.Process(target=ReadNames, args=(5228, 7843, directory, F3))
    p4 = multiprocessing.Process(target=ReadNames, args=(7843, 10457, directory, F4))
    p5 = multiprocessing.Process(target=ReadNames, args=(10457, 13071, directory, F5))
    p6 = multiprocessing.Process(target=ReadNames, args=(13071, 15685, directory, F6))
    p7 = multiprocessing.Process(target=ReadNames, args=(15685, 18299, directory, F7))
    p8 = multiprocessing.Process(target=ReadNames, args=(18299, 20914, directory, F8))
    p9 = multiprocessing.Process(target=ReadNames, args=(20914, 23528, directory, F9))
    p10 = multiprocessing.Process(target=ReadNames, args=(23528, 26142, directory, F10))

    p11 = multiprocessing.Process(target=ReadNames, args=(26142, 28756, directory, F11))
    p12 = multiprocessing.Process(target=ReadNames, args=(28756, 31370, directory, F12))
    p13 = multiprocessing.Process(target=ReadNames, args=(31370, 33985, directory, F13))
    p14 = multiprocessing.Process(target=ReadNames, args=(33985, 36599, directory, F14))
    p15 = multiprocessing.Process(target=ReadNames, args=(36599, 39213, directory, F15))
    p16 = multiprocessing.Process(target=ReadNames, args=(39213, 41827, directory, F16))
    p17 = multiprocessing.Process(target=ReadNames, args=(41827, 44441, directory, F17))
    p18 = multiprocessing.Process(target=ReadNames, args=(44441, 47056, directory, F18))
    p19 = multiprocessing.Process(target=ReadNames, args=(47056, 49670, directory, F19))
    p20 = multiprocessing.Process(target=ReadNames, args=(49670, 52284, directory, F20))


    p1.start(); p2.start(); p3.start(); p4.start(); p5.start()
    p6.start(); p7.start(); p8.start(); p9.start(); p10.start()
    p11.start(); p12.start(); p13.start(); p14.start(); p15.start()
    p16.start(); p17.start(); p18.start(); p19.start(); p20.start()


    p1.join(); p2.join(); p3.join(); p4.join(); p5.join()
    p6.join(); p7.join(); p8.join(); p9.join(); p10.join()
    p11.join(); p12.join(); p13.join(); p14.join(); p15.join()
    p16.join(); p17.join(); p18.join(); p19.join(); p20.join()

    p1.terminate(); p2.terminate(); p3.terminate(); p4.terminate(); p5.terminate()
    p6.terminate(); p7.terminate(); p8.terminate(); p9.terminate(); p10.terminate()
    p11.terminate(); p12.terminate(); p13.terminate(); p14.terminate(); p15.terminate()
    p16.terminate(); p17.terminate(); p18.terminate(); p19.terminate(); p20.terminate()

    filenames += list(F1) + list(F2) + list(F3) + list(F4) + list(F5) + list(F6) + list(F7) + list(F8) + list(F9) + list(F10) + list(F11) + list(F12) + list(F13) + list(F14) + list(F15) + list(F16) + list(F17) + list(F18) + list(F19) + list(F20)

    # Файлы в списке
    # print(filenames)

    print('СПИСОК ФАЙЛОВ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))

    # ========================================================================================================

    BigAr = []  # Полный массив данных аргументов из файлов (52284файлов*400чисел)
    BigAr2 = []  # Полный массив значений функций от аргументов из файлов (52284числа)
    Ar = np.array([])  # Для представления в TensorFlow в виде NumPy-массива аргументов
    Ar2 = np.array([])  # Для представления в TensorFlow в виде NumPy-массива значений функций от аргументов
    X_data = []  # Аргументы функции для обучения
    y_data = []  # Значения функции для обучения

    start_time = time.time()

    print('Создаю список аргументов и значений функций...')
    # Создание прокси-списков для доступа к ним в общей памяти
    LX1 = manager.list(); LX2 = manager.list(); LX3 = manager.list(); LX4 = manager.list(); LX5 = manager.list()
    LX6 = manager.list(); LX7 = manager.list(); LX8 = manager.list(); LX9 = manager.list(); LX10 = manager.list()
    LX11 = manager.list(); LX12 = manager.list(); LX13 = manager.list(); LX14 = manager.list(); LX15 = manager.list()
    LX16 = manager.list(); LX17 = manager.list(); LX18 = manager.list(); LX19 = manager.list(); LX20 = manager.list()

    LY1 = manager.list(); LY2 = manager.list(); LY3 = manager.list(); LY4 = manager.list(); LY5 = manager.list()
    LY6 = manager.list(); LY7 = manager.list(); LY8 = manager.list(); LY9 = manager.list(); LY10 = manager.list()
    LY11 = manager.list(); LY12 = manager.list(); LY13 = manager.list(); LY14 = manager.list(); LY15 = manager.list()
    LY16 = manager.list(); LY17 = manager.list(); LY18 = manager.list(); LY19 = manager.list(); LY20 = manager.list()



    p1 = multiprocessing.Process(target=DbFill, args=(0, 2614, LX1, LY1, directory, filenames))
    p2 = multiprocessing.Process(target=DbFill, args=(2614, 5228, LX2, LY2, directory, filenames))
    p3 = multiprocessing.Process(target=DbFill, args=(5228, 7843, LX3, LY3, directory, filenames))
    p4 = multiprocessing.Process(target=DbFill, args=(7843, 10457, LX4, LY4, directory, filenames))
    p5 = multiprocessing.Process(target=DbFill, args=(10457, 13071, LX5, LY5, directory, filenames))
    p6 = multiprocessing.Process(target=DbFill, args=(13071, 15685, LX6, LY6, directory, filenames))
    p7 = multiprocessing.Process(target=DbFill, args=(15685, 18299, LX7, LY7, directory, filenames))
    p8 = multiprocessing.Process(target=DbFill, args=(18299, 20914, LX8, LY8, directory, filenames))
    p9 = multiprocessing.Process(target=DbFill, args=(20914, 23528, LX9, LY9, directory, filenames))
    p10 = multiprocessing.Process(target=DbFill, args=(23528, 26142, LX10, LY10, directory, filenames))

    p11 = multiprocessing.Process(target=DbFill, args=(26142, 28756, LX11, LY11, directory, filenames))
    p12 = multiprocessing.Process(target=DbFill, args=(28756, 31370, LX12, LY12, directory, filenames))
    p13 = multiprocessing.Process(target=DbFill, args=(31370, 33985, LX13, LY13, directory, filenames))
    p14 = multiprocessing.Process(target=DbFill, args=(33985, 36599, LX14, LY14, directory, filenames))
    p15 = multiprocessing.Process(target=DbFill, args=(36599, 39213, LX15, LY15, directory, filenames))
    p16 = multiprocessing.Process(target=DbFill, args=(39213, 41827, LX16, LY16, directory, filenames))
    p17 = multiprocessing.Process(target=DbFill, args=(41827, 44441, LX17, LY17, directory, filenames))
    p18 = multiprocessing.Process(target=DbFill, args=(44441, 47056, LX18, LY18, directory, filenames))
    p19 = multiprocessing.Process(target=DbFill, args=(47056, 49670, LX19, LY19, directory, filenames))
    p20 = multiprocessing.Process(target=DbFill, args=(49670, 52284, LX20, LY20, directory, filenames))


    p1.start(); p2.start(); p3.start(); p4.start(); p5.start()
    p6.start(); p7.start(); p8.start(); p9.start(); p10.start()
    p11.start(); p12.start(); p13.start(); p14.start(); p15.start()
    p16.start(); p17.start(); p18.start(); p19.start(); p20.start()

    p1.join(); p2.join(); p3.join(); p4.join(); p5.join()
    p6.join(); p7.join(); p8.join(); p9.join(); p10.join()
    p11.join(); p12.join(); p13.join(); p14.join(); p15.join()
    p16.join(); p17.join(); p18.join(); p19.join(); p20.join()

    p1.terminate(); p2.terminate(); p3.terminate(); p4.terminate(); p5.terminate()
    p6.terminate(); p7.terminate(); p8.terminate(); p9.terminate(); p10.terminate()
    p11.terminate(); p12.terminate(); p13.terminate(); p14.terminate(); p15.terminate()
    p16.terminate(); p17.terminate(); p18.terminate(); p19.terminate(); p20.terminate()


    BigAr += list(LX1) + list(LX2) + list(LX3)+ list(LX4) + list(LX5) + list(LX6) + list(LX7)+ list(LX8) + list(LX9) + list(LX10) + list(LX11) + list(LX12) + list(LX13)+ list(LX14) + list(LX15) + list(LX16) + list(LX17)+ list(LX18) + list(LX19) + list(LX20)
    BigAr2 += list(LY1) + list(LY2) + list(LY3)+ list(LY4) + list(LY5) + list(LY6) + list(LY7)+ list(LY8) + list(LY9) + list(LY10) + list(LY11) + list(LY12) + list(LY13)+ list(LY14) + list(LY15) + list(LY16) + list(LY17)+ list(LY18) + list(LY19) + list(LY20)
    # print(BigAr)
    # print(BigAr2)
    print('СПИСКИ АГРУМЕНТОВ И ЗНАЧЕНИЙ ФУНКЦИИ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))

    # Преобразование в NumPy-массивы======================================
    start_time = time.time()
    print('Выполняю преобразование списков в массивы NumPy')
    Ar = np.vstack(BigAr)
    Ar2 = np.vstack(BigAr2)
    print('ПРЕОБРАЗОВАНИЕ ВЫПОЛНЕНО! --- %s seconds ---' % (time.time() - start_time))
    # ====================================================================

    '''
    print('Строю график функции...')
    #x = np.linspace(0, 10, 100)
    plt.plot(Ar, Ar2)
    #plt.plot(x, BigAr2[x])
    plt.show()
    print('ГРАФИК ФУНКЦИЙ ВЫПОЛНЕН!')

    '''

    # ******************НЕЙРОННАЯ СЕТЬ************************

    '''

    n_samples = 400
    batch_size = 100
    num_steps = 10000

    # набрасываем n_samples случайных точек равномерно на интервале [0; 1]
    X_data = np.random.uniform(0, 1, (n_samples, 1))
    # подсчитываем "правильные ответы"по формуле y = 2x+1+e, где е - случайно распеределенный шум с дисперсией 0.2
    y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1))

    # создаем заглушки, задаем размерность
    X = tf.placeholder(tf.float32, shape=(batch_size, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, 1))

    # инициализируем переменные k и b
    with tf.variable_scope('linear-regression'):
        k = tf.Variable(tf.random_normal((1, 1), stddev=0.01), name='slope')
        b = tf.Variable(tf.zeros(1, ), name='bias')

    # задаем модель
    y_pred = tf.matmul(X, k) + b

    # строим фукцию ошибки
    loss = tf.reduce_sum(np.power(y - y_pred, 2))

    # задаем оптимизатор
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(loss)

    # запускаем сессию
    display_step = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            # берем случайное подмножество из batch_size индексов данных
            indices = np.random.choice(n_samples, batch_size)

            # берем набор данных по выбранным индексам
            X_batch, y_batch = X_data[indices], y_data[indices]

            # подаем в функцию sess.run список переменных, которые нужно подсчитать
            _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b], feed_dict={X: X_batch, y: y_batch})

            # выводим результат
            if (i + 1) % display_step == 0:
                print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val, b_val))
    '''

