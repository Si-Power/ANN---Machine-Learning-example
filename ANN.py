import os
import tensorflow as tf
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# РЕАЛИЗАЦИЯ МНОГОПОТОЧНОГО ЧТЕНИЯ ФАЙЛОВ БЕЗ ГЕНЕРАТОРА СПИСКОВ, ВРУЧНУЮ
#===========================================================================================

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


def getArray2(FilePath, Filename):
    '''Returns Float array from file with <FilePath>/<File>.<Extension>'''
    _file = open(FilePath + '\\' + Filename)
    line = _file.read()
    vec = line.split(', ')
    vec2 = []
    for i in range(len(vec)):
        vec2.append(float(vec[i]))
    _file.close()
    return vec2


def DbFill(A, B, Arr, Arr2, Path, Files):
    for i in range(A,B):
        Arr += getArray(Path, Files[i])         #Формируем список аргументов
        LArr = len(getArray(Path, Files[i]))    #Оцениваем длину файла с аргументами
        for j in range(LArr):                   #Порождаем значения функции в соответствии с количеством аргументов
            Arr2 += [float(Files[i][:4])]


def ReadNames(A, B, Path, Files):
    for item in range(A, B):
        Files.append(os.listdir(Path)[item])


#=======================================================================================

#Реализация с параллелизмом
if __name__ == '__main__':

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    manager = multiprocessing.Manager()

    directory = r"C:\ANALYZER\train_db"
    filenames = []


    #=============================================================================================
    print('Формирую список файлов...')
    F1 = manager.list(); F2 = manager.list(); F3 = manager.list(); F4 = manager.list()
    #F5 = manager.list(); F6 = manager.list(); F7 = manager.list(); F8 = manager.list(); F9 = manager.list(); F10 = manager.list()


    p1 = multiprocessing.Process(target=ReadNames, args=(0, 250, directory, F1))
    p2 = multiprocessing.Process(target=ReadNames, args=(250, 500, directory, F2))
    p3 = multiprocessing.Process(target=ReadNames, args=(500, 750, directory, F3))
    p4 = multiprocessing.Process(target=ReadNames, args=(750, 1000, directory, F4))

    '''    
    p1 = multiprocessing.Process(target=ReadNames, args=(0, 5228, directory, F1))
    p2 = multiprocessing.Process(target=ReadNames, args=(5228, 10457, directory, F2))
    p3 = multiprocessing.Process(target=ReadNames, args=(10457, 15685, directory, F3))
    p4 = multiprocessing.Process(target=ReadNames, args=(15685, 20914, directory, F4))
    p5 = multiprocessing.Process(target=ReadNames, args=(20914, 26142, directory, F5))
    p6 = multiprocessing.Process(target=ReadNames, args=(26142, 31370, directory, F6))
    p7 = multiprocessing.Process(target=ReadNames, args=(31370, 36599, directory, F7))
    p8 = multiprocessing.Process(target=ReadNames, args=(36599, 41827, directory, F8))
    p9 = multiprocessing.Process(target=ReadNames, args=(41827, 47056, directory, F9))
    p10 = multiprocessing.Process(target=ReadNames, args=(47056, 52284, directory, F10))
    '''

    p1.start(); p2.start(); p3.start(); p4.start()
    #p5.start(); p6.start(); p7.start(); p8.start(); p9.start(); p10.start()

    p1.join(); p2.join(); p3.join(); p4.join()
    #p5.join(); p6.join(); p7.join(); p8.join(); p9.join(); p10.join()

    p1.terminate(); p2.terminate(); p3.terminate(); p4.terminate()
    #p5.terminate(); p6.terminate(); p7.terminate(); p8.terminate(); p9.terminate(); p10.terminate()


    filenames += list(F1) + list(F2) + list(F3) + list(F4)
    #filenames += list(F1) + list(F2) + list(F3) + list(F4) + list(F5) + list(F6) + list(F7) + list(F8) + list(F9) + list(F10)

    # Файлы в списке
    #print(filenames)

    print('СПИСОК ФАЙЛОВ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))


#========================================================================================================


    BigAr = []              #Полный массив данных аргументов из файлов (52284файлов*400чисел)
    BigAr2 = []             #Полный массив значений функций от аргументов из файлов (52284числа)


    start_time = time.time()

    print('Создаю список аргументов и значений функций...')
    #Создание прокси-списков для доступа к ним в общей памяти
    LX1 = manager.list(); LX2 = manager.list(); LX3 = manager.list(); LX4 = manager.list()
    #LX5 = manager.list(); LX6 = manager.list(); LX7 = manager.list(); LX8 = manager.list(); LX9 = manager.list(); LX10 = manager.list()

    LY1 = manager.list(); LY2 = manager.list(); LY3 = manager.list(); LY4 = manager.list()
    #LY5 = manager.list(); LY6 = manager.list(); LY7 = manager.list(); LY8 = manager.list(); LY9 = manager.list(); LY10 = manager.list()

    p1 = multiprocessing.Process(target=DbFill, args=(0, 250, LX1, LY1, directory, filenames))
    p2 = multiprocessing.Process(target=DbFill, args=(250, 500, LX2, LY2, directory, filenames))
    p3 = multiprocessing.Process(target=DbFill, args=(500, 750, LX3, LY3, directory, filenames))
    p4 = multiprocessing.Process(target=DbFill, args=(750, 1000, LX4, LY4, directory, filenames))

    '''
    p1 = multiprocessing.Process(target=DbFill, args=(0, 5228, LX1, LY1, directory, filenames))
    p2 = multiprocessing.Process(target=DbFill, args=(5228, 10457, LX2, LY2, directory, filenames))
    p3 = multiprocessing.Process(target=DbFill, args=(10457, 15685, LX3, LY3, directory, filenames))
    p4 = multiprocessing.Process(target=DbFill, args=(15685, 20914, LX4, LY4, directory, filenames))
    p5 = multiprocessing.Process(target=DbFill, args=(20914, 26142, LX5, LY5, directory, filenames))
    p6 = multiprocessing.Process(target=DbFill, args=(26142, 31370, LX6, LY6, directory, filenames))
    p7 = multiprocessing.Process(target=DbFill, args=(31370, 36599, LX7, LY7, directory, filenames))
    p8 = multiprocessing.Process(target=DbFill, args=(36599, 41827, LX8, LY8, directory, filenames))
    p9 = multiprocessing.Process(target=DbFill, args=(41827, 47056, LX9, LY9, directory, filenames))
    p10 = multiprocessing.Process(target=DbFill, args=(47056, 52284, LX10, LY10, directory, filenames))
    '''

    p1.start(); p2.start(); p3.start(); p4.start()
    #p5.start(); p6.start(); p7.start(); p8.start(); p9.start(); p10.start()

    p1.join(); p2.join(); p3.join(); p4.join()
    #p5.join(); p6.join(); p7.join(); p8.join(); p9.join(); p10.join()

    p1.terminate(); p2.terminate(); p3.terminate(); p4.terminate()
    #p5.terminate(); p6.terminate(); p7.terminate(); p8.terminate(); p9.terminate(); p10.terminate()

    BigAr += list(LX1) + list(LX2) + list(LX3)+ list(LX4)
    BigAr2 += list(LY1) + list(LY2) + list(LY3)+ list(LY4)

    #BigAr += list(LX1) + list(LX2) + list(LX3)+ list(LX4) + list(LX5) + list(LX6) + list(LX7)+ list(LX8) + list(LX9) + list(LX10)
    #BigAr2 += list(LY1) + list(LY2) + list(LY3)+ list(LY4) + list(LY5) + list(LY6) + list(LY7)+ list(LY8) + list(LY9) + list(LY10)
    #print(BigAr)
    #print(BigAr2)
    print('СПИСКИ АГРУМЕНТОВ И ЗНАЧЕНИЙ ФУНКЦИИ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))


    #Запись в отдельные файлы прочитанных результатов============================
    print('Записываю в файл...')
    X_File = open(directory + '\\' + 'FULLX.txt', 'w')
    Y_File = open(directory + '\\' + 'FULLY.txt', 'w')
    StrX = ''.join(str(BigAr)[1:-1])
    StrY = ''.join(str(BigAr2)[1:-1])
    X_File.write(StrX)
    Y_File.write(StrY)
    X_File.close()
    Y_File.close()
    print('ЗАПИСЬ В ФАЙЛ ЗАВЕРШЕНА!!!')
    #=============================================================================