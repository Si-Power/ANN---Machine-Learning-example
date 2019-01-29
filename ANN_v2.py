import os
import tensorflow as tf
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# РЕАЛИЗАЦИЯ МНОГОПОТОЧНОГО ЧТЕНИЯ ФАЙЛОВ С ПОМОЩЬЮ ГЕНЕРАТОРА СПИСКОВ, АВТОМАТИЧЕСКИ
#===========================================================================================

def getArray(FilePath, Filename):   #Чтение данных из отдельных файлов
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
    for i in range(A,B):
        Arr += getArray(Path, Files[i])         #Формируем список аргументов
        Arr2 += [float(Files[i][:4])]           #Порождаем значения функции в соответствии с количеством аргументов


def ReadNames(A, B, Path, Files):               #Преобразование имен файлов в значения функций
    for item in range(A, B):
        Files.append(os.listdir(Path)[item])


#=======================================================================================

#Реализация с параллелизмом
if __name__ == '__main__':

    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    manager = multiprocessing.Manager()

    #directory = r"C:\ANALYZER\train_db"   #Для тренировочных данных
    directory = r"C:\ANALYZER\test_db"    #Для тестовых данных
    filenames = []

    #=============================================================================================
    print('Формирую список файлов...')

    numproc = 4     #Количество процессов
    numf = 150       #Количество файлов


    F = []      #Фрагменты списка имен файлов для быстрого доступа к ним
    p1 = []     #Процессы
    for i in range(numproc):
        F.append(manager.list())
        Icurr = i*round(numf / numproc)
        Inext = (i + 1) * round(numf / numproc)
        if (i == (numproc-1)):          #Если цикл - последний
            if (Inext == numf):     #Если итеративное количество файлов равно конечному
                #print('Количество равно конечному! Смотрю файлы с ', Icurr, 'по ', Inext)
                p1.append(multiprocessing.Process(target=ReadNames, args=(Icurr, Inext, directory, F[i])))
            else:       #Иначе приравниваем конечное число количеству файлов, чтобы не потерять последние файлы
                #print('Количество не равно конечному!!! Смотрю файлы с ', Icurr, 'по ', numf)
                p1.append(multiprocessing.Process(target=ReadNames, args=(Icurr, numf, directory, F[i])))
        else:       #Если цикл не последний
            #print('Смотрю файлы с ', Icurr, 'по ', Inext)
            p1.append(multiprocessing.Process(target=ReadNames, args=(Icurr, Inext, directory, F[i])))
        p1[i].start()
        p1[i].join()
        p1[i].terminate()

    for i in range(numproc):        #Последовательная сборка результатов в единый список имен файлов
        filenames += list(F[i])

    #print(filenames)        #Файлы в списке

    print('СПИСОК ФАЙЛОВ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))

    #========================================================================================================


    start_time = time.time()

    print('Создаю список аргументов и значений функций...')

    BigAr = []      #Полный массив данных аргументов и функций из файлов (52284 файлов * 400 чисел)
    BigAr2 = []     #Полный массив данных аргументов и функций из файлов (52284 значений)

    # Создание прокси-списков для доступа к ним в общей памяти===
    LX = []
    LY = []
    #============================================================

    p2 = []     # Процессы

    for i in range(numproc):
        LX.append(manager.list())
        LY.append(manager.list())
        Icurr = i*round(numf/numproc)
        Inext = (i + 1) * round(numf / numproc)
        if (i == (numproc-1)):          #Если цикл - последний
            if (Inext == numf):     #Если итеративное количество файлов равно конечному
                #print('Количество равно конечному! Смотрю файлы с ', Icurr, 'по ', Inext)
                p2.append(multiprocessing.Process(target=DbFill, args=(Icurr, Inext, LX[i], LY[i], directory, filenames)))
            else:       #Иначе приравниваем конечное число количеству файлов, чтобы не потерять последние файлы
                #print('Количество не равно конечному!!! Смотрю файлы с ', Icurr, 'по ', numf)
                p2.append(multiprocessing.Process(target=DbFill, args=(Icurr, numf, LX[i], LY[i], directory, filenames)))
        else:       #Если цикл не последний
            #print('Смотрю файлы с ', Icurr, 'по ', Inext)
            p2.append(multiprocessing.Process(target=DbFill, args=(Icurr, Inext, LX[i], LY[i], directory, filenames)))
        p2[i].start()
        p2[i].join()
        p2[i].terminate()

    for i in range(numproc):  # Последовательная сборка результатов
        BigAr += list(LX[i])
        BigAr2 += list(LY[i])

    #print(BigAr); print(BigAr2)
    print('СПИСКИ АГРУМЕНТОВ И ЗНАЧЕНИЙ ФУНКЦИИ ВЫПОЛНЕН! --- %s seconds ---' % (time.time() - start_time))


    #Запись в отдельные файлы прочитанных результатов============================
    print('Записываю в файл...')
    # Тренировочные данные ======================================================
    '''
    X_File = open(directory + '\\' + 'FULLX.txt', 'w')
    Y_File = open(directory + '\\' + 'FULLY.txt', 'w')
    StrX = ''.join(str(BigAr)[1:-1])
    StrY = ''.join(str(BigAr2)[1:-1])
    X_File.write(StrX)
    Y_File.write(StrY)
    X_File.close()
    Y_File.close()
    '''
    # Тестовые данные ============================================================

    X_File = open(directory + '\\' + 'TEST_X.txt', 'w')
    Y_File = open(directory + '\\' + 'TEST_Y.txt', 'w')
    StrX = ''.join(str(BigAr)[1:-1])
    StrY = ''.join(str(BigAr2)[1:-1])
    X_File.write(StrX)
    Y_File.write(StrY)
    X_File.close()
    Y_File.close()

    print('ЗАПИСЬ В ФАЙЛ ЗАВЕРШЕНА!!!')
    #=============================================================================

