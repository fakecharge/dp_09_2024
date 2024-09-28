import os
import subprocess
import json
import re
import numpy as np
from scipy.signal import find_peaks
import array as arr
import csv

# Получение номеров ключевых кадров из метаданных *.mp4
def extract_frame_metadata(video_file):

    MinPeakDistance = 6  # [frame] Удалим ключевые кадры, идущие чаще чем MinPeakDistance
    max_time_no_key_frame = 3  # [сек] Если в течении max_time_no_key_frame нет ключевых кадров, добавим их

    # Считаем все номера ключевых кадров из файла
    command = f'ffmpeg -i "{video_file}" -vf "select=\'eq(pict_type,I)\',showinfo" -an -f null -'
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        status = result.returncode
        cmdout = result.stdout + result.stderr  # Объединяем stdout и stderr
    except Exception as e:
        print("An error occurred:", e)
        return []
    fps_string = re.findall(r'(\d+)\s+fps', cmdout)
    fps = float(fps_string[0])
    duration_time_strings = re.findall(r'duration_time:\s*([\d\.]+)', cmdout)
    duration_time = float(duration_time_strings[0]);

    pts_time_strings = re.findall(r'pts_time:\s*([\d\.]+)', cmdout)  # получим все длительности между ключевыми кадрами
    pts_time_array = [float(num) for num in pts_time_strings]
    pts_time_array = (np.array(pts_time_array)/duration_time).astype('int')
    diff_pts_time_array = np.diff(pts_time_array)
    #tmp2 = (num == diff_pts_time_array[0] for num in diff_pts_time_array)
    #if len(set(diff_pts_time_array))>1:
    if len(diff_pts_time_array)>1:
        if max(np.diff(diff_pts_time_array)) > 2:
            return np.round(pts_time_array)

    # Ключевые кадры не попадают на смену, сцен, а расставленны кодеком при сжатии равномерно. Найдем самостоятельно смену сцен
    command = f'ffmpeg -i "{video_file}" -vf "select=not(mod(n\\,1)),showinfo" -f null -'

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        status = result.returncode
        cmdout = result.stdout + result.stderr  # Объединяем stdout и stderr

        # Используем регулярные выражения для извлечения стандартных отклонений
        stdev_strings = re.findall(r'stdev:\[([0-9\. \[\]]+)\]', cmdout)
        data = [list(map(float, line.split())) for line in stdev_strings]
        Std = np.array(data)

        # Вычисляем сумму с учетом весов
        weighted_sum = np.sum(Std * np.array([0.299, 0.587, 0.114]), axis=1)

        # Вычисляем разности и берем абсолютные значения
        stddiff = np.abs(np.diff(weighted_sum))

        # Вычисляем итоговое значение
        tr2 = 3 * np.mean(stddiff)

        # Находим индексы пиков
        good_ind, _ = find_peaks(stddiff > tr2, distance=MinPeakDistance)
        if len(good_ind)==0:
            return [0]
        good_ind += 1

        # Создаем массив good_frame
        good_frame = np.zeros_like(stddiff)

        # Устанавливаем значения на позициях good_ind в 1
        good_frame[good_ind] = 1

        maxF = round(max_time_no_key_frame * fps)

        for i in range(len(good_ind) - 1):
            f1 = good_ind[i]
            f2 = good_ind[i + 1]
            if f2 - f1 > maxF:
                num_add_frame = (f2 - f1) // maxF
                # Добавляем дополнительные кадры
                for j in range(1, num_add_frame + 1):
                    additional_frame_index = f1 + round(j * (f2 - f1) / (num_add_frame + 1))
                    good_frame[additional_frame_index] = 1

        # Находим индексы хороших кадров
        good_ind = np.where(good_frame == 1)[0]
        if good_ind[0]!=0:
            good_ind = np.insert(good_ind,0,0)
        if len(good_ind)==0:
            print('ads')
        return good_ind
    except Exception as e:
        print("An error occurred:", e)
        return[]

if __name__ == "__main__":

    #video_file = 'd:\\yappi\\2.mp4'  # Укажите путь к вашему видеофайлу
    #video_file = 'd:\\yappi\\train_data_yappy\\train_dataset\\04e8f057-dffd-4cf0-a047-b44b9aa5ae07.mp4'
    #index = extract_frame_metadata(video_file)
    #print("Ключевые кадры:", index)

    filename_csv = "output.csv"
    dir = 'D:\\pycharm\dp2024_Moscow\\train_data_yappy\\train_dataset\\'
    filenames_list = os.listdir(dir)
    with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=' ')

        # Записываем данные в цикле
        for idx, filename in enumerate(filenames_list):
            full_filename = os.path.join(dir, filename)
            index = extract_frame_metadata(full_filename)

            #writer.writerow([filename, ", ".join(map(str, index))])
            writer.writerow([filename, ' '.join(str(xi) for xi in index)])
            print([filename, ' '.join(str(xi) for xi in index)])



