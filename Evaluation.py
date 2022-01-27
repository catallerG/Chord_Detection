import numpy as np
import glob


def open_dataset(path):
    dataset = []
    folder = glob.glob(path + "\\*.txt")
    for count, txtFile in enumerate(folder):
        array = np.loadtxt(txtFile, dtype=str)
        array[:, 0] = array[:, 0].astype(float)
        dataset.append(array)
    return dataset


def compute_OR(series, groundtruth_series): #groudtruth_series = dataset[i], numpy array
    error_time = 0
    time = 0
    for i, g_content in enumerate(groundtruth_series):
        if g_content[2] == "N":
            if time == np.shape(series)[0]-1:
                return 1 - error_time / float(groundtruth_series[np.shape(groundtruth_series)[0]-1, 1])
            while float(series[time, 1]) <= float(g_content[1]):
                time = time + 1
                if time == np.shape(series)[0]:
                    return 1 - error_time / float(groundtruth_series[np.shape(groundtruth_series)[0] - 1, 1])
            continue
        while float(series[time, 1]) <= float(g_content[1]):
            if float(series[time, 0]) < float(g_content[0]):   #ground change but series doesn't change condition
                if series[time-1, 2] != groundtruth_series[i-1, 2] and groundtruth_series[i-1, 2] != "N":
                    error_time = error_time + float(g_content[0]) - float(series[time-1, 0]) #unsupported operand type(s) for +: 'int' and 'numpy.str_'
                    if error_time == float(series[np.shape(series)[0]-1, 1]):
                        print("stop")
                if series[time, 2] != g_content[2]:
                    error_time = error_time + float(series[time, 1]) - float(g_content[0])
                    if error_time == float(series[np.shape(series)[0]-1, 1]):
                        print("stop")
            elif float(series[time, 0]) >= float(g_content[0]):
                if series[time, 2] != g_content[2]:
                    error_time = error_time + float(series[time, 1]) - float(series[time, 0])
                    if error_time == float(series[np.shape(series)[0]-1, 1]):
                        print("stop")
            time = time + 1
            if time == np.shape(series)[0]-1:
                return 1 - error_time / float(groundtruth_series[np.shape(groundtruth_series)[0]-1, 1])
    return 1 - error_time / float(groundtruth_series[np.shape(groundtruth_series)[0]-1, 1])



def compute_WAOR(duration_OR_series): #Nx2array, song length & OR
    numerator = 0
    for OR in duration_OR_series:
        numerator = numerator + OR[0] * OR[1]
    return numerator/np.sum(duration_OR_series[:, 0])



