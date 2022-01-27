# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import glob

import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise
from matplotlib.colors import LogNorm
import scipy.io.wavfile
import scipy.fftpack
import scipy.signal
import librosa
import math
import Evaluation
from sklearn import preprocessing


def create_HMM_state():
    return np.array(["C", "c_minor3", "Db", "db_minor3", "D", "d_minor3", "Eb", "eb_minor3", "E", "e_minor3",
                     "F", "f_minor3", "Gb", "gb_minor3", "G", "g_minor3", "Ab", "ab_minor3", "A", "a_minor3",
                     "Bb", "bb_minor3", "B", "b_minor3", "N"])


def tunning_matrix(WINDOW_LENGTH, fs):
    pitches = np.array(([261.23, 523.25, 1046.5, 2093.0],
                     [277.18, 554.37, 1108.7, 2217.5],
                     [293.66, 587.33, 1174.7, 2349.3],
                     [311.13, 622.25, 1244.5, 2489.0],
                     [329.63, 659.26, 1318.5, 2637.0],
                     [349.23, 698.46, 1396.9, 2793.8],
                     [369.99, 739.99, 1480.0, 2960.0],
                     [392.00, 783.99, 1568.0, 3136.0],
                     [415.30, 830.61, 1661.2, 3322.4],
                     [440.00, 880.00, 1760.0, 3520.0],
                     [466.16, 932.33, 1864.7, 3729.3],
                     [493.88, 987.77, 1975.5, 3951.1]))
    matrix = np.zeros((12, WINDOW_LENGTH))
    for i, pitch_name in enumerate(pitches):
        for j, pitch in enumerate(pitch_name):
            x = pitch * WINDOW_LENGTH/fs
            L = int(x)
            R = L+1
            y = (x-R)/(L-R)
            matrix[i][L] = y
            matrix[i][R] = 1-y
    return matrix


def pitch_class(WINDOW_LENGTH, fs):
    pitchClass = np.zeros((12, WINDOW_LENGTH))
    for frame in np.arange(12):
        for ex in np.arange(7):
            m = round((32.2 * math.pow(2, ex) + 1.945 * math.pow(2, ex)) * WINDOW_LENGTH / fs)
            n = round((33.2 * math.pow(2, ex) + 1.945 * math.pow(2, ex)) * WINDOW_LENGTH / fs)
            pitchClass[frame][m:n + 1] = 1
    return pitchClass


class chord_class:
    def __init__(self):
        self.C = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.c_minor3 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        #self.c_dim3 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        #self.C_aug3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        #self.C_major7 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        #self.C_7 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
        #self.c_minor7 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])

        self.Db = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        self.db_minor3 = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
       # self.db_dim3 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        #self.Db_aug3 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        #self.Db_major7 = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        #self.Db_7 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        #self.db_minor7 = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])

        self.D = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        self.d_minor3 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        #self.d_dim3 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        #self.D_aug3 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        #self.D_major7 = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        #self.D_7 = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        #self.d_minor7 = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])

        self.Eb = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
        self.eb_minor3 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
        #self.eb_dim3 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        #self.Eb_aug3 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        #self.Eb_major7 = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        #self.Eb_7 = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
       # self.eb_minor7 = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

        self.E = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
        self.e_minor3 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        #self.e_dim3 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        #self.E_aug3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        #self.E_major7 = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
        #self.E_7 = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1])
        #self.e_minor7 = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1])

        self.F = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        self.f_minor3 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        #self.f_dim3 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1])
        #self.F_aug3 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        #self.F_major7 = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])
        #self.F_7 = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
       # self.f_minor7 = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])

        self.Gb = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.gb_minor3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        #self.gb_dim3 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        #self.Gb_aug3 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        #self.Gb_major7 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        #self.Gb_7 = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        #self.gb_minor7 = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0])

        self.G = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        self.g_minor3 = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        #self.g_dim3 = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        #self.G_aug3 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        #self.G_major7 = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        #self.G_7 = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])
        #self.g_minor7 = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])

        self.Ab = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        self.ab_minor3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1])
        #self.ab_dim3 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        #self.Ab_aug3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        #self.Ab_major7 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        #self.Ab_7 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
        #self.ab_minor7 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

        self.A = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
        self.a_minor3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
        #self.a_dim3 = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
       # self.A_aug3 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        #self.A_major7 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
        #self.A_7 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
       # self.a_minor7 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])

        self.Bb = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        self.bb_minor3 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        #self.bb_dim3 = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
       # self.Bb_aug3 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        #self.Bb_major7 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])
        #self.Bb_7 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0])
        #self.bb_minor7 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0])

        self.B = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        self.b_minor3 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        #self.b_dim3 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        #self.B_aug3 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        #self.B_major7 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1])
        #self.B_7 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
        #self.b_minor7 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1])


def freq2class(freq):
    if freq != 0:
        key = round(12 * math.log(freq / 440, 2)) + 69
    else:
        key = 0
    return key % 12


def toPitchVector(spectrogram, rate, WINDOW_LENGTH):
    pitchVector = np.zeros((12, spectrogram.shape[1]))
    for frame in np.arange(spectrogram.shape[1]):
        for freq in np.arange(start=int(WINDOW_LENGTH * 130.81 / rate), stop=int(WINDOW_LENGTH * 2093 / rate)):
            if spectrogram[freq][frame] > 0:
                pitchVector[freq2class(rate * (freq / WINDOW_LENGTH))][frame] = \
                pitchVector[freq2class(rate * (freq / WINDOW_LENGTH))][frame] + spectrogram[freq][frame]
        if np.max(pitchVector[:, frame]) == 0:
            continue
        pitchVector[:, frame] = normalization(pitchVector[:, frame])
    return pitchVector


def ifDownmix(wav):
    if wav.shape[-1] != 2:
        return wav
    else:
        mono = wav[:, 0] / 2 + wav[:, 1] / 2
        return mono


def block_audio(x, blockSize, hopSize, fs):
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)] * np.hamming(blockSize)
    return xb, t


def normalization(array):
    mx = np.max(array)
    mn = np.min(array)
    return (array - mn) / (mx - mn)


def normalize(array):
    return array / np.max(array)


def normalize_zscore(featureData):
    normFeatureMatrix = np.zeros((featureData.shape[0], featureData.shape[1]))

    for n, feature in enumerate(featureData):
        mean = np.mean(feature)
        stdev = np.std(feature)
        for i, x in enumerate(feature):
            x = (x - mean) / stdev
            normFeatureMatrix[n, i] = x

    return normFeatureMatrix


def select_20_largest_value(spec):
    X = spec.T
    for i, frame in enumerate(X):
        for j in frame.argsort()[0:frame.shape[0]-20]:
            frame[j] = 0
    return X.T


def toSpectrogram(wave, BlockSize, HopSize, rate):
    block, timeInSec = block_audio(wave, BlockSize, HopSize, rate)
    block = scipy.fftpack.fft(block)
    spec = np.absolute(np.vstack(block).T)
    spec = normalization(spec)
    return spec, timeInSec


def get_transition_probability_from_annotation(dataset, state, fs):
    #compute Pt by frame
    frame = 1/fs


    transition_probability = np.zeros((25, 25))
    start_probability = np.zeros(25)
    for ground_truth_series in dataset:
        vector = []
        for i, row in enumerate(ground_truth_series):
            if not row[2] in state:
                row[2] = row[2].split(":")[0]
            if row[2] in state:
                position = int(np.where(state == row[2])[0][0])
                vector.append(position)
                transition_probability[position][position] = transition_probability[position][position] + 1# + int((float(row[1])-float(row[0]))/frame)
        start_probability[vector[0]] = start_probability[vector[0]] + 1
        for i1, value in enumerate(vector):
            if i1 > 0:
                transition_probability[vector[i1-1]][value] = transition_probability[vector[i1-1]][value] + 1
    template = transition_probability.copy()

    #shifting
    for i in np.arange(25):
        transition_probability = transition_probability + np.roll(np.roll(template, i, axis=0), i, axis=1)

    transition_probability[transition_probability == 0] = 1
    for j, row2 in enumerate(transition_probability):
        if np.sum(row2) == 0:
            continue
        transition_probability[j] = row2 / np.sum(row2)
    start_probability = start_probability / np.sum(start_probability)
    return transition_probability, start_probability


def matlab_max(vector):
    max = np.max(vector)
    position = np.where(vector == max)
    return max, position[0][0]


def viterbi(Ps, Pt, Pe):
    I = np.zeros(np.shape(Pe))
    P_res = np.zeros(np.shape(Pe))
    P_res[:, 0] = Pe[:, 0]*Ps
    for n in np.arange(start=1, stop=np.shape(Pe)[1]):
        if n == 15:
            print("syto")
        for s in np.arange(start=0, stop=np.shape(Pe)[0]):
            Pmax, I[s][n] = matlab_max(P_res[:, n-1]*Pt[:, s])
            P_res[s][n] = Pe[s][n] * Pmax
    p = np.zeros((1, np.shape(Pe)[1]))
    prob, p[:, np.shape(p)[1]-1] = matlab_max(P_res[:, np.shape(P_res)[1]-1])
    for n in np.arange(start=np.shape(Pe)[1]-2, stop=0, step=-1):
        p[0][n] = I[int(p[0][n+1])][n+1]
    #for n = size(P_E, 2) -1:-1: 1
    #p(n) = I(p(n + 1), n + 1);
    return p


def chord_template():
    T = np.zeros((24, 12))
    T[0, 0] = 1 / 3
    T[0, 4] = 1 / 3
    T[0, 7] = 1 / 3
    T[1, 0] = 1 / 3
    T[1, 3] = 1 / 3
    T[1, 7] = 1 / 3
    for i in np.arange(start=1, stop=12):
        T[2*i] = np.roll(T[2*(i-1)], 1)
        T[2*i+1] = np.roll(T[2*(i-1)+1], 1)
    T = np.vstack((T, np.array([1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])))
    plt.figure()
    plt.imshow(T, cmap=plt.get_cmap("inferno"))
    plt.show()
    return T


def chord_detection(pitch_chroma):
    cd = chord_class()
    chord_progression = np.zeros((0, 12))
    chord_progression_name = []
    for frame, vector in enumerate(pitch_chroma.T):
        if np.max(vector) == 0:
            chord_progression = np.vstack((chord_progression, np.zeros((1, 12))))
            chord_progression_name.append("N")
            continue
        max_index = np.argmax(vector)
        similarity = 0
        for reference in vars(cd).items():  # 这个max_index有剧毒solution:原来是reference是个二元tuple
            if reference[1][max_index] == 1 \
                    and float(sklearn.metrics.pairwise.cosine_similarity([vector], [reference[1]])) > similarity:
                similarity = float(sklearn.metrics.pairwise.cosine_similarity([vector], [reference[1]]))
                chord = reference[1]
                chord_name = reference[0]
        #for reference in chord_template(cd):
        #    if reference[1][max_index] == 1 and float(sklearn.metrics.pairwise.cosine_similarity([vector], [reference[1]])) > similarity:
        #        similarity = float(sklearn.metrics.pairwise.cosine_similarity([vector], [reference[1]]))
        #        chord = reference[1]
        #        chord_name = reference[0]
        chord_progression = np.vstack((chord_progression, chord))
        chord_progression_name.append(chord_name)
    return chord_progression.T, chord_progression_name


def to_start_end_name_array(chord_progression_name, timeInSec):
    array = np.zeros((0, 3))
    chord_progression_name.append("chord_progression_name[len(timeInSec)-1]")
    flag = 0
    for i, time in enumerate(timeInSec):
        if chord_progression_name[i] != chord_progression_name[i+1]:
            chord = np.array([flag, timeInSec[i], chord_progression_name[i]])
            array = np.vstack((array, chord))
            flag = timeInSec[i]
    #for j, value in enumerate(array):
    #    if j == array[-1]:
    #        break
    #    if array[j, 2] == array[j+1, 2]:
    #        array[j, 1] == array[j+1, 1]
    #        np.delete(array, j+1, 0)
    del(chord_progression_name[-1])
    return array


def readFile(filepath):
    rat, wav = scipy.io.wavfile.read(filepath)
    duratio = len(wav) / rat
    tim = np.arange(0, duratio, 1 / rat)
    return rat, wav, duratio, tim


def extract_pitch_chroma(X, fs, tfInHz):
    X=X.T
    pitchChroma = np.zeros((12, X.shape[0]))
    for i, frame in enumerate(X):
        for freq in np.arange(start=int(X.shape[1] * 130.81 / fs), stop=int(X.shape[1] * 987.77 / fs)):
            if X[i][freq] > 0:
                pitchChroma[freq2class(fs * (freq / X.shape[1]))][i] = \
                    pitchChroma[freq2class(fs * (freq / X.shape[1]))][i] + X[i][freq]
        if np.max(pitchChroma[:, i]) == 0:
            continue
        pitchChroma[:, i] = pitchChroma[:, i] / np.linalg.norm(pitchChroma[:, i])
    return pitchChroma


def get_cdprg_fromfile(path, matrix):
    WINDOW_LENGTH = int(4096)
    BlockSize = WINDOW_LENGTH
    HopSize = int(2048)
    data = []
    chroma = []
    folder = glob.glob(path + "\\*.wav")
    for count, wavFile in enumerate(folder):
        rate, wave, duration, timeinsec = readFile(wavFile)
        wave = ifDownmix(wave)

        #fs, time, test_spec = scipy.signal.spectrogram(wave, fs=1.0, nfft=WINDOW_LENGTH,
        #                                               window=np.hamming(WINDOW_LENGTH),
        #                                               scaling='spectrum', mode='magnitude', noverlap=WINDOW_LENGTH / 2)
        #test_spec = normalization(test_spec)
        #test_chroma = librosa.feature.chroma_stft(time, rate, test_spec)

        #spec, timeInSec = toSpectrogram(wave, BlockSize, HopSize, rate)
        fs, timeInSec, spec = scipy.signal.spectrogram(wave, fs=1.0, nfft=WINDOW_LENGTH,
                                                       window=np.hamming(WINDOW_LENGTH),
                                                       scaling='spectrum', mode='magnitude', noverlap=WINDOW_LENGTH / 2)
        #spec = select_20_largest_value(spec)
        timeInSec = timeInSec * duration / np.max(timeInSec)
        #chroma = toPitchVector(spec, rate, WINDOW_LENGTH)

        #chroma = extract_pitch_chroma(spec, rate, 452)

        chroma.append(librosa.feature.chroma_stft(duration, rate, spec))

        #plot_all(spec, spec, chroma, chroma)

        chord, chord_progression_name = chord_detection(chroma[count])
        #plot_chordprog(chord, chord, chord_progression_name, chord_progression_name)
        #plot_chordprog(chord, chord, chord_progression_name, chord_progression_name)
        output = to_start_end_name_array(chord_progression_name, timeInSec)
        data.append(output)
    return data, chroma


def plot_chordprog(test_chord, chord, test_chord_name, chord_name):
    sca, ind = plot_label()
    plt.figure()
    plt.subplot(2, 1, 1)
    #plt.yticks(sca, ind)
    #locs1, x_test = plt.xticks(np.arange(len(test_chord_name)), test_chord_name)

    #n_test = len(test_chord_name) - 1
    #while n_test > 0:
    #    if test_chord_name[n_test] == test_chord_name[n_test - 1]:
    #        x_test[n_test].set_visible(False)
    #    n_test = n_test-1

    #plt.imshow(test_chord, cmap=plt.get_cmap("Greys"), aspect="auto")
    #plt.gca().invert_yaxis()
    plt.subplot(2, 1, 2)
    plt.yticks(sca, ind)
    #plt.xticks(np.arange(np.shape(chord)[1]), chord_name)

    #locs2, x = plt.xticks(np.arange(len(chord_name)), chord_name)
    #n = len(chord_name) - 1
    #while n > 0:
    #    if x[n] == x[n - 1]:
    #        x[n].label1.set_visible(False)
    #    n = n - 1

    plt.imshow(chord, cmap=plt.get_cmap("Greys"), aspect="auto")
    plt.gca().invert_yaxis()
    plt.show()


def chord_name(cdname):
    n = cdname.shape[1] - 1
    while n > 0:
        if cdname[n] == cdname[n - 1]:
            cdname[n] = 0
        n = n - 1
    return cdname


def plot_all(test_spec, spec, test_chroma, chroma):
    sca, ind = plot_label()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(test_spec, cmap=plt.get_cmap("inferno"),
               norm=LogNorm(vmin=0.01, vmax=np.max(spec)), aspect="auto")
    plt.gca().invert_yaxis()
    plt.subplot(2, 2, 2)
    plt.imshow(spec, cmap=plt.get_cmap("inferno"), norm=LogNorm(vmin=0.01, vmax=np.max(spec)), aspect="auto")
    plt.gca().invert_yaxis()
    plt.ylim(1, WINDOW_LENGTH / 2)
    plt.subplot(2, 2, 3)
    plt.yticks(sca, ind)
    plt.imshow(test_chroma, cmap=plt.get_cmap("inferno"), aspect="auto")
    plt.gca().invert_yaxis()
    plt.subplot(2, 2, 4)
    plt.yticks(sca, ind)
    plt.imshow(chroma, cmap=plt.get_cmap("inferno"),
               norm=LogNorm(vmin=0.01, vmax=np.max(chroma)), aspect="auto")
    plt.gca().invert_yaxis()
    plt.show()


def plot_label():
    scale = np.arange(12)
    index = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return scale, index


def signal_xHz(A, freq, time_sec, sample_rate):
    return A * np.sin(np.linspace(0, freq * time_sec * 2 * np.pi, sample_rate * time_sec))


WINDOW_LENGTH = int(4096)
BlockSize = WINDOW_LENGTH
HopSize = int(2048)
rate = 44100

T = chord_template()
matrix = tunning_matrix(WINDOW_LENGTH, rate)

dataset = Evaluation.open_dataset("D:\SchoolWork\LAB_RESEARCH\TestDataset\QMUL_beatles/01_-_Please_Please_Me/1")

s = np.arange(start=0, stop=24, dtype=int)
state = create_HMM_state()
Pt, Ps = get_transition_probability_from_annotation(dataset, state, rate)

data, chroma = get_cdprg_fromfile("D:\SchoolWork\LAB_RESEARCH\TestDataset\QMUL_beatles/01_-_Please_Please_Me", matrix)
path = []
for obs in chroma:
    Pe = np.dot(T, obs)
    viterbi(Ps, Pt, Pe)
    #path.append(Viterbi(obs, s, Ps, Pt, Pe))

result = []
for series, ground_truth_series in zip(data, dataset):
    result.append(Evaluation.compute_OR(series, ground_truth_series))
print(result)


