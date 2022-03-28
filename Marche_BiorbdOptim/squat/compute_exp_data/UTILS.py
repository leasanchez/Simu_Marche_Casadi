import numpy as np
from scipy import interpolate, signal
from ezc3d import c3d

def interpolation(x, y, x_new):
    f = interpolate.interp1d(x, y, kind='cubic')
    return f(x_new)

class utils:
    @staticmethod
    def load_c3d(path, list_exp_files, extract_forceplat_data=False):
        loaded_c3d = [c3d(path + file + '.c3d', extract_forceplat_data=extract_forceplat_data) for file in list_exp_files]
        return loaded_c3d

    @staticmethod
    def get_q_name(model):
        q_name = []
        for q in range(model.nbSegment()):
            for d in range(model.segment(q).nbQ()):
                q_name.append(f"{model.segment(q).name().to_string()}_{model.segment(q).nameDof(d).to_string()}")
        return q_name

    @staticmethod
    def load_txt_file(file_path, size):
        data_tp = np.loadtxt(file_path)
        nb_frame = int(len(data_tp) / size)
        out = np.zeros((size, nb_frame))
        for n in range(nb_frame):
            out[:, n] = data_tp[n * size: n * size + size]
        return out

    @staticmethod
    def define_butterworth_filter(fs, fc):
        w = fc / (fs / 2)
        b, a = signal.butter(4, w, 'low')
        return b, a

    @staticmethod
    def has_nan(x):
        return np.isnan(np.sum(x))

    @staticmethod
    def fill_nan(y):
        x = np.arange(y.shape[0])
        good = np.where(np.isfinite(y))
        f = interpolate.interp1d(x[good], y[good], bounds_error=False, kind='cubic')
        y_interp = np.where(np.isfinite(y), y, f(x))
        return y_interp

    @staticmethod
    def divide_squat_repetition(x, index):
        x_squat = []
        for idx in range(int(len(index) / 2)):
            x_squat.append(x[:, index[2 * idx]:index[2 * idx + 1]])
        return x_squat

    @staticmethod
    def interpolate_repetition(x, index, freq):
        x_squat = utils.divide_squat_repetition(x, index)
        x_interp = []
        for (i, r) in enumerate(x_squat):
            start = np.arange(0, r.shape[-1])
            interp = np.linspace(0, start[-1], 2*freq)
            x_interp.append(interpolation(start, r, interp))
        return x_interp

    @staticmethod
    def compute_mean(x, index, freq):
        x_interp = utils.interpolate_repetition(x, index, freq)
        mean = np.mean(np.array(x_interp), axis=0)
        std = np.std(np.array(x_interp), axis=0)
        return mean, std