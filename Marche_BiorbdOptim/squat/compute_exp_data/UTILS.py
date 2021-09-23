import numpy as np
from scipy import interpolate, signal


def interpolation(x, y, x_new):
    f = interpolate.interp1d(x, y, kind='cubic')
    return f(x_new)

class utils:
    @staticmethod
    def define_butterworth_filter(fs, fc):
        """
        define filter parameters
        input : fs : sample frequency, fc : cut frequency
        output : signal parameter for low pass filter
        """
        w = fc / (fs / 2)
        b, a = signal.butter(4, w, 'low')
        return b, a

    @staticmethod
    def has_nan(x):
        return np.isnan(np.sum(x))

    @staticmethod
    def fill_nan(y):
        '''
        interpolate to fill nan values using cubic interpolation
        '''
        x = np.arange(y.shape[0])
        good = np.where(np.isfinite(y))
        f = interpolate.interp1d(x[good], y[good], bounds_error=False, kind='cubic')
        y_interp = np.where(np.isfinite(y), y, f(x))
        return y_interp

    @staticmethod
    def divide_squat_repetition(x, index):
        x_squat = []
        for idx in range(int(len(index) / 2)):
            x_squat.append(x[:, :, index[2 * idx]:index[2 * idx + 1]])
        return x_squat

    @staticmethod
    def interpolate_repetition(x, index):
        x_squat = utils.divide_squat_repetition(x, index)
        x_interp = []
        for (i, r) in enumerate(x_squat):
            start = np.arange(0, r.shape[-1])
            interp = np.linspace(0, start[-1], 200)
            r_new = np.ndarray((3, 52, 200))
            for m in range(r.shape[1]):
                r_new[:, m, :] = np.array([interpolation(start, r[i, m, :], interp) for i in range(3)])
            x_interp.append(r_new)
        return np.array(x_interp)

    @staticmethod
    def compute_mean(x, index):
        x_interp = utils.interpolate_repetition(x, index)
        mean = np.mean(x_interp, axis=0)
        std = np.std(x_interp, axis=0)
        return mean, std