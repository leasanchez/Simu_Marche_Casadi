import numpy as np
from scipy import interpolate, signal

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