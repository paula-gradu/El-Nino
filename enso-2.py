import matplotlib.pyplot as plt
import numpy as np

def ONI(signal, m = 3):
    """Return the ONI of the signal."""
    oni = np.array(signal)
    length = signal.shape[0]
    for i in range(length):
        oni[i] = np.mean(signal[max(0, (i - m + 1)) : min((i + 1), length)])
    return oni

def climatology(signal):
    """Return the climatology of the signal."""
    clm = np.zeros(12)
    length = signal.shape[0]
    for month in range(12):
        section = [12 * i + month for i in range(length // 12)]
        clm[month] = np.mean(signal[section])
    return clm

def anomaly(signal, clm):
    """Return the anomaly of the signal by subtracting the monthly climatology."""
    anm = np.array(signal)
    length = signal.shape[0]
    for i in range(length):
        anm[i] = signal[i] - clm[i % 12]
    return anm

def correlation(pred, true):
    """Return the correlation between the predicted signal and the true signal."""
    pred0 = pred - np.mean(pred, axis = 0)
    true0 = true - np.mean(true, axis = 0)
    return np.mean(pred0 * true0, axis = 0) / np.sqrt((np.mean(pred0**2, axis = 0) * (np.mean(true0**2, axis = 0))))

def rmse(pred, true):
    """Return the RMSE between the predicted signal and the true signal."""
    return np.sqrt(((pred - true)**2).mean(axis = 0))

def normalize(signal):
    """Normalize the given signal."""
    return (signal - np.mean(signal)) / np.std(signal)

def ts2history(signals, T, H):
    """Transform time series to supervised learning format."""
    signal_length = signals.shape[0]
    n_signals = signals.shape[1]
    size = signal_length - H - T 
    data = np.ndarray((size, H, n_signals))
    for i in range(size):
        data[i, 0:H, :] = signals[i:(i + H), :]
    return data

def ts2diff(signal, T, H):
    """Return the difference between value at current time and at prediction time."""
    signal_length = signal.shape[0]
    size = signal_length - H - T 
    diff = np.ndarray((size, T), dtype = np.float64)
    for t in range(T):
        for i in range(size):
            diff[i, t] = signal[i + H + t] - signal[i + H - 1]
    return diff

def ts2remainder(signal, T, H):
    """Return the value at current time."""
    signal_length = signal.shape[0]
    size = signal_length - H - T
    remainder = np.ndarray((size, T), dtype = np.float64)
    for t in range(T):
        for i in range(size):
            remainder[i, t] = signal[i + H - 1]
    return remainder

def time2time2D(time, T, H):
    """Create time2D from time."""
    size = time.shape[0] - H - T
    time2D = np.ndarray((size, T))
    for t in range(T):
        for i in range(size):
            time2D[i, t] = time[i + H + t]
    return time2D

def split(data, start, end):
    """Split data into training and validation."""
    size = data.shape[0]
    split = size // 10   
    j = np.arange(size, dtype = int)
    j_t = j.copy()
    j_v = j.copy()
    j_v = j_v[start * split: end * split]
    j_t = np.delete(j_t, j_v)
    train = np.array(data[j_t])
    val = np.array(data[j_v])
    return (train, val)

def LSTM_prep(data, batch_size):
    new_shape = range(data.shape[0] - data.shape[0] % batch_size)
    return data[new_shape]

def LSTM_prep_tuple(tup, batch_size):
    (x, y) = tup
    return (LSTM_prep(x, batch_size), LSTM_prep(y, batch_size))

def plot_training(training_history):
    """Plot training errors."""
    plt.plot(training_history.history['mean_squared_error']);
    plt.plot(training_history.history['mean_absolute_error']);
    plt.ylabel('error');
    plt.xlabel('epoch');

def plot_monthly_correlation(pred, true, T, cmap = None):
    """Plot correlation between prediction and truth for each month."""
    monthly_corr = np.zeros((12, T))
    for m in range(12):
        monthly_corr[m] = correlation(true[m::12], pred[m::12])
    plt.pcolormesh(monthly_corr.T, cmap = cmap);
    plt.colorbar();

def plot_monthly_rmse(pred, true, T, cmap = None):
    """Plot RMSE between prediction and truth for each month."""
    monthly_rmse = np.zeros((12, T))
    for m in range(12):
        monthly_rmse[m] = rmse(true[m::12], pred[m::12])
    plt.pcolormesh(monthly_rmse.T, cmap = cmap);
    plt.colorbar();

def persistance_corr_rmse(signal, T):
    """Return correlation and RMSE of persistance."""
    persistance_corr = np.zeros(T)
    persistance_rmse = np.zeros(T)

    for t in range(T):
        persistance_corr[t] = correlation(signal[:(- t - 1)], signal[(t + 1):])
        persistance_rmse[t] = rmse(signal[:(- t - 1)], signal[(t + 1):])

    return (persistance_corr, persistance_rmse)

def plot_ts(pred, true, section, T):
    """Plot true versus predicted time series for each prediction timeline for the given section."""
    plt.figure(figsize = (8, 10))
    for t in range(T):
        plt.subplot(4, 3, t + 1);
        plt.plot(true[:, t][section], '--', label = "Truth", color = 'blue');
        plt.plot(pred[:, t][section], ':', label = "Prediction", color = 'red');
        plt.title(t + 1);
        plt.legend();