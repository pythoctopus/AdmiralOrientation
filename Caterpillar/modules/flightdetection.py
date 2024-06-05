import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import cluster

class FlightDetector:
    def __init__(self, video_name, load_dir, save_dir):
        self.bgr_file = '{}/bgr_mean{}.npy'.format(load_dir, video_name)
        self.plot_path = '{}/{}.jpg'.format(save_dir, video_name)

        means = np.load(self.bgr_file)
        self.times = means[:, 1]
        self.means = np.sum(means[:, 2:], axis=1)/3
        del means

        self.save_dir = save_dir
        self.load_dir = load_dir
        self.video_name = video_name

        self.labels = None

    def detect(self, df_kwargs=None):
        t, l = detect_flight(self.means, self.plot_path, self.times,
                             **df_kwargs)

        self.labels = l
        tname = '{}/{}_time.npy'.format(self.save_dir, self.video_name)
        lname = '{}/{}_labels.npy'.format(self.save_dir, self.video_name)
        np.save(tname, np.int64(t+df_kwargs['window_size']//2))
        np.save(lname, l)

    def zero_label(self):
        lname = '{}/{}_labels.npy'.format(self.save_dir, self.video_name)
        newlabels = np.zeros_like(self.labels)
        np.save(lname, newlabels)

    def one_label(self):
        lname = '{}/{}_labels.npy'.format(self.save_dir, self.video_name)
        newlabels = np.ones_like(self.labels)
        np.save(lname, newlabels)

def detect_flight(arr, name, times, window_size=50, percent=99, show=0):
    clusterer = cluster.KMeans(2, n_init=50)
    diffs = comp_diff(arr, window_size)
    pr = np.percentile(diffs, percent)
    diffs *= diffs<=pr
    prediction = clusterer.fit(diffs)
    lbs = prediction.labels_
    if np.mean(diffs[np.where(lbs > 0)]) < np.mean(diffs[np.where(lbs < 1)]):
        lbs = 1 - lbs

    f, ax = plt.subplots(figsize=(20, 5))
    ax.plot(times[:-window_size], arr[:-window_size] + 100, 'black')
    ax.plot(times[:-window_size], lbs*50)
    ax.grid()
    plt.savefig(name, dpi=120)
    if show:
        plt.show()

    plt.close()
    return times[:-window_size], lbs

def comp_diff(arr, win_size):
        ranges = []
        for i in range(arr.shape[0] - win_size):
            w0 = arr[i:i + win_size]
            w_range = np.max(w0) - np.min(w0)
            ranges.append([w_range, 0])

        return np.array(ranges)
