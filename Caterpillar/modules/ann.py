import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2
import os
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models

from tqdm import tqdm, trange

class ModResNet18(models.resnet.ResNet):
    def __init__(self, out_features=10, custom_classifier=False):
        super(ModResNet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2],
                                          num_classes=out_features)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), bias=False)

        if custom_classifier:
            classifier = Classifier(self.fc.in_features, out_features)
            self.fc = classifier

class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return torch.tanh(x)

class ANNImplementer:
    def __init__(self, model, video_name, video_dir, params_dir, flight_dir,
                 track_dir, save_video_dir, mode=1, device='cpu'):

        '''
        mode: 0, 1, 2; 0 -> flight=0; 1 -> flight=1; 2 -> all labels
        '''

        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self.video_dir = video_dir
        self.params_dir = params_dir
        self.flight_dir = flight_dir
        self.track_dir = track_dir
        self.save_video_dir = save_video_dir

        self.video_name = video_name
        if self.video_name != None:
            self.params = get_params(video_name, params_dir, flight_dir)

        self.mode = mode

        self.orientation = None

    def next_video(self, video_name):
        self.video_name = video_name
        self.params = get_params(video_name, self.params_dir, self.flight_dir)


    def make_track(self, l_dimension=224, show=0):
        labels, times, h1, h2, w1, w2 = self.params
        if self.mode == 2:
            indexes = times

        else:
            indexes = np.where(labels == self.mode)[0]
            indexes.sort()
            indexes = times[indexes]

        #indexes = indexes[:250]
        if not indexes.size:
            return None

        cap = cv2.VideoCapture(os.path.join(self.video_dir, self.video_name))
        result = []

        for i in trange(indexes[-1] + 1):
            ok, frame = cap.read()
            if ok:
                if i in indexes:
                    frame = frame[h1:h2, w1:w2]
                    frame = cv2.resize(frame, (l_dimension, l_dimension), cv2.INTER_LINEAR)
                    tt = np.array([frame[..., 2], frame[..., 1], frame[..., 0]])
                    tt = tt[np.newaxis, :]
                    tt = torch.tensor(tt).float()
                    tt = tt.to(self.device)

                    with torch.no_grad():
                        y, x = self.model(tt)[0].to('cpu').detach().numpy()

                    angle = np.angle(x + y*1j)

                    result.append([i, np.cos(angle), -np.sin(angle),
                                   labels[i - times[0]]])

            else:
                raise Exception('Cannot read video!')

        cap.release()

        result = np.array(result)
        df = {'FrameN': result[:, 0],
              'Cosine': result[:, 1],
              'Sine': result[:, 2],
              'IsFlying': result[:, 3]
              }

        df = pd.DataFrame(df)
        track_name_csv = '{}_mode_{}.csv'.format(self.video_name[:-4], self.mode)
        track_name_csv = os.path.join(self.track_dir, track_name_csv)
        df.to_csv(track_name_csv, index=False)

        track_name_plot = '{}_mode_{}.png'.format(self.video_name[:-4], self.mode)
        track_name_plot = os.path.join(self.track_dir, track_name_plot)
        orientation = create_track(track_name_plot, result[:, 1:], show)
        return orientation

    def make_video(self, fps=25, l_dimension=224):
        labels, times, h1, h2, w1, w2 = self.params
        if self.mode == 2:
            indexes = times

        else:
            indexes = np.where(labels == self.mode)[0]
            indexes.sort()
            indexes = times[indexes]

        indexes = indexes[:500]
        if not indexes.size:
            return None

        cap = cv2.VideoCapture(os.path.join(self.video_dir, self.video_name))

        out_name = '{}_mode_{}.mp4'.format(self.video_name[:-4], self.mode)
        out_name = os.path.join(self.save_video_dir, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (l_dimension, l_dimension)
        writer = cv2.VideoWriter(out_name, fourcc, fps, size)

        for i in trange(indexes[-1] + 1):
            ok, frame = cap.read()
            if ok:
                if i in indexes:
                    frame = frame[h1:h2, w1:w2]
                    frame = cv2.resize(frame, (l_dimension, l_dimension), cv2.INTER_LINEAR)
                    tt = np.array([frame[..., 2], frame[..., 1], frame[..., 0]])
                    tt = tt[np.newaxis, :]
                    tt = torch.tensor(tt).float()
                    tt = tt.to(self.device)

                    with torch.no_grad():
                        y, x = self.model(tt)[0].to('cpu').detach().numpy()

                    frame = plot_angle(frame, x, y)
                    writer.write(frame)

            else:
                raise Exception('Cannot read video!')

        cap.release()
        writer.release()

def create_track(name, vectors, show):
    labels = vectors[:, 2]
    vectors = vectors[:, :2]
    track = [np.sum(vectors[:i], axis=0) for i, val in enumerate(vectors)]
    track = np.array(track)
    #print(track.shape)
    x_final, y_final = track[-1]
    angle = np.angle(x_final + 1j * y_final, deg=True)
    displacement = (x_final**2 + y_final**2)**0.5/vectors.shape[0]
    text = '{}\nAngle, deg: {:.2f}\nDisplacement/track: {:.2f}'.\
                                    format(name, angle, displacement)

    colors = ['r' if f else 'b' for f in labels]
    lines = [(x0, x1) for x0, x1 in zip(track[:-1], track[1:])]
    colored_lines = LineCollection(lines, colors=colors)

    f, ax = plt.subplots(figsize=(10, 10))
    minx = track[:, 0].min()
    maxx = track[:, 0].max()
    miny = track[:, 1].min()
    maxy = track[:, 1].max()
    ax.set_xlim((minx - 0.1*(maxx - minx), maxx + 0.1*(maxx - minx)))
    ax.set_ylim((miny - 0.1*(maxy - miny), maxy + 0.1*(maxy - miny)))
    ax.set_aspect('equal')
    ax.add_collection(colored_lines)
    ax.set_title(text, loc='left')
    ax.grid()
    plt.savefig(name, dpi=600, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()
    return angle

def plot_angle(frame, x, y, length=46, color=[255, 255, 0]):
    angle = np.angle(x + y*1j)
    center = frame.shape[0]//2
    dot = [center + int(length*np.cos(angle)), center + int(length*np.sin(angle))]
    frame = cv2.line(frame, [center, center], dot, color, 5)
    return frame

def get_params(video, params_dir, flight_dir):
    name = video[:-4]
    params_path = os.path.join(params_dir, 'crop_params{}.npy'.format(name))
    flight_path = os.path.join(flight_dir, '{}_labels.npy'.format(name))
    times_path = os.path.join(flight_dir, '{}_time.npy'.format(name))
    h1, h2, w1, w2 = np.load(params_path)
    times = np.load(times_path)
    labels = np.load(flight_path)
    #print(times_path, times)
    return labels, times, h1, h2, w1, w2
