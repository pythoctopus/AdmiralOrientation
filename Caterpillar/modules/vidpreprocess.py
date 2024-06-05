# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

from tqdm import tqdm

class Cropper:
    def __init__(self, name, video_path, save_path, allinfolder):
        self.video_path = video_path
        self.save_path = save_path
        self.allinfolder = allinfolder
        self.name = name

    def preprocess(self):
        if self.allinfolder:
            videos = [i for i in os.listdir(self.video_path)\
                      if i[-4:] == '.mp4']
            for video in tqdm(videos):
                read_path = os.path.join(self.video_path, video)
                center, side = find_crop(read_path)
                w1, h1 = center - side
                w2, h2 = center + side
                crop_name = 'crop_params' + video[:-4] + '.npy'
                np.save(os.path.join(self.save_path, crop_name),
                        np.array([h1, h2, w1, w2]))

            for video in tqdm(videos):
                read_path = os.path.join(self.video_path, video)
                means = find_means(read_path, h1, h2, w1, w2)
                mean_name = 'bgr_mean' + video[:-4] + '.npy'
                np.save(os.path.join(self.save_path, mean_name), means)

        else:
            center, side = find_crop(self.video_path)
            w1, h1 = center - side
            w2, h2 = center + side
            crop_name = 'crop_params' + self.name + '.npy'
            np.save(os.path.join(self.save_path, crop_name),
                    np.array([h1, h2, w1, w2]))

            means = find_means(self.video_path, h1, h2, w1, w2)
            mean_name = 'bgr_mean' + self.name + '.npy'
            np.save(os.path.join(self.save_path, mean_name), means)

def find_crop(video_path):
    X = []
    ds = []
    drawing = False
    dist = 0
    cap = cv2.VideoCapture(video_path)
    for i in range(100):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            print(video_path)
            raise Exception('cannot read video')

    def selectROI(event, x, y, flags, param):
        img = frame.copy()
        nonlocal drawing, X, ds, dist
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            X.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                dist = np.abs(X[-1][0] - x)
                pt1 = (X[-1][0]-dist, X[-1][1]-dist)
                pt2 = (X[-1][0]+dist, X[-1][1]+dist)
                cv2.rectangle(img, pt1, pt2, [0, 0, 255])
                cv2.imshow('ROIselection', img)
        elif event == cv2.EVENT_LBUTTONUP:
            ds.append(dist)
            drawing = False

    cv2.imshow('ROIselection', frame)
    cv2.setMouseCallback('ROIselection', selectROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    return (np.array(X[-1]), ds[-1])

def find_means(video_path, h1, h2, w1, w2):
    result = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    while True:
        ok, frame = cap.read()
        if ok:
            frame = frame[h1:h2, w1:w2]
            B = frame[:, :, 0].mean()
            G = frame[:, :, 1].mean()
            R = frame[:, :, 2].mean()
            result.append([fps, count, B, G, R])
            #if count >= 100: break
            count += 1
        else:
            break

    cap.release()
    return np.array(result)
