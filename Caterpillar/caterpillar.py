import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

modules_path = os.path.join(os.getcwd(), 'modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

from vidpreprocess import Cropper
from flightdetection import FlightDetector
from ann import ANNImplementer, ModResNet18

class Caterpillar:
    def __init__(self, conf_path, fd_kwargs, sd_path='model_state_dict.pt'):
        self.conf_path = conf_path
        self.model = ModResNet18(2, custom_classifier=True)
        self.model.load_state_dict(torch.load(sd_path))

        self.params = pd.read_csv('conf.csv', header=None, index_col=0)
        wd = self.params.loc['working_dir', 1]
        os.chdir(wd)

        result_dirs = {'params_dir' : 'video_params',
                       'track_dir': 'tracks',
                       'flight_dir': 'flight_labels',
                       'save_video_dir': 'video_results'
                       }

        for key in result_dirs.keys():
            dirname = make_subdir(self.params.loc['video_dir', 1], result_dirs[key])
            self.params.loc[key, 1] = dirname

        ann_kwargs, crop_kwargs, fd_class_kwargs = get_kwargs(self.params)

        self.fd_kwargs = fd_kwargs
        self.mode = int(self.params.loc['mode', 1])
        self.name = self.params.loc['video_name', 1]

        if self.name == 'all':
            crop_kwargs['allinfolder'] = True
            crop_kwargs['name'] = None
            crop_kwargs['video_path'] = self.params.loc['video_dir', 1]

        else:
            crop_kwargs['allinfolder'] = False
            crop_kwargs['name'] = self.name[:-4]
            video_path = os.path.join(self.params.loc['video_dir', 1], self.name)
            crop_kwargs['video_path'] = video_path


        self.crop = Cropper(**crop_kwargs)
        self.ann_imp = ANNImplementer(model=self.model, video_name=None,
                                      mode=None, **ann_kwargs)

        self.fd_class_kwargs = fd_class_kwargs

    def run(self):
        video_dir = self.params.loc['video_dir', 1]

        if int(self.params.loc['preprocess', 1]):
            self.crop.preprocess()

        if self.name == 'all':
            videos = [i for i in os.listdir(video_dir) if i[-4:] == '.mp4']

            detectors = []
            for video in videos:
                fdr = FlightDetector(video[:-4], **self.fd_class_kwargs)
                detectors.append(fdr)

            print('Making flying labels:')
            for detector in tqdm(detectors):
                detector.detect(self.fd_kwargs)

            all_fine = input('Is everything fine? [Y]/n: ')
            if all_fine == 'n':
                txt = 'Select the bad videos (no commas, only spaces): '
                bad_ones = input(txt).split()
                bad_modes = []
                for bad_one in bad_ones:
                    bad_mode = input('{}; value to replace all labels [0]/1 : '.format(bad_one))
                    bad_modes.append(bad_mode == '1')

                for detector in detectors:
                    if detector.video_name in bad_ones:
                        bad_mode = bad_modes[bad_ones.index(detector.video_name)]
                        if bad_mode:
                            detector.one_label()
                        else:
                            detector.zero_label()

            else:
                bad_ones = []
                bad_modes = []

        else:
            videos = [self.name]
            detector = FlightDetector(self.name[:-4], **self.fd_class_kwargs)
            detector.detect(self.fd_kwargs)
            all_fine = input('Is everything fine? [Y]/n : ')
            if all_fine == 'n':
                toones = input('Select the value to replace all labels [0]/1 : ')
                if toones == '1':
                    detector.one_label()
                else:
                    detector.zero_label()

                bad_ones = [detector.video_name]
                bad_modes = [toones == '1']

            else:
                bad_ones = []
                bad_modes = []

        names, modes, orientation, ratios = run_ann(self.ann_imp, videos,
                                                    self.mode,
                                                    bad_ones,
                                                    bad_modes,
                                                    int(self.params.loc['make_track', 1]),
                                                    int(self.params.loc['make_video', 1])
                                                    )

        checkfile = os.path.isfile('resulting_angles_{}.csv'.format(video_dir))
        with open('resulting_angles_{}.csv'.format(video_dir), 'a') as inf:
            if not checkfile:
                inf.write('video_name,mode,orientation,flattering flight ratio\n')

            for nm, md, ort, fr in zip(names, modes, orientation, ratios):
                inf.write('{},{},{},{}\n'.format(nm, md, ort, fr))

def get_kwargs(df):
    ann_kwargs = ['video_dir', 'params_dir', 'track_dir',
                  'flight_dir', 'save_video_dir']

    ann_kwargs = dict.fromkeys(ann_kwargs)
    for key in ann_kwargs.keys():
        ann_kwargs[key] = df.loc[key, 1]

    if int(df.loc['use_gpu', 1]):
        ann_kwargs['device'] = 'cuda'

    else:
        ann_kwargs['device'] = 'cpu'

    crop_kwargs = {'save_path': df.loc['params_dir', 1]}
    fd_class_kwargs = {'load_dir': df.loc['params_dir', 1],
                       'save_dir': df.loc['flight_dir', 1]
                       }

    return ann_kwargs, crop_kwargs, fd_class_kwargs

def run_ann(ann_imp, videos, mode, bad_ones, bad_modes, track, isvideo):
    name = []
    modes = []
    orientation = []
    ratios = []

    modes_option = [[0], [1], [2], [0, 1, 2]]
    mode = modes_option[mode]

    for video in videos:
        if (mode == modes_option[3]) and (video[:-4] in bad_ones):
            ann_imp.mode = 0
            ann_imp.next_video(video)
            if track:
                print('Video {}; create track mode {}'.format(ann_imp.video_name,
                                                                ann_imp.mode))
                angle = ann_imp.make_track()
                f_ratio = ann_imp.params[0]
                f_ratio = f_ratio.sum()/f_ratio.size

                name.append(video[:-4])
                modes.append(0)
                orientation.append(angle)
                ratios.append(f_ratio)

            if isvideo:
                print('Video {}; create video mode {}'.format(ann_imp.video_name,
                                                                ann_imp.mode))
                ann_imp.make_video()

            continue

        ann_imp.next_video(video)
        for m in mode:
            ann_imp.mode = m
            if track:
                print('Video {}; create track mode {}'\
                        .format(ann_imp.video_name, ann_imp.mode))

                angle = ann_imp.make_track()
                f_ratio = ann_imp.params[0]
                f_ratio = f_ratio.sum()/f_ratio.size

                name.append(video[:-4])
                modes.append(m)
                orientation.append(angle)
                ratios.append(f_ratio)

            if isvideo:
                print('Video {}; create video mode {}'\
                        .format(ann_imp.video_name, ann_imp.mode))

                ann_imp.make_video()

    return name, modes, orientation, ratios

def make_subdir(parent_dir, name):
    dirname = os.path.join(parent_dir, name)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    return dirname

if __name__ == '__main__':
    fd_kwargs = {'window_size': 50,
                 'percent': 99,
                 'show': True
                 }

    cpl = Caterpillar('conf.csv', fd_kwargs)
    cpl.run()
