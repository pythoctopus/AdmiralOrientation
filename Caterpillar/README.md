# AdmiralOrientation
ANN approach for butterfly orientation measurement inside a Mouritsen-Frost flight simulator

So I will try to make a little description. I sincerely hope one day someone
aside from me will be able to use this stuff.
I'm not going to describe the general principles of the model's work in detail here.
Just as a little reminder:

1) Original video preprocessing (video crop, average values per frame per channel
calculation)

2) Flight labels calculation

3) Track/Labelled video creation

Currently, all the info for the model is stored in the conf.csv file. It contains
8 parameters:
```text
__________________________________________________________________________
preprocess      1 | 0                   Do you need to preprocess the video?
                                        0 only if you already did that

working_dir     working directory       Directory where the video folder is
                                        located

video_name      all | video_name        Name of the video of interest
                                        (the only supported extension
                                        is .mp4), If you want to analyse
                                        all videos in some folder,
                                        use 'all'

video_dir       Directory where         Folder with videos
                videos are stored

mode            0 | 1 | 2 | 3           There are 4 available modes:
                                        0: model chooses parts of the video
                                        where the butterfly does not flutter
                                        1: model chooses parts of the video
                                        where the butterfly flutters
                                        2: model takes both parts where the
                                        butterfly does and does not flutter
                                        3: model goes through all three
                                        previously described modes

make_track      1 | 0                   Do you want to plot the track?

make_video      1 | 0                   Do you want to create a labeled
                                        video?

use_gpu         1 | 0                   Do you want to use GPU?
__________________________________________________________________________
```
If all of these parameters are filled correctly, everything should work just fine.

To run the model you just need to run a Python script

>[!WARNING]
>YOU NEED pytorch, openCV, and tqdm installed

The configuration file can be feeded as a command line argument. If this option is ignored, the default file is __'conf.csv'__ in the script directory.

E.g you can run the script as:
```bash
python script-directory/catterpillar.py path-to-conf
```
or
```bash
python script-directory/catterpillar.py
```
The latter is equal to:
```bash
python script-directory/catterpillar.py script-directory/conf.csv
```
The script will ask you to:

1) select a region of interest for each video
2) after fluttering labels are predicted, it will ask to clarify that the labels are
predicted correctly. If not, it will ask with what value you want to fill these
labels (1 or 0)

## OUTPUTS
are stored in folders automatically created inside video contained directory.
These can contain fluttering labels/plots, tracks (plot & csv with an orientation at
each step), labeled videos

The model also creates a summary file in the working directory (WD) with name, orientation,
mode, and fluttering ratio (fluttering time/total duration) for all analysed videos
