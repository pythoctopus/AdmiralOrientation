# AdmiralOrientation
In order to analyze the behavior of the red admiral from video recordings, we employed a two-step approach. First, we divided the behaviors of the butterflies into two subcategories: active fluttering flight and passive hanging inside the flight simulator. The regions of interest where the butterfly exhibited active flight were later analyzed using a ResNet18-based CNN to determine the animal's orientation. The summary of the model is depicted below:
![admiral_model](https://github.com/pythoctopus/AdmiralOrientation/assets/56726936/e747bb69-3890-4572-8a13-1d61ce6474bd)

At the first stage, the original recording (a) is cropped to the region of interest containing the butterfly (b). These cropped frames are further used to compute the mean values across the three channels, resulting in a time series where each point corresponds to the mean value of pixel brightness for a specific frame (c). As the butterfly flaps its wings, the brightness of the frame changes accordingly, thus allowing us to distinguish between active flight and passive hanging. The means are used to compute a range of brightness values within a timeframe of 2 seconds. We applied K-means clustering to these ranges to determine whether the butterfly was actively flying (d). The "active" frames were then further analyzed by the CNN to determine the butterfly's orientation (sine and cosine of the angle). This orientation allows us to recreate a track of the butterfly's movement inside the flight simulator (e).

>[!NOTE]
>Flight analysis based on mean values of brightness is a pretty straight-forward decision for our setup as butterflies clearly show the upmentioned pattern:
>![bfl](https://github.com/pythoctopus/AdmiralOrientation/assets/56726936/00c41998-cede-4524-b30e-912d5e8071d2)

## CNN
We used a pretrained ResNet18 with a custom 
