This folder contains all the details regarding ANN training. The notebook contains all the necessary code.
Files angular_differences.csv and angular_differences_human.csv contain model and human error in degrees, obtained on the test dataset
### Data
The model is ResNet18 with a modified linear classifier. The model was trained on 1500 frames randomly selected from the videos of 2021-2022. The test dataset contained 480 frames. The labels at the frames include the coordinates of both antennas, the head, thorax, and the bottom of the abdomen.

### Training

All the data (including the test dataset) is located under the **TrainDataset** directory. Training and evaluation loss is stored under the **runs** directory of the tensorboard writer implemented in PyTorch (also in the plot below).
We trained the model for 500 epochs with a batch size of 40 and an lr parameter of 10**-3. We used L1Loss and Adam optimizer.

![Loss](https://github.com/pythoctopus/AdmiralOrientation/assets/56726936/928e6f67-aa7f-49cb-a82a-fad5b4550b53)
