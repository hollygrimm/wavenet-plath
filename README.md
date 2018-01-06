# Single Voice Training and Synthesizing using WaveNet

Generate raw audio waveforms using WaveNet, a deep neural network.

## Dependencies

1. [tensorflow 1.4.1](https://www.tensorflow.org/install/)
1. [cadl](https://github.com/pkmital/pycadl)

## Dataset

Three MP3 files with a total of 80 minutes of poetry spoken by Sylvia Plath

## Pre-processing Dataset

For this project, each main MP3 file was considered a chapter. Create a folder for each chapter and add it to the sounds directory:

* sounds/
    * CHAPTERNAME/
        * CHAPTERNAME-CLIPNUMBER.mp3

Training was done using short sequences, so I used Audacity to break the ~30 minute MP3 files into smaller clips using the following steps:

* Select *Analyze..Sound Finder* with the following settings to create labels
  * Treat audio below this level as silence [-dB] = 26.0
  * Minimum duration of silence between sounds [seconds] = .250
  * Label starting point = .1
  * Label ending point = .1
* Select *Edit...Preferences Import/Export* and turn off "Show Metadata Editor prior to export step"
* Select *File...Export Multiple* with the following settings
  * Format: MP3
  * Numbering after File name prefix: *CHAPTERNAME*

Place the MP3 clips under sounds/CHAPTERNAME. Listen to clips below 30k in file size, and delete any clips that are silent.


## Training

### Hyperparameters
The batch size is set to 2 for a 2GB GPU. It should be increased if you have more GPU memory.

Execute

```python wavenet-plath.py```

## Monitor Training

Execute

```tensorboard --logdir=.``` to view loss chart and audio synthesis


## Synthesizing
Generate a 10-second wav file using the trained model.

```
import wavenet_plath
wavenet_plath.synthesize()
```
