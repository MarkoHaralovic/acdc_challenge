# ACDC Challenge

---

## 1. Environment Setup

The `acdc_segmenter` requires **Python >= 2.7 and <= 3.7**.  

Create and activate a virtual environment:

```bash
conda create -n acdc_venv37
conda activate acdc_venv37
pip install --upgrade pip
```

## 2. Install Dependencies
Navigate to the segmenter directory and install the required packages.
```bash
cd acdc_segmenter
pip install -r requirements.txt
pip install git+https://github.com/lmkoch/medpy/@b06b6decf41c63489e746f6a83e8fa5ff509adfa#egg=MedPy
pip install tensorflow-gpu==1.13.1

# since CUDA requirements are that of version 10.0, install appropriate cuda deps
conda install cudatoolkit=10.0 cudnn=7.4

```

## 3. Download the Dataset
Download the ACDC dataset and place it in a directory of your choice.


## 4. Configure System Paths

Open the system configuration file:
```bash
acdc_challenge/acdc_segmenter/config/system.py
```

Modify the configuration variables to match your local setup:
```bash
at_biwi = False

project_root = "path_to/acdc_segmenter"
data_root = "path_to_acdc_dataset/training"
test_data_root = "path_to_acdc_dataset/testing"

local_hostnames = ["localhost"]
```


## How to use the repo

## Experiment configurations
under acdc_segmenter/experiments we define the configurations to run the model with. There are all the configurations that we can use.

Looking at each, these are important parameters we can ablate:

**Model:**
- `model_handle`: defines which model to use, options: standard 2D U-net, standard 3D U-net, modified 2D U-net, FCN-8

**Data:**
- `data_mode`: 2D or 3D
- `image_size`: the target spatial size each image is transformed into -> default (212, 212)
- `target_resolution`: the target in-plane resolution -> default (1.36719, 1.36719)
- `nlabels`: number of segmentation classes -> default 4

**Training:**
- `batch_size`: batch size for training -> default 10
- `learning_rate`: learning rate for optimizer -> default 0.01
- `optimizer_handle`: which optimizer to use for the training -> options: SGD, ADAM, ADAMW
- `loss_type`: which loss to use for our experiments -> options: Cross Entropy, weighted Cross Entropy, DICE
- `max_epochs`: num of epochs to train for

**Augmentations:**
- `augment_batch`: whether to apply augmentations
- `do_rotations`: rotation augmentation
- `do_scaleaug`: scale augmentation
- `do_fliplr`: horizontal flip
