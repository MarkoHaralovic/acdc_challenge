# ACDC Challenge

---

## 1. Environment Setup

The `acdc_segmenter` requires **Python >= 2.7 and <= 3.7**.  

Create and activate a virtual environment:

```bash
python3.7 -m venv acdc_venv37
source acdc_venv37/bin/activate
pip install --upgrade pip
```

## 2. Install Dependencies
Navigate to the segmenter directory and install the required packages.
```bash
cd acdc_segmenter
pip install -r requirements.txt
pip install git+https://github.com/lmkoch/medpy/@b06b6decf41c63489e746f6a83e8fa5ff509adfa#egg=MedPy
pip install tensorflow-gpu==1.13.1
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