# ACDC Challenge: UTwente JupyterLab Setup & Training Guide

**Running Legacy TF1.x Code on a Modern CUDA 11.8 Server**

## Overview & The Core Problem

This guide outlines the exact steps required to run the legacy ACDC Challenge segmentation codebase (originally written for TensorFlow 1.x) on the **University of Twente (UT) JupyterLab servers**.

**The Conflict:** The original codebase is old and expects an older version of TensorFlow (1.x) and an older GPU driver. However, the UT server is equipped with modern hardware running **CUDA 11.8**. Because this is a shared university server, **we do not have the administrator rights to downgrade or change the server's CUDA version.**

### What is CUDA?

**CUDA** (Compute Unified Device Architecture) is a software layer and programming interface created by NVIDIA.

* **What it does:** It allows deep learning frameworks (like TensorFlow and PyTorch) to communicate directly with the NVIDIA GPU. Instead of doing math slowly one-by-one on the CPU, CUDA tells the GPU to perform thousands of matrix multiplications simultaneously (parallel processing), which is essential for training neural networks.
* **Why it caused an issue:** Every version of TensorFlow is hardcoded to work with a specific version of CUDA. TensorFlow 1.x does not understand CUDA 11.8. If we tried to install TF1.x, the GPU would simply not be recognized.

**Our Solution:** We install **TensorFlow 2.13** (which perfectly matches the server's CUDA 11.8) and utilize its built-in `compat.v1` module to "trick" the modern TensorFlow into executing our legacy TF 1.x code.

---

## Phase 1: Environment Setup & Dependencies

Before installing any packages, we must create an isolated workspace. We use **Conda** for this, which ensures our specific Python version and packages do not interfere with other projects on the server.

### 1. Create and Activate the Conda Environment

Open a terminal in JupyterLab and run the following commands to create a new environment named `acdc-tf2` using Python 3.10 (which is highly stable for this setup):

```bash
# Create the environment with Python 3.10
conda create --name acdc-tf2 python=3.10 -y

# Activate the environment
conda activate acdc-tf2

```

*(Note: You must run `conda activate acdc-tf2` every time you open a new terminal before running your training scripts).*

### 2. Install Core Packages

The original `requirements.txt` contains outdated packages that conflict with modern Python environments and the new TensorFlow version. We must manually install specific versions to stabilize the environment.

Run the following commands in your activated `acdc-tf2` environment:

```bash
# 1. Install TF compatible with the UT server's CUDA 11.8
pip install tensorflow==2.13.0

# 2. Install NumPy 1.x and specific typing-extensions to prevent TF crashes 
# (TF 2.13 will crash if the newer NumPy 2.0 is installed)
pip install numpy==1.24.3 typing-extensions==4.5.0

# 3. Downgrade nibabel to ensure compatibility with older NumPy (used for .nii.gz files)
pip install "nibabel<5.0"

# 4. Install OpenCV (Headless version prevents missing GUI library errors on Linux servers)
pip install opencv-python-headless

# 5. Install the remaining required dependencies
pip install pandas scipy scikit-image matplotlib SimpleITK h5py tqdm networkx

```

*Note: You can safely ignore any `pip` warnings related to PyTorch if they appear, as we are strictly using TensorFlow for this pipeline.*

---

## Phase 2: Codebase Adaptation (TF1 to TF2 Bridge)

Because the codebase uses TF1 syntax (`tf.variable_scope`, `tf.get_variable`), we must force TensorFlow 2 to behave like TensorFlow 1.

### 1. Global Import Replacement

In **every** executable Python file (`train.py`, `model.py`, `unet2D_bn_xent.py`, `system.py`, etc.), you must change the TensorFlow import.

**Find:**

```python
import tensorflow as tf

```

**Replace with:**

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```

### 2. Fixing `layers.py` (Keras Initializers & Batch Norm)

The old `tf.contrib` module was completely removed in TF2. We must replace the old initializers and batch normalization functions with their modern Keras/Compat equivalents.

At the top of `layers.py`, replace the `variance_scaling_initializer` and `xavier_initializer` imports with these drop-in wrapper functions:

```python
# --- Fix for TF1 -> TF2 Keras initializers ---
def variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32):
    distribution = 'uniform' if uniform else 'truncated_normal'
    return tf.keras.initializers.VarianceScaling(scale=factor, mode=mode.lower(), distribution=distribution)

def xavier_initializer(uniform=True, dtype=tf.float32):
    if uniform:
        return tf.keras.initializers.GlorotUniform()
    else:
        return tf.keras.initializers.GlorotNormal()
# -------------------------------------------------

```

Furthermore, update the `batch_normalisation_layer` function in `layers.py` to use the v1 compat module:

```python
def batch_normalisation_layer(bottom, name, training):
    # TF2 Compat fix: replacing tf.contrib.layers.batch_norm
    h_bn = tf.compat.v1.layers.batch_normalization(
        inputs=bottom, momentum=0.99, epsilon=1e-3, training=training,
        name=name, center=True, scale=True
    )
    return h_bn

```

---

## Phase 3: Configuring Your Experiment (The Config File)

The file `unet2D_bn_xent.py` (located in your main project folder `~/Deep_learning_project/acdc_challenge/acdc_segmenter/`) acts as the "control panel" for your experiment. You can duplicate this file to create different experiments.

Inside this file, you can tweak various hyperparameters and swap out core components of the training pipeline.

### How to customize your setup:

* **Adjusting Hyperparameters:**
You can directly change values like `learning_rate = 0.01` to `0.001`, change `loss_type = 'crossentropy'` to `'dice'`, or toggle data augmentations like `do_rotations = True`.
* **Swapping the Network Architecture:**
The config file defines the model via `model_handle = model_zoo.unet2D_bn`.
* *How to find other models:* Open the `model_zoo.py` file in your repository. Look at the functions defined inside (e.g., `unet2D`, `unet3D`). You can change your config to use any of these by updating the handle (e.g., `model_handle = model_zoo.unet2D`).


* **Swapping the Optimizer:**
The config file defines the optimizer via `optimizer_handle = tf.train.AdamOptimizer`.
* *How to find other optimizers:* Because we are using the TF1 compatibility bridge, you can swap this with other standard TF1 optimizers by searching the TensorFlow documentation. For example, you could change it to `tf.train.GradientDescentOptimizer` or `tf.train.RMSPropOptimizer`.

---

## Phase 4: Resource Management & Training

### 1. Fixing the OOM (Out of Memory) Error

When running the U-Net model with `batch_size = 24`, the GPU will run out of VRAM (resulting in a `RESOURCE_EXHAUSTED` error).

**Solution:** Open your configuration file (`unet2D_bn_xent.py`) and lower the batch size so it fits inside the GPU's memory.

```python
# Training settings
batch_size = 16  # Reduced from 24 to prevent OOM
```

---

### 2. Running the Training in the Background (`nohup`)

To prevent the training from aborting when you close your laptop or lose connection to the UT JupyterLab, run the script using `nohup` (no hangup). This ensures the process continues running even if your terminal session ends.

**Start the training in the background:**

```bash
nohup python train.py --config unet2D_bn_xent.py > training_output.log 2>&1 &
```

* This runs the script in the background, redirects all output to `training_output.log`, and appends the process ID to the log.

---

### 3. Monitoring Training Progress

You can monitor the training progress by checking the log file:

```bash
tail -f training_output.log
```

*(Press `Ctrl + C` to exit the `tail` command without stopping the training.)*

---

### 4. Monitoring GPU Usage

To verify that the GPU is actively processing your data and to monitor VRAM usage, open a terminal and run:

```bash
watch -n 1 nvidia-smi
```

---

## Phase 5: Monitoring with TensorBoard

**What is TensorBoard?**
TensorBoard is a built-in visualization tool. It reads the log files generated by our training script and creates live, interactive graphs of our metrics (like Training Loss and Validation Dice scores).

### 1. Fixing the TensorBoard Startup Bug

Modern Python environments often have `setuptools` versions that recently removed the `pkg_resources` module, which TensorBoard 2.13 requires to boot.

**Fix:**

```bash
pip install "setuptools<70.0.0"

```

### 2. Starting TensorBoard

Open a **new** terminal (while your training runs in the background via tmux), activate your environment (`conda activate acdc-tf2`), and start TensorBoard:

```bash
tensorboard --logdir ~/Deep_learning_project/acdc_challenge/acdc_logdir --bind_all

```

*(Ensure you use the correct, case-sensitive path to your log directory).*

### 3. Accessing the Web Interface (UTwente Proxy)

Because we are behind the university's firewall, we cannot simply surf to `localhost:6006`. We must use the Jupyter proxy trick:

1. Look at your current JupyterLab URL. It likely looks like:
`https://jupyter.utwente.nl/user/[your-username]/lab/tree/...`
2. Change the URL to route directly to port 6006 by replacing `/lab/tree/...` with `/proxy/6006/`:
`https://jupyter.utwente.nl/user/[your-username]/proxy/6006/`
3. Hit Enter. You should now see the TensorBoard dashboard tracking your metrics in real-time.

---

## Phase 6: Output Files (Checkpoints & Logs)

As the model trains, it automatically saves files to `~/Deep_learning_project/acdc_challenge/acdc_logdir/unet2D_bn_xent`.

You will find:

* **`events.out.tfevents...`**: The raw metric data read by TensorBoard.
* **`model_best_dice.ckpt`**: The network weights at the epoch where it achieved the highest Dice score on the validation set. **(Use this for your final testing/inference).**
* **`model_best_xent.ckpt`**: The weights where it achieved the lowest Cross-Entropy Loss.
* **`unet2D_bn_xent.py`**: A backup copy of the config file used for this specific run, ensuring total reproducibility of the experiment.
