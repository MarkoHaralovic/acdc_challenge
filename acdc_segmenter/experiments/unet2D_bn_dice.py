import model_zoo
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# AANGEPAST: Nieuwe experiment naam voor TensorBoard
experiment_name = 'unet2D_bn_dice'

# Model settings
model_handle = model_zoo.unet2D_bn

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
nlabels = 4

# Training settings
batch_size = 16  # Gehouden op 16 om OOM errors te voorkomen
learning_rate = 0.01
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None

# AANGEPAST: De loss functie is nu Dice om de class imbalance aan te pakken
loss_type = 'dice'  # crossentropy/weighted_crossentropy/dice/dice_onlyfg

# Augmentation settings
# Staan nog uit, we veranderen 1 variabele per keer voor zuiver wetenschappelijk vergelijk
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False

# Gehouden op 100, aangezien je baseline na 60 epochs al afvlakte
max_epochs = 100
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100  # reduced to 100 for early stopping