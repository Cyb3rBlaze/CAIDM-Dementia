import os, random, time, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload

# --- Define path of clients
AD_CLIENT_PATH = "/home/mmorelan/proj/dementia/template/contrastive/ctr_v3/client-3d-raw_AD_AV45_only.yml"
CN_CLIENT_PATH = "/home/mmorelan/proj/dementia/template/contrastive/ctr_v3/client-3d-raw_CN_AV45_only.yml"
INPUT_SHAPE = (96, 160, 160, 1)


@overload(Client)
def preprocess(self, arrays, **kwargs):

    # --- Extract pre-calculated whole exam mu/sd and normalize
    arrays["xs"]["dat"] = (arrays["xs"]["dat"] - kwargs["row"]["mu"]) / kwargs["row"][
        "sd"
    ]

    # --- Scale to 0/1 using 5/95 percentiles
    lower = np.percentile(arrays["xs"]["dat"], 1)
    upper = np.percentile(arrays["xs"]["dat"], 99)
    arrays["xs"]["dat"] = arrays["xs"]["dat"].clip(min=lower, max=upper)
    arrays["xs"]["dat"] = (arrays["xs"]["dat"] - lower) / (upper - lower)

    return arrays


# --- Autoselect GPU
gpus.autoselect()

# --- Prepare hyperparams
p = params.load("./hyper.csv", row=0)

MODEL_NAME = "{}/model.hdf5".format(p["output_dir"])

# --- Prepare model
inputs = {
    "pos": Input(shape=INPUT_SHAPE, name="pos"),
    "unk": Input(shape=INPUT_SHAPE, name="unk"),
    "neg": Input(shape=INPUT_SHAPE, name="neg"),
}

gen_train = contrastive_generator()
gen_valid = contrastive_generator(valid=True)

model = prepare_model(inputs, use_cosine_similarity=True)

# --- Set training variables
steps_per_epoch = 100
validation_freq = 1

# --- Determine total loop iterations needed
epochs = int(p["iterations"] / steps_per_epoch)

# --- Prepare Tensorboard
log_dir = "{}/jmodels/logdirs/{}".format(
    os.path.dirname(p["output_dir"]), os.path.basename(p["output_dir"])
)

# --- Prepare Model checkpoint
ckpt_dir = "{}/jmodels/checkpoints/{}".format(
    os.path.dirname(p["output_dir"]), os.path.basename(p["output_dir"])
)

# --- Define training callbacks
tensorboard_log = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
model_checkpoint = callbacks.ModelCheckpoint(log_dir=ckpt_dir)
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lr_decay = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

# --- Train the model
model.fit(
    x=gen_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=gen_valid,
    validation_steps=steps_per_epoch,
    validation_freq=validation_freq,
    callbacks=[tensorboard_log, model_checkpoint],
)

# --- Save model
model.save(MODEL_NAME)
