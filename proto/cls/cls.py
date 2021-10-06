import os, numpy as np, pandas as pd
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, datasets, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload, tools as jtools

import arch

def prepare_client(paths, p):

    client = Client(CLIENT_TRAINING if os.path.exists(CLIENT_TRAINING) else CLIENT_TEMPLATE, configs={
        'batch': {
            'size': p['batch_size'],
            'fold': p['fold']}})

    return client

def load_existing(model, p):

    # --- Create output_dir
    os.makedirs(p['output_dir'], exist_ok=True)

    # --- Load existing model if present
    if os.path.exists(MODEL_NAME):
        print('Loading existing model weights: {}'.format(MODEL_NAME))
        model.load_weights(MODEL_NAME)

    return model

# =======================================================
# PREPARATION 
# =======================================================

# --- Autoselect GPU
gpus.autoselect()

# --- Look up path
paths = jtools.get_paths('adni')

# --- Prepare hyperparams
p = params.load('./hyper.csv', row=0)

# --- Set constants
CLIENT_TEMPLATE = '/home/mmorelan/proj/dementia/yml/client-3d-raw-mod.yml'
CLIENT_TRAINING = '{}/client.yml'.format(p['output_dir'])
MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- Prepare client
client = prepare_client(paths, p)
# client.load_data_in_memory()
gen_train, gen_valid = client.create_generators()

# --- Prepare model
model = arch.cls_v01(client.get_inputs(Input))
model = load_existing(model, p)

# =======================================================
# TRAINING LOOP 
# =======================================================

# --- Assume a 400:100 ratio of train:valid
steps_per_epoch = 100
validation_freq = 1

# --- Determine total loop iterations needed
epochs = int(p['iterations'] / steps_per_epoch)

# --- Prepare Tensorboard 
log_dir = '{}/jmodels/logdirs/{}'.format(
    os.path.dirname(p['output_dir']),
    os.path.basename(p['output_dir']))

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# --- Train
model.fit(
    x=gen_train, 
    epochs=epochs,
    steps_per_epoch=steps_per_epoch, 
    validation_data=gen_valid,
    validation_steps=steps_per_epoch,
    validation_freq=validation_freq,
    callbacks=[tensorboard_callback]
)

# --- Save model
model.save(MODEL_NAME)

# --- Save client
client.to_yml(CLIENT_TRAINING)