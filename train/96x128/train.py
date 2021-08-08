import os
os.environ["PYTHONPATH"] = "WILL BE POPULATED UP ./run.sh"

import tensorflow as tf
from tensorflow.keras import Input, callbacks
from jarvis.train import params
from jarvis.utils.general import gpus, overload, tools as jtools

from data import contrastive_generator
from model import prepare_model
from config import INPUT_SHAPE


# --- Autoselect GPU
gpus.autoselect()

def load_data(p):
    # --- Prepare model
    inputs = {
        'pos': Input(shape=INPUT_SHAPE, name='pos'),
        'unk': Input(shape=INPUT_SHAPE, name='unk'),
        'neg': Input(shape=INPUT_SHAPE, name='neg'),
    }

    gen_train = contrastive_generator(p)
    gen_valid = contrastive_generator(p, valid=True)
    
    return gen_train, gen_valid, inputs

def train_model(model, p, steps_per_epoch=100, validation_freq=1):
    # --- Determine total loop iterations needed
    epochs = int(p['iterations'] / steps_per_epoch)

    # --- Prepare Tensorboard 
    log_dir = '{}/jmodels/logdirs/{}'.format(
        os.path.dirname(p['output_dir']),
        os.path.basename(p['output_dir']))

    # --- Prepare CSV path
    csv_log_path = '{}/train_log.csv'.format(p['output_dir'])

    # --- Define training callbacks
    tensorboard_log = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    csv_log = callbacks.CSVLogger(csv_log_path)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # --- Train the model
    history = model.fit(
        x=gen_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=gen_valid,
        validation_steps=steps_per_epoch,
        validation_freq=validation_freq,
        callbacks=[tensorboard_log, csv_log]
    )

    # --- Save model
    MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])
    model.save(MODEL_NAME)
    
    return history
    
if __name__ == "__main__":
    p = params.load('./hyper.csv', row=0)
    gen_train, gen_valid, inputs = load_data(p)
    model = prepare_model(inputs, p, use_cosine_similarity=True)
    history = train_model(model, p)