import os, numpy as np, pandas as pd
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, datasets, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload, tools as jtools

def classification_generator(valid=False):
    client = prepare_client(paths, p)
    gen_train, gen_valid = client.create_generators()
    
    while True:
        if valid:
            xs_final, ys_final = next(gen_valid)
        else:
            xs_final, ys_final = next(gen_train)

        xs = {}
        ys = {}
        
        xs['dat'] = xs_final['dat']
        ys['lbl'] = ys_final['lbl'].reshape((p['batch_size'], 1))
            
        yield xs, ys
        
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

def prepare_model(inputs):

    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_initializer': 'he_uniform'}

    # --- Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    elu = lambda x : layers.ELU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    
    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : norm(elu(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : norm(elu(conv(x, filters, strides=(2, 2, 2))))
    tran2 = lambda filters, x : norm(elu(tran(x, filters, strides=(2, 2, 2))))
    
    # --- Define contracting layers
    l1 = conv1(16, inputs['dat'])
    l2 = conv1(24, conv2(24, l1))
    l3 = conv1(32, conv2(32, l2))
    l4 = conv1(48, conv2(48, l3))
    l5 = layers.Flatten()(l4)
    l6 = layers.Dense(64, activation='relu')(l5)
    l7 = layers.Dense(8, activation='relu')(l6)
    
    # --- Create logits
    logits = {}
    logits['lbl'] = layers.Dense(1, name='lbl')(l7)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits)

    # --- Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']), 
        loss={'lbl': losses.BinaryCrossentropy(from_logits=True)},
        metrics={'lbl': metrics.BinaryAccuracy()},
        experimental_run_tf_function=False
    )

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
CLIENT_TEMPLATE = '/home/mmorelan/proj/dementia/yml/client-3d-96x128.yml'
CLIENT_TRAINING = '{}/client.yml'.format(p['output_dir'])
MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- Prepare client
client = prepare_client(paths, p)
# client.load_data_in_memory()
gen_train = classification_generator()
gen_valid = classification_generator(valid=True)

# --- Prepare model
model = prepare_model(client.get_inputs(Input))
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

# --- Train
model.fit(
    x=gen_train, 
    epochs=epochs,
    steps_per_epoch=steps_per_epoch, 
    validation_data=gen_valid,
    validation_steps=steps_per_epoch,
    validation_freq=validation_freq
)

# --- Save model
model.save(MODEL_NAME)

# --- Save client
client.to_yml(CLIENT_TRAINING)