import os, numpy as np, pandas as pd
import tensorflow.keras.backend as K
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, datasets, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload, tools as jtools

# --- Overload jarvis client
@overload(Client)
def preprocess(self, arrays, **kwargs):
    """
    Method to create a custom msk array for class weights and/or masks
    """
    
    # Binarize data into two classes
    arrays['ys']['lbl-anc'] = arrays['ys']['lbl-anc'] >= 1
    arrays['ys']['lbl-neg'] = arrays['ys']['lbl-neg'] >= 1
    arrays['ys']['lbl-pos'] = arrays['ys']['lbl-pos'] >= 1
    
    return arrays

def euclidean_distance(inputs):
	(x1, x2) = inputs
	sumSquared = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def subtract(inputs):
	(x1, x2) = inputs
	return x1-x2

def prepare_client(paths, p):

    client = Client(CLIENT_TRAINING if os.path.exists(CLIENT_TRAINING) else CLIENT_TEMPLATE, configs={
        'batch': {
            'size': p['batch_size'],
            'fold': p['fold']}})

    return client

def prepare_model(inputs):

    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_initializer': 'he_uniform'}

    # --- Add kernel regularizer if set
    if p['kernel_regularizer'] != 0:
        kwargs['kernel_regularizer'] = regularizers.l2(p['kernel_regularizer'])

    # --- Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    elu = lambda x : layers.ELU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    drop = lambda x : layers.Dropout(rate=p['dropout'])(x)

    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : norm(elu(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : norm(elu(conv(x, filters, strides=(2, 2, 2))))
    tran2 = lambda filters, x : norm(elu(tran(x, filters, strides=(2, 2, 2))))

    # --- Extract alpha value
    a = p['alpha']
    
    # --- Define dummy anchor model
    a1 = conv1(int(a*8), inputs['anc'])
    a2 = conv1(int(a*16), conv2(int(a*16), a1))
    a3 = tran2(int(a*8), a2)
    a4 = conv1(int(a*8), conv1(int(a*8), a3))
    
    encoder_unknown = a2
    decoder_unknown = a4
    
    # --- Define dummy control model
    n1 = conv1(int(a*8), inputs['neg'])
    n2 = conv1(int(a*16), conv2(int(a*16), n1))
    n3 = tran2(int(a*8), n2)
    n4 = conv1(int(a*8), conv1(int(a*8), n3))
    
    encoder_cn = n2
    decoder_cn = n4
    
    # --- Define dummy AD model
    p1 = conv1(int(a*8), inputs['pos'])
    p2 = conv1(int(a*16), conv2(int(a*16), p1))
    p3 = tran2(int(a*8), p2)
    p4 = conv1(int(a*8), conv1(int(a*8), p3))
    
    encoder_ad = p2
    decoder_ad = p4
    
    # --- Create logits
    logits = {}
    logits['lbl-anc'] = layers.Conv3D(filters=2, name='lbl-anc', kernel_size=(1, 1, 1), padding='same')(a4)
    logits['lbl-neg'] = layers.Conv3D(filters=2, name='lbl-neg', kernel_size=(1, 1, 1), padding='same')(n4)
    logits['lbl-pos'] = layers.Conv3D(filters=2, name='lbl-pos', kernel_size=(1, 1, 1), padding='same')(p4)
    
    # --- Compute Euclidean distance
    ad_unknown = layers.Lambda(euclidean_distance)([encoder_ad, encoder_unknown])
    cn_unknown = layers.Lambda(euclidean_distance)([encoder_cn, encoder_unknown])
    
    final_difference = layers.Lambda(subtract)([ad_unknown, cn_unknown])
    flatten_difference = layers.Flatten()(final_difference)
    final_activation = layers.Dense(1, name="final_activation", activation="sigmoid")(flatten_difference)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits)

    # --- Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']),
        loss={
            'lbl-anc': losses.SparseCategoricalCrossentropy(from_logits=True),
            'lbl-neg': losses.SparseCategoricalCrossentropy(from_logits=True),
            'lbl-pos': losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        metrics={
            'lbl-anc': custom.dsc(cls=1),
            'lbl-neg': custom.dsc(cls=1),
            'lbl-pos': custom.dsc(cls=1)
        },
        experimental_run_tf_function=False
    )

    return model

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
paths = jtools.get_paths('ct/kidney')

# --- Prepare hyperparams
p = params.load('~/projects/dementia_test/hyper.csv', row=0)

# --- Set constants
CLIENT_TEMPLATE = '/home/mmorelan/projects/dementia_test/yml/client-dementia-3d.yml'
CLIENT_TRAINING = '{}/client.yml'.format(p['output_dir'])
MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- Prepare client
client = prepare_client(paths, p)
# client.load_data_in_memory()
gen_train, gen_valid = client.create_generators()

# --- Prepare model
model = prepare_model(client.get_inputs(Input))
model = load_existing(model, p)

# =======================================================
# TRAINING LOOP 
# =======================================================

# --- Assume a 400:100 ratio of train:valid
steps_per_epoch = 50
validation_freq = 1

# --- Determine total loop iterations needed
epochs = int(p['iterations'] / steps_per_epoch)

# --- Prepare Tensorboard
log_dir = '{}/jmodels/logdirs/{}'.format(
    os.path.dirname(p['output_dir']),
    os.path.basename(p['output_dir']))

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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
