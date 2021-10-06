from tensorflow.keras import Input, Model, layers

def cls_v01(inputs, p):
    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_initializer': 'he_uniform'}

    # Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    elu = lambda x : layers.ELU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    pool = lambda x : layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    
    # Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : norm(elu(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : norm(elu(conv(x, filters, strides=(2, 2, 2))))
    tran2 = lambda filters, x : norm(elu(tran(x, filters, strides=(2, 2, 2))))

    # Extract alpha value
    a = p['alpha']
    
    # Define contracting layers
    l1 = conv1(int(a*8), inputs['dat'])
    l2 = pool(conv1(int(a*16), conv2(int(a*16), l1)))
    l3 = conv1(int(a*32), conv2(int(a*32), l2))
    l4 = pool(conv1(int(a*48), conv2(int(a*48), l3)))
    l5 = layers.Flatten()(l4)
    l6 = layers.Dense(256, activation='relu')(l5)
    l7 = layers.Dense(64, activation='relu')(l6)
    l8 = layers.Dense(8, activation='relu')(l7)
    
    # Create logits
    logits = {}
    logits = layers.Dense(1, activation='sigmoid', name='lbl')(l8)

    # Create model
    model = Model(inputs=inputs, outputs=logits)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']), 
        loss=losses.BinaryCrossentropy(),
        metrics=metrics.BinaryAccuracy(),
        experimental_run_tf_function=False
    )

    return model


def cls_v02(inputs, p):

    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_initializer': 'he_uniform'}

    # Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    elu = lambda x : layers.ELU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    pool = lambda x : layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    drop = lambda x, drop_rate : layers.Dropout(rate=drop_rate)(x)
    
    # Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : norm(elu(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : norm(elu(conv(x, filters, strides=(2, 2, 2))))
    tran2 = lambda filters, x : norm(elu(tran(x, filters, strides=(2, 2, 2))))

    # Encoder
    l1 = drop(conv1(4, inputs['dat']), p['drop_rate'])
    l2 = drop(conv1(8, conv2(8, l1)), p['drop_rate'])
    l3 = drop(conv1(16, conv2(16, l2)), p['drop_rate'])
    l4 = drop(conv1(32, conv2(32, l3)), p['drop_rate'])
    l5 = layers.Conv3D(filters=4, kernel_size=(1, 1, 1))(l4)
    l6 = layers.Flatten()(l5)
    l7 = layers.Dense(10, activation="relu")(l6)
    
    # Create logits
    logits = {}
    logits = layers.Dense(1, activation="sigmoid", name="lbl")(l7)

    # Create model
    model = Model(inputs=inputs, outputs=logits)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']), 
        loss=losses.BinaryCrossentropy(),
        metrics=metrics.BinaryAccuracy(),
        experimental_run_tf_function=False
    )

    return model

def choi_v01(inputs, p):
    """
    Title: Predicting cognitive decline with deep learning of brain metabolism and amyloid imaging
    Author(s): Hongyoon Choi, Kyong Hwan Jin
    URL: https://pubmed.ncbi.nlm.nih.gov/29454006/
    """

    # --- Define contracting layers
    l0 = layers.ZeroPadding3D(padding=(32, 0, 0))(inputs['dat'])
    l1 = layers.Conv3D(filters=64, kernel_size=(7, 7, 7), strides=4, activation='relu', padding='valid', kernel_initializer='he_uniform')(l0)
    l2 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l1)
    l3 = layers.Conv3D(filters=128, kernel_size=(7, 7, 7), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l2)
    l4 = layers.MaxPool3D(pool_size=(3, 3, 3,), strides=2, padding='valid')(l3)
    l5 = layers.Conv3D(filters=512, kernel_size=(6, 6, 6), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l4)
    
    
    # --- Create logits
    logits = {}
    logits['lbl'] = layers.Conv3D(filters=1, name='lbl', kernel_size=(1, 1, 1), padding='valid')(l5)

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

def choi_weights_v01(inputs, p):  
    # --- Define contracting layers
    l0 = layers.ZeroPadding3D(padding=(32, 0, 0))(inputs['dat'])
    l1 = layers.Conv3D(filters=64, kernel_size=(7, 7, 7), strides=4, activation='relu', padding='valid', kernel_initializer='he_uniform')(l0)
    l2 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l1)
    l3 = layers.Conv3D(filters=128, kernel_size=(7, 7, 7), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l2)
    l4 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l3)
    l5 = layers.Conv3D(filters=512, kernel_size=(6, 6, 6), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l4)
    
    # --- Create logits
    logits = layers.Reshape(target_shape=(1,))(layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', padding='valid')(l5))

    # --- Create model
    model = Model(inputs=inputs, outputs=logits)

    # --- Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']), 
        loss=losses.BinaryCrossentropy(),
        metrics=metrics.BinaryAccuracy(),
        experimental_run_tf_function=False
    )

    return model

def choi_weights_v02(inputs, p):    
    # --- Define contracting layers
    l1 = layers.Conv3D(filters=32, kernel_size=(7, 7, 7), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(inputs['dat'])
    l2 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l1)
    l3 = layers.Conv3D(filters=64, kernel_size=(5, 5, 5), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l2)
    l4 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l3)
    l5 = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l4)
    l6 = layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='valid')(l5)
    l7 = layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=1, activation='relu', padding='valid', kernel_initializer='he_uniform')(l6)
    l8 = layers.Conv3D(filters=8, kernel_size=(1, 1, 1), strides=2, activation='relu', padding='valid', kernel_initializer='he_uniform')(l7)
    
    l9 = layers.Flatten()(l8)
    
    # --- Create logits
    logits = layers.Dense(1, activation='sigmoid')(l9)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits)

    # --- Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p['LR']), 
        loss=losses.BinaryCrossentropy(),
        metrics=metrics.BinaryAccuracy(),
        experimental_run_tf_function=False
    )

    return model