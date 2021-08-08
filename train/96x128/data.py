import random, time, numpy as np
from jarvis.train.client import Client

from config import AD_CLIENT_PATH, CN_CLIENT_PATH


def contrastive_generator(p, valid=False):
    # --- Create generators for AD/CN 
    client_AD = Client(AD_CLIENT_PATH, configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}})
    client_CN = Client(CN_CLIENT_PATH, configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}})
    
    gen_train_AD, gen_valid_AD = client_AD.create_generators()
    gen_train_CN, gen_valid_CN = client_CN.create_generators()
    
    while True:
        if valid:
            xs_AD, ys_AD = next(gen_valid_AD)
            xs_CN, ys_CN = next(gen_valid_CN)
        else:
            xs_AD, ys_AD = next(gen_train_AD)
            xs_CN, ys_CN = next(gen_train_CN)
        
        # --- Randomize for AD-AD-CN or AD-CN-CN
        choice_index = random.randint(0, 1)
        
        if choice_index == 0:
            xs_final = np.concatenate((xs_AD['dat'], xs_CN['dat'][:1]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'], ys_CN['lbl'][:1]), axis=0)
        else:
            xs_final = np.concatenate((xs_AD['dat'][:1], xs_CN['dat']), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:1], ys_CN['lbl']), axis=0)

        xs = {}
        ys = {}
        
        xs['pos'] = np.expand_dims(xs_final[0], axis=0)
        xs['unk'] = np.expand_dims(xs_final[1], axis=0)
        xs['neg'] = np.expand_dims(xs_final[2], axis=0)
        ys['enc1'] = ys_final[0].reshape((1))
        ys['enc2'] = ys_final[1].reshape((1))
        ys['enc3'] = ys_final[2].reshape((1))
        ys['dec1'] = np.expand_dims(xs_final[0], axis=0)
        ys['dec2'] = np.expand_dims(xs_final[1], axis=0)
        ys['dec3'] = np.expand_dims(xs_final[2], axis=0)
        ys['ctr1'] = ys['enc1'] == ys['enc2']
        ys['ctr2'] = ys['enc2'] == ys['enc3']
            
        yield xs, ys