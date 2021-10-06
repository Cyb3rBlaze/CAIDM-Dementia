from jarvis.train import params
from jarvis.train.client import Client

def custom_generator(valid=False):
    """
    Generator that yields an array instead of a dictionary. Useful for using TensorFlow's built-in class weights on Model.fit().
    """
    
    # --- Create generators for AD/CN
    client = Client(CLIENT_TEMPLATE, configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}})
    
    gen_train, gen_valid = client.create_generators()
    
    while True:
        if valid:
            xs, ys = next(gen_valid)
        else:
            xs, ys = next(gen_train)
            
        yield xs['dat'], ys['lbl']