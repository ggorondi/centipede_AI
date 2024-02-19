import numpy as np
from keras.models import load_model
import os

class GorondiAgent:

    def __init__(self):
        if os.path.exists('agents/gorondi/all_first_models/first_simple_model_630.h5'):
            model = load_model('agents/gorondi/all_first_models/first_simple_model_630.h5')
            model.compile(loss='mse', optimizer='adam')
            self.model=model
            print(f'GorondiAgent imported model')
        else:
            print(f'GorondiAgent couldnt find pre-trained model')                           

    def action(self,state):
        Q = self.model.predict(np.expand_dims(state, axis=0), verbose=0)   
        action = np.argmax(Q)+10  
        return action
    
    def name(self):
        return {'nombre':'Gabor', 'apellido':'Gorondi', 'legajo':33723}