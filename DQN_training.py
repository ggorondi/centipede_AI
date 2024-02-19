"""
Código que entrenó al agente.
"""

import numpy as np
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense
from keras.regularizers import l2 
from keras import backend as K
import gc #garbage collect
import random   
import os

import gymnasium as gym
from state_extractor import StateExtractor

extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')


if os.path.exists('agents/gorondi/all_first_models/prosigo_training_model.h5'):
    model = load_model('agents/gorondi/all_first_models/prosigo_training_model.h5')
    print("Imported model")
else:
    model = Sequential()                                                                    # El modelo es un NN relativamente pequeño
    model.add(Dense(64, input_shape=(22,), activation='relu',kernel_regularizer=l2(0.1)))   # Input: 11 coordenadas fil,col
    model.add(Dense(32, activation='relu',kernel_regularizer=l2(0.1)))                     
    model.add(Dense(32, activation='relu',kernel_regularizer=l2(0.1)))                     
    model.add(Dense(4, activation='linear',kernel_regularizer=l2(0.1)))                     # Output: 4 acciones: 10 LEFTFIRE, 11 RIGHTFIRE, 12 UPFIRE, 13 DOWNFIRE
    print("Created new model")

model.compile(loss='mse', optimizer='adam')

target_model = clone_model(model)                                   # Esto estabiliza el training al parecer en DQNs especificamente
target_model.set_weights(model.get_weights())

num_episodes = np.arange(0,711)                                     # Cantidad de episodes/epochs de aprendizaje. Llegué a entrenar 710.
experiences = []                                                    # Memoria en cada episode/epoch para minibatches
observetime = 1000                                                  # Cantidad de steps de juego por epoch
epsilons = np.linspace(0.99,0.1,len(num_episodes))                  # Simmulated anhealing
gamma = 0.9                                                         # Discount factor de Bellman Eq
mb_size = 100                                                       # minibatch

kernel_for_trickle = np.array([0,0,0,0,0,1,0.9,0.8,0.65,0.5,0.4])   # Uso esto para estirar (convolucionando) los rewards hacia atras algunos frames porque disparar suele tardar varios frames en recibir su reward. Asumo que agrega linealidad al aprendizaje, más allá de lo que hace la ecuación de Bellman

for episode,epsilon in zip(num_episodes,epsilons):

    # 1) Observation

    states_mem = []
    actions_mem = []
    reward_mem = []
    states_new_mem = []
    terminated_mem = []
    truncated_mem = []

    obs,info = env.reset()        
    state = extractor.extract(obs)
    terminated = False
    truncated = False
    death_timer=0

    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.choice([10, 11, 12, 13])                     # 10 LEFTFIRE, 11 RIGHTFIRE, 12 UPFIRE, 13 DOWNFIRE
        else:
            Q = model.predict(np.expand_dims(state, axis=0), verbose=0)             
            action = np.argmax(Q)+10                                                
        
        obs_new, reward, terminated,truncated, info = env.step(action)
        state_new = extractor.extract(obs_new)
        dead = extractor.extract_extra(obs_new)

        # Pongo reward negativo si recién murió
        death_timer-=1
        if dead and death_timer<0:
            death_timer=30                                                  # frames que no vuelvo a fijarme si recién murió
            reward=-400
        
        states_mem.append(state)
        actions_mem.append(action)
        reward_mem.append(reward)
        states_new_mem.append(state_new)
        terminated_mem.append(terminated)
        truncated_mem.append(truncated)

        state = state_new

        if terminated or truncated:
            obs,info = env.reset()           # Restart si terminó. Siempre junto 1000 steps
            state = extractor.extract(obs)
            death_timer=0


    # Modifico un poco los rewards
    reward_mem = np.array(reward_mem)
    reward_mem = np.flip(reward_mem)
    reward_mem[reward_mem == 5] = 0                                         # No quiero que rewardee estas acciones, son de cuando suma los cuadraditos al morir
    reward_mem = np.convolve(reward_mem, kernel_for_trickle, mode='same')   # Las balas tardan en llegar, y pq los rewards son demasiado sparse, estiro rewards hacia atras manualmente
    reward_mem = np.flip(reward_mem)
    reward_mem = list(reward_mem)
    

    # 2) Learning (Experience/memory replay)

    experiences = list(zip(states_mem,actions_mem,reward_mem,states_new_mem,terminated_mem,truncated_mem))

    for i in range(4):                                          # 4 minibatches
 
        minibatch = random.sample(experiences, mb_size)                        

        inputs = np.zeros((mb_size, 22)) 
        targets = np.zeros((mb_size, 4))

        for i in range(0, mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            terminated = minibatch[i][4]
            truncated = minibatch[i][5]
            

            inputs[i] = state 
            targets[i] = model.predict(np.expand_dims(state, axis=0), verbose=0)
            Q_sa = target_model.predict(np.expand_dims(state_new, axis=0), verbose=0)
            
            if terminated or truncated:
                targets[i, action-10] = reward
            else:
                targets[i, action-10] = reward + gamma * np.max(Q_sa)               # Bellman Eq
        
        model.train_on_batch(inputs, targets)

    K.clear_session()   # Keras va acumulando, si no hago esto se me explota el RAM en unas horas
    gc.collect()        # Por las dudas


    if episode % 5 == 0:
        target_model.set_weights(model.get_weights())                                   # Cada tanto update el target (clone) network


    if episode % 5 == 0:                                                                # Aca voy guardando los modelos
        model.save(f'agents/gorondi/all_first_models/first_simple_model_{episode}.h5')
        # print(f'Model saved at episode {episode}')


    # 3) Por ultimo, voy testeando que tan bien juega el modelo 
    if episode % 5 == 0:
        
        obs,info = env.reset()
        state = extractor.extract(obs)
        done = False
        tot_reward = 0.0
        while not (terminated or truncated):
            Q = model.predict(np.expand_dims(state, axis=0), verbose=0)        
            action = np.argmax(Q)+10        
            obs, reward, terminated, truncated, info = env.step(action)
            state = extractor.extract(obs)  
            tot_reward += reward
        print(f'Episode: {episode}, Total Reward: {tot_reward}')