#Program implementing neural network

#imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from logic_2048 import *
import random

#initialize the values
#lmda = 0.9
gma = 0.90
al = 0.21


def save_model(model): 
    #returning a representation of the model as a JSON string. 
    #the representation does not include the weights, only the architecture
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)
    #saves the weights of the model as a HDF5 file.
    model.save_weights('weights.h5', overwrite=True)

def load_model():
    model = model_from_json(open('model.json').read())
    #loads the weights of the model from a HDF5 file (created by save_weights)
    model.load_weights('weights.h5')
    model.compile(optimizer='RMSprop', loss = 'mean_squared_error')
    return model

#Define the neural network
def nnBlock(input_size,f):
    #define the layers
    A_input = keras.Input(input_size)
    A_flat = keras.layers.Flatten()(A_input)
    A = keras.layers.Dense(128,activation='relu')(A_flat)
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dropout(rate = 0.125)(A)
    A = keras.layers.Dense(32)(A) 
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dense(8)(A) 
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dense(1)(A)
    
    model = keras.Model(inputs = A_input,outputs = A)
    return model

#uncomment these for the first time to train the weights

#model = nnBlock((4,4,1),4)
#model.summary()
#RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#model.compile(optimizer=RMSprop,loss='mean_squared_error',metrics=['accuracy'])
#tb = TensorBoard(log_dir = '/Users/Ayush/2048_temp')
 
def do_greedy(ast,model):
    global al
    global gma
    a_scaled = np.log(ast+1)
    #Generates output predictions for the input sample
    pred1 = model.predict(a_scaled.reshape(1,4,4,1))[0] 
    lb = check(ast)
    lb_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb.items())))
    maxval = -128
    maxr = -8
    if len(lb_ins) == 0:
        return pred1
    for ins in lb_ins:
		#perform the action, get the next state and reward
        a_next,rew = do(ast.copy(),ins) 
        #get the next state by generating number 2 at random position through process2 function
        a_next = process2(a_next)
        a_scaled = np.log(a_next+1)
        #Generates output predictions for the input sample
        pred2 = model.predict(a_scaled.reshape(1,4,4,1))[0]
        if (pred2 > maxval and (rew>0 or (maxr<0 and rew<0))) or (maxval<-8) or (maxr<0 and rew>0):
            maxval = pred2
			#scale the reward
            maxr = np.sign(rew)*np.log10(abs(rew)+1)
			#al = alpha, gma = gamma
    return (1-al)*pred1 + al*(maxr+gma*maxval) 


def action(model,a,ep):    
    global al
    global gma   
    l = check(a)
    l_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,l.items())))
    if len(l_ins) == 0:
        return -1,{}  #flag = -1
    rand = np.random.rand()
    pred0 = model.predict(np.log(a+1).reshape(1,4,4,1))[0]
    cdict = {}
    cdict['val'] = -8
    cdict['rew'] = -8
    
    if rand >= ep:
        for ins in l_ins:
            a_next,rew = do(a.copy(),ins)
            a_next = process2(a_next)
            
            a_scaled = np.log(a_next+1)
            pred1 = model.predict(a_scaled.reshape(1,4,4,1))[0]
			#Apply the bellman equation
			#gma is a discount factor, pred1 + alpha*(r+gma*vst2 - pred1)
            newval = pred1 + al*(-1 + gma*pred1 - pred1) 
            lb1 = check(a_next)
            lb_ins1 = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb1.items())))
            if len(lb_ins1) != 0:
                maxval2 = -128
                R2 = -10
                for ins1 in lb_ins1:
                    a_next2,rew2 = do(a_next.copy(),ins1)
                    a_next2 = process2(a_next2)
                    val2 = do_greedy(a_next2.copy(),model)
                    
                    if (val2 > maxval2 and rew2>0) or (maxval2<-8) or (R2<0 and rew2>0):
                        maxval2 = val2
                        R2 = rew2
                newval = pred1 + al*(np.sign(R2)*np.log10(abs(R2)+1) + gma*maxval2 - pred1)
                cdict['a'+ins+'next'] = maxval2
                cdict['a'+ins+'nextr'] = np.sign(R2)*np.log10(abs(R2)+1)
                cdict['a'+ins] = newval
                cdict['a'+ins+'r'] = np.sign(rew)*np.log10(abs(rew)+1)
               
            if (newval > cdict['val'] and (rew>0 or (cdict['rew']<0 and rew<0))) or (cdict['val']<-8) or (cdict['rew']<0 and rew>0) :
                cdict['ins'] = ins
                cdict['val'] = newval
                cdict['rew'] = np.sign(rew)*np.log10(abs(rew)+1)
        
        newval0 = (1-al)*pred0 + al*(cdict['rew']+gma*cdict['val'])
        model.fit(np.log(a+1).reshape(1,4,4,1),newval0,epochs = 2)
        return 4,cdict['ins']

    else:
        ins = random.choice(l_ins)
        a_new,rew = do(a.copy(),ins)
        a_new = process2(a_new)
        a_scaled = np.log(a_new+1)
        a_res = np.reshape(a_scaled,(1,4,4,1))
        pred = model.predict(a_res)[0]
        newval = pred
        print('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-')
        lb = check(a_new)
        lb_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb.items())))
        maxval = -128
        if len(lb_ins) != 0:
            for ins1 in lb_ins:
                a_new1,rew1 = do(a_new.copy(),ins1)
                a_new1 = process2(a_new1)
                pred1 = do_greedy(a_new1,model)
            newval = pred + al*(np.sign(rew1)*np.log10(abs(rew1)+1)+gma*pred1 - pred)
        cdict['ins'] = ins
        cdict['val'] = newval
        cdict['rew'] = np.sign(rew)*np.log10(abs(rew)+1)
        newval0 = (1-al)*pred + al*(cdict['rew']+gma*cdict['val'])
        model.fit(np.log(a+1).reshape(1,4,4,1),newval0,epochs = 2)
        return 4,ins

def train(n,model,alpha,ep,gamma):
    global al
    global gma
    al = alpha
    gma = gamma
    maxlist = []
    #experience = []  # for experience replay
    ep1 = ep
    for i in range(n):
        a = np.zeros((4,4))
        a = process2(a)  #add a new number randomly
        loop = True
        ep = ep1
        while loop == True:
            print('a\n',a)
            flag,cdins  = action(model,a.copy(),ep)
			#epoch decay
            ep = ep*0.99   
            if flag > 0:
                #calling do function to get the next state and reward
                a,_ = do(a.copy(),cdins)
                #calling function to generate number randomly in the new state
                a = process2(a)
                
            else:
                val = model.predict(np.log(a+1).reshape(1,4,4,1))[0]
                print('a\n',a,val)
                maxlist.append((a.max(),val,a.argmax()))
                
                val2 = model.predict(np.log(a+1).reshape(1,4,4,1))
                newval2 = (1-alpha)*val2 + alpha*(-1 + gma*val2)
                model.fit(np.log(a+1).reshape(1,4,4,1),newval2,epochs = 3)
    
                loop = False
    return maxlist





#Uncomment it for the first time to train and store the weights
#num_of_games = 1000
#m = train(num_of_games,model,0.2,0.99,0.81)
#print (model)
#print (type(model))
#save_model(model)

    
    
    
    