import time
import numpy as np
import nerons as n
import random as r
import copy
from numba import njit, jit
from numba.typed import Dict


class Cache:
    def __init__(self, maxsize = 5):

        self.keys = []
        self.items = [] 
        self.rel = []
        self.maxsize = maxsize


    def cacheItem(self, key, item):

        if key in self.keys:
            self.rel = [i-1 for i in self.rel]
            self.rel[self.keys.index(key)] += 3

        elif len(self.keys) == self.maxsize:

            self.rel = [i-1 for i in self.rel]

            i = self.rel.index(min(self.rel))
            
            self.rel[i] = self.maxsize
            self.items[i] = item
            self.keys[i] = key
        
        else:

            self.rel.append(self.maxsize)
            self.items.append(item)
            self.keys.append(key)

        

    def retrive(self, key):

        if key in self.keys:

            i = self.keys.index(key)

            return self.items[i]
        
        else:

            return None


        
        

    

class network:
    def __init__(self, inputs, outputs, HLC, HLNC, id):

        self.inP = inputs
        self.outP = outputs
        self.HLC = HLC
        self.HLNC = HLNC
        self.id = id
        self.score = 0
        self.nerons = self.makeNerons()
        self.cache = Cache()
        self.hasHiddenLayers = not (self.HLC == 0 or self.HLNC == 0)


        self.inputWires = [t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[0]]]]
        self.inputWires = np.array(self.inputWires)

        self.outputWires = [t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[-2]]]]
        self.outputWires = np.array(self.outputWires)

        
        self.wireValues = []
        self.neronValues = []

        for i in self.nerons[1:-2]:
            self.wireValues.append(np.array([t for t in [o for o in [floatifyList(np.array(u.wires)) for u in i]]]))
            self.neronValues.append(np.rot90(np.array([[o.value/1 for o in i], [o.bias/1 for o in i]])))


        if self.hasHiddenLayers:
            self.wireValues.append(np.array([t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[-3]]]]))
    
        else:
            self.wireValues = np.zeros((2,2,2), dtype=float)
  
        self.inputNerons = [[o.value/1 for o in self.nerons[0]], [o.bias/1 for o in self.nerons[0]]]
        self.outputNerons = [[o.value/1 for o in self.nerons[-1]], [o.bias/1 for o in self.nerons[-1]]]

        self.neronValues.append([[o.value/1 for o in self.nerons[-2]], [o.bias/1 for o in self.nerons[-2]]])

        self.neronValues.append(np.rot90(np.array([[o.value/1 for o in self.nerons[-2]], [o.bias/1 for o in self.nerons[-2]]])))
        self.inputNerons = np.rot90(np.array(self.inputNerons))
        self.outputNerons = np.rot90(np.array(self.outputNerons))

        self.biasValues = self.neronValues[1]
        self.neronValues = self.neronValues[0]

        self.inputNeronValues = np.array([i[0] for i in self.inputNerons])
        self.inputbiasValues = np.array([i[1] for i in self.inputNerons])

        self.outputNeronValues = np.array([i[0] for i in self.outputNerons])
        self.outputbiasValues = np.array([i[1] for i in self.outputNerons])

        self.neronValues = np.rot90(self.neronValues)
        self.biasValues = np.array(self.biasValues)

    def makeNerons(self):
        '''
        uses the paramiters defined in the __init__ function to create nerons. Runs on start
        '''
        x=[copy.deepcopy(self.inP)]

        for a in range(self.HLC):
             x.append(list(range(self.HLNC)))

        x.append(copy.deepcopy(self.outP))

        for i in range(len(self.outP)):
                x[-1][i] = n.neron(-1,0.0)

        for i in range(self.HLC):
            for o in range(self.HLNC):
                x[self.HLC-i][o] = n.neron(x[self.HLC-i+1],0.0)

        for i in range(len(self.inP)):
                x[0][i] = n.neron(x[1],0.0)
        #print(x)
        return x
    

    def getOutputs(self, inputs):
        

        rc = self.cache.retrive(inputs)

        if rc != None:

            return numbaDict(rc)

        
        k = speedyOutputs(inputs, self.HLC, self.HLNC, self.neronValues, self.wireValues, self.outP, self.inputNerons, self.outputNerons, self.inputWires, self.outputWires, self.hasHiddenLayers, self.biasValues, self.inputNeronValues, self.inputbiasValues, self.outputNeronValues, self.outputbiasValues)
        
        #print(k)
        '''
        Gives an output based on inputs
        '''
        z=dict()
        for key, value in k.items():

            z[key] = value

        self.cache.cacheItem(inputs, z)


        return k
    
    def recount(self):
        self.hasHiddenLayers = not (self.HLC == 0 or self.HLNC == 0)


        self.inputWires = [t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[0]]]]
        self.inputWires = np.array(self.inputWires)

        self.outputWires = [t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[-2]]]]
        self.outputWires = np.array(self.outputWires)

        
        self.wireValues = []
        self.neronValues = []

        for i in self.nerons[1:-2]:
            self.wireValues.append(np.array([t for t in [o for o in [floatifyList(np.array(u.wires)) for u in i]]]))
            self.neronValues.append(np.rot90(np.array([[o.value/1 for o in i], [o.bias/1 for o in i]])))


        if self.hasHiddenLayers:
            self.wireValues.append(np.array([t for t in [o for o in [floatifyList(np.array(u.wires)) for u in self.nerons[-3]]]]))
    
        else:
            self.wireValues = np.zeros((2,2,2), dtype=float)
  
        self.inputNerons = [[o.value/1 for o in self.nerons[0]], [o.bias/1 for o in self.nerons[0]]]
        self.outputNerons = [[o.value/1 for o in self.nerons[-1]], [o.bias/1 for o in self.nerons[-1]]]

        self.neronValues.append([[o.value/1 for o in self.nerons[-2]], [o.bias/1 for o in self.nerons[-2]]])

        self.neronValues.append(np.rot90(np.array([[o.value/1 for o in self.nerons[-2]], [o.bias/1 for o in self.nerons[-2]]])))
        self.inputNerons = np.rot90(np.array(self.inputNerons))
        self.outputNerons = np.rot90(np.array(self.outputNerons))

        self.biasValues = self.neronValues[1]
        self.neronValues = self.neronValues[0]

        self.inputNeronValues = np.array([i[0] for i in self.inputNerons])
        self.inputbiasValues = np.array([i[1] for i in self.inputNerons])

        self.outputNeronValues = np.array([i[0] for i in self.outputNerons])
        self.outputbiasValues = np.array([i[1] for i in self.outputNerons])

        self.neronValues = np.rot90(self.neronValues)
        self.biasValues = np.array(self.biasValues)



    def mutate(self):
        '''
        returns a copy of itself with some of the nerons wires and biases tweaked
        '''
    
        nself = copy.deepcopy(self)
        
        for i in range(r.randint(1,5)):

            a = r.randrange(0,len(self.nerons))
            b = r.randrange(0,len(self.nerons[a]))

            nself.nerons[a][b] = nself.nerons[a][b].mutate()
        

        nself.recount()

        return nself


@njit(cache=True, fastmath = True, parallel = True)
def multipyWires(inputWires, nextNeronsValue, NeronValue, NeronBias):

    for i in range(len(inputWires)):

        for o in range(len(inputWires[i])):
    
            nextNeronsValue[o] += (NeronValue[i] * inputWires[i][o]) + NeronBias[i]

    return nextNeronsValue



@njit(cache=True, fastmath = True)
def floatifyList(list):

    return [i/1 for i in list]




#@njit(cache=True, fastmath = True)
def speedyOutputs(inputs, HLC, HLNC, neronValues, wireValues, outP, inputNerons, outputNerons, inputWires, outputWires, hasHiddenLayers, biasValues, inputNeronValues, inputbiasValues, outputNeronValues, outputbiasValues):

    for i in range(len(inputs)):
        inputNeronValues[i] = inputs[i]


    if hasHiddenLayers:


        neronValues[0] = np.atleast_2d(np.add(inputNeronValues, inputbiasValues)) @ np.ascontiguousarray(inputWires)


        for i in range(len(neronValues)-1):
            
            neronValues[i+1] = np.atleast_2d(np.add(neronValues[i], biasValues[i])) @ wireValues[i]
    
        outputNeronValues = np.add(np.atleast_2d(np.add(neronValues[-1], biasValues[-1])) @ outputWires, outputbiasValues)

    
    else:

        outputNeronValues = np.add(np.atleast_2d(np.add(inputbiasValues, inputNeronValues)) @ inputWires, outputbiasValues)



    r = Dict()

    
    for i in range(len(outP)):
             
            r[outP[i]] = outputNeronValues.flatten()[i]/1
        
    return r




def numbaDict(dict):

    x = Dict()

    for k,v in dict.items():
        
        x[k] = v
    
    return x


