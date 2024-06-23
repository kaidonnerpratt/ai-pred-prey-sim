import random as r
import copy
randtable = (1, 1, 1, 1, 2, 2, 2, 3, 3, 4)



class neron:
    def __init__(self, targets, bias):
        self.bias = bias
        self.targets = targets
        self.value = 0

        self.wires = []
        
        if targets != -1:
            for i in targets:
                self.wires.append(0)
        else:
            self.wires = -1



    def setWeight(self, index, value):
        
        '''
        sets a weight to a value
        '''
        
        self.wires[index] = value

    def mutate(self):
        '''
        returns a copy of itself with some of the its wires and biases tweaked
        '''
        
        nself = copy.deepcopy(self)

        a = r.randint(0,1)
        b = r.randint(0,1)

        if not (a and b):
            b = 1

        if a:
            nself.bias = nself.bias + r.random() + (r.randrange(0,len(randtable)) * (-1 ** r.randint(1,2)))
        
        if b and nself.wires != -1:            
            n = r.randrange(0,len(nself.wires))
            nself.wires[n] = nself.wires[n] + r.random() + (r.randrange(0,len(randtable)) * (-1 ** r.randint(1,2)))
        
        return nself


        

