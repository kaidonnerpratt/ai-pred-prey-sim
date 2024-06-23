
import networks as net
import random as r

'''
module for ai. create a training group with:

[group name] = pyai.trainingGroup(inputs, outputs, HLC, HLNC, netCount, scoreing)

"scoreing" is a function.
'''

class trainingGroup:

    def __init__(self, inputs, outputs, HLC, HLNC, netCount, scoreing):

        self.inP = inputs
        self.outP = outputs
        self.HLNC = HLNC
        self.HLC = HLC
        self.netCount = netCount
        self.networks = []
        self.scoreing = scoreing

        for i in range(self.netCount):
            self.networks.append(net.network(self.inP, self.outP, self.HLC, self.HLNC, i))




    def getBestAnswer(self, input):

        '''
        ! please run the scoreing function before doing this !

        gets the output of the current top network. Calabrate the top network with your scoreing function.
        '''

        return self.networks[0].getOutputs(input)