import sys
import pyai
import pygame as p
import COLORS as c
import math as m
import time
import random as r
import copy
from numba import njit, jit
from functools import wraps
import numpy as np
global tbl, lst
from sincostable import tbl, lst

class timer:
    def __init__(self):

        self.startTime = time.time()

    def end(self, printout):

        print(printout, " ", time.time() - self.startTime)
        
    def reset(self):
        
        self.startTime = time.time()




def Queue(qi = 0, tasksi = 0, inp = 0, stopWhenDone = True):
    
    def mid(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            
            tasks = args[tasksi]

            args = list(args)

            ret = None

            while True:

                nextTask = tasks.get()
                
                if nextTask is None:
                    tasks.task_done()
                    if stopWhenDone:
                        return 
                    else: 
            
                        continue
            
                elif nextTask == "STOP":
                    tasks.task_done()
                    return

                    
                args[inp] = nextTask[0]

                ret = function(*args, *nextTask[1:],**kwargs)
                
                #rets.append(ret)

                tasks.task_done()
                
                args[qi].put(ret)

            #return rets
        
        return wrapper

    return mid






p.font.init()
font = p.font.SysFont('Comic Sans MS', 30)

@njit(cache = True, fastmath = True)
def rayHit(a,b,s,r):

    #p.draw.line(screen, c.yellow, a, b)
    #p.draw.line(screen, c.yellow, a, s)
    #sp.draw.line(screen, c.yellow, b, s)

    ax, ay = a
    bx, by = b
    px, py = s

    ab_squared = (bx - ax) ** 2 + (by - ay) ** 2
    
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ab_squared
    t = max(0, min(1, t))  

    closest_x = ax + t * (bx - ax)
    closest_y = ay + t * (by - ay)

    distance = m.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    
    return distance <= r, distance


class critter:

    def __init__(self, pos, prey, id, hp = False, parent = None):
        self.id = id
        self.pos = pos
        self.prey = prey
        self.reproduceCounter = 4 * r.random()
        self.starveCounter = 4 * r.random()
        self.shine = False
        if prey:
            self.brain = pyai.net.network(list(range(20)), ['vel', 'rot'], 2, 20, id)
        else:
            self.brain = pyai.net.network(list(range(12)), ['vel', 'rot'], 2, 20, id)

        if hp:
            if r.randint(0,1):
                
                self.brain = parent.brain.mutate()
                for i in range(r.randint(1,3)):
                    self.brain = self.brain.mutate()

            else:
                self.brain = copy.deepcopy(parent.brain)
        else:
            for i in range(r.randint(0,10)):
                self.brain = self.brain.mutate()

            

        self.rvel = 0
        self.vel = 0
        self.rotation = 0
    
        self.brain.id = id

    def draw(self, screen):

        if self.shine == True:
            drawcritter(c.yellow, self.pos, self.rotation, screen)

        elif self.prey:
            drawcritter(c.green, self.pos, self.rotation, screen)

        else: 
            drawcritter(c.red, self.pos, self.rotation, screen)


    def think(self, crits): 

        inputs = self.see(crits) + [self.vel/1, self.rvel/1]

        outputs = self.brain.getOutputs(inputs)
        
        if outputs["vel"] < 0.2 and outputs["vel"] > -0.2 and self.prey:

            self.reproduceCounter += deltatime
            self.starveCounter -= deltatime
            
            if self.starveCounter < 0:
                self.starveCounter = 0

        elif self.prey:

            self.starveCounter += deltatime/2


        return outputs
    
    def see(self, crits):

        critwithoutyou = [i for i in crits if i != self]

        critspos=[list(i.pos) for i in critwithoutyou]

        critspos = np.array(critspos)
        
        typeindex = []
        for i in range(len(critwithoutyou)):

            x = 0.0

            if critwithoutyou[i].prey:
                x = 1.0

            typeindex.append(x)

        if self.prey:

            return multiraycast(ycount, ydence, ylen, self.rotation, self.pos, critspos, lst, typeindex)
            
        else:
            return multiraycast(Pcount, Pdence, Plen, self.rotation, self.pos, critspos, lst, typeindex)






@njit(cache=True, fastmath = True)
def multiraycast(count, dence, leng, rotation, pos, critspos, lst, typeindex):

    inp=[]


    for i in range(count):

        offset = radToDeg(dence*i - (sum(range(count))*dence) / count)


        posible = [leng + 1.0]

        for a in range(len(critspos)):

            o = critspos[a]

            if m.hypot(pos[0]-o[0], pos[1]-o[1]) <= leng:

                

                poc = pointOnCircle(pos, rotation, offset, leng, lst)


                rh, dis = rayHit(pos, poc, o, 10)


                if rh:
    
                    #p.draw.line(screen, c.yellow, pos, (pos[0]+(m.cos(nrr)*leng) , pos[1]+(m.sin(nrr)*leng)))
    
                    posible.append(dis)


                    #else:
                    #p.draw.line(screen, c.white, pos, (pos[0]+(m.cos(nrr)*leng) , pos[1]+(m.sin(nrr)*leng)))

                    posible.append(leng + 1.0)

        if min(posible) != leng + 1:

            inp.append(min(posible))
            inp.append(typeindex[a])
        
        else:
            inp.append(leng/1)
            inp.append(0.0)

    return inp


@njit(cache = True, fastmath = True)
def radToDeg(rad):

    return rad * 180 / 3.14159

@njit(cache = True, fastmath = True)
def degToRad(deg):

    return deg * 3.14159 / 180


@njit(cache = True, fastmath = True)
def pointOnCircle(pos, rot, offset, radius, lst):

    rotation = m.floor((rot+offset)*10)/10

    while rotation < 0:
        rotation += 360
        rotation = m.floor(rotation*10)/10

    while rotation >= 360:
        rotation -= 360
        rotation = m.floor(rotation*10)/10

    lookup = lst[round(rotation*10)]

    return (pos[0] + (lookup[1]*radius), pos[1] + (lookup[0]*radius))


#jit(cache = True, forceobj=True, fastmath = True)
def drawcritter(color, pos, rotation, screen):

    p.draw.circle(screen, color, pos, 10)
    
    p.draw.circle(screen, c.white, pointOnCircle(pos, rotation, 345, 10, lst), 3)
    p.draw.circle(screen, c.black, pointOnCircle(pos, rotation, 345, 11, lst), 1)

    p.draw.circle(screen, c.white, pointOnCircle(pos, rotation, -345, 10, lst), 3)
    p.draw.circle(screen, c.black, pointOnCircle(pos, rotation, -345, 11, lst), 1)

def movement(pos, vel, rotation, rvel, deltatime, speed, moves):


    if moves['rot'] >= 0.2:

        rvel = moves["rot"] 

        if rvel > 1.5:
            rvel = 1.5

    elif moves['rot'] <= -0.2:

        rvel = moves["rot"] 

        if rvel < -1.5:
            rvel = -1.5

    rotation += rvel * deltatime * speed

    if moves['vel'] >= 0.2:

        vel = moves['vel']

        if vel> 1.5:
            vel = 1.5

    elif moves['vel'] <= -0.2:

        vel = moves['vel']

        if vel < -1.5:
            vel = -1.5



    pos[0], pos[1] = pointOnCircle(pos, rotation, 0, vel*deltatime*speed, lst)

    while pos[0] < 0:
        pos[0] += 900

    while pos[0] > 900:
        pos[0] -= 900

    while pos[1] < 0:
        pos[1] += 900

    while pos[1] > 900:
        pos[1] -= 900

    else:
        
    #print(deltatime)

        rvel = 0

    
    return 0, 0, pos, rotation


screen = p.display.set_mode((900, 900)) 

p.display.set_caption('AI sim') 
  
rotation = 90
rvel=0
screen.fill(c.grey) 

p.display.flip() 
if __name__ == '__main__':  

    running = True
    #mp.set_start_method('spawn')

 
else:
    running = False



FPS = 60
gameclock = p.time.Clock()

global speed
speed = 50
 
global Pdence
Pdence = 0.1
global Pcount
Pcount = 5
global Plen
Plen = 110

global ydence
ydence = 0.65
global ycount
ycount = 9
global ylen
ylen = 80




first = True


def critmaker(o, prey):

    crits=[]
    for i in o:

        crits.append(critter([r.randint(0,600),r.randint(0,600)], prey, i))

    #q.put(crits)
    return crits


@Queue(qi = 0, tasksi = 1 , inp = 2, stopWhenDone = False)
def critterize(q, tasks, crit, crits, deltatime, speed):
    
    
    moves = crit.think(crits)
   

    crit.vel, crit.rvel, crit.pos, crit.rotation = movement(crit.pos, crit.vel, crit.rotation, crit.rvel, deltatime, speed, moves)

    

    #q.put(crits)
    return crit


def critreploop(crits):

    crits = sorted(crits, key=lambda x: x.id)


    critstodie = []

    for i in range(len(crits)):
        crits[i].shine = False


        if crits[i].starveCounter > 20:
            critstodie += [i,]

        elif crits[i].reproduceCounter > 40:
            crits[i].reproduceCounter = 0
            crits.append(critter([crits[i].pos[0] + r.randint(-21,21), crits[i].pos[1] + r.randint(-21,21)], crits[i].prey, crits[-1].id+1, parent = crits[i], hp = True))


        
        Pcrits = [x.pos for x in crits if x.prey and not x == crits[i]]
        csx = [x[0] for x in Pcrits]
        csy = [x[1] for x in Pcrits]
        indexes = [x for x in range(len(crits)) if crits[x].prey and not crits[x] == crits[i]]

        if len(Pcrits) == 0:
            Preyoverlaps = []
        else:
            Preyoverlaps = circlesOverlaping(crits[i].pos[0], crits[i].pos[1], csx, csy , indexes)

        if Preyoverlaps != [] and not crits[i].prey:


            critstodie += Preyoverlaps
            crits[i].shine = True
            crits[i].reproduceCounter += 10 
            crits[i].starveCounter -= 5 * deltatime
            if crits[i].starveCounter < 0:
                crits[i].starveCounter = 0

            Preyoverlaps = []

        elif not crits[i].prey:

            crits[i].starveCounter += deltatime / 5


        Xcrits = [x.pos for x in crits if not x.prey and not x == crits[i]]
        csx = [x[0] for x in Xcrits]
        csy = [x[1] for x in Xcrits]
        indexes = [x for x in range(len(crits)) if not crits[x].prey and not crits[x] == crits[i]]

        if len(Xcrits) == 0 or crits[i].prey:
            Predoverlaps = []
        else:
            Predoverlaps = circlesOverlaping(crits[i].pos[0], crits[i].pos[1], csx, csy , indexes)


        overlaps = Preyoverlaps + Predoverlaps



        for x in overlaps:
            
            crits[i].pos[0] -= (((np.sign(crits[x].pos[0] - crits[i].pos[0]) * 10) + crits[i].pos[0]) - ((np.sign(crits[i].pos[0] - crits[x].pos[0]) * 10) + crits[x].pos[0]))/8

            crits[i].pos[1] -= (((np.sign(crits[x].pos[1] - crits[i].pos[1]) * 10) + crits[i].pos[1]) - ((np.sign(crits[i].pos[1] - crits[x].pos[1]) * 10) + crits[x].pos[1]))/8


                

        

    for i in critstodie:
        crits[i] = None

    crits = [i for i in crits if i != None]

    return crits




@njit(cache=True)
def circlesOverlaping(ax, ay, csx, csy, indexes):

    if len(indexes) == 0:
        return indexes


    overlaps = []


    for i in range(len(indexes)):

        if m.hypot((csx[i]-ax),(csy[i]-ay)) <= 15:

            overlaps.append(indexes[i])

    return overlaps




def noqueuecritterize(crits, deltatime, speed):
    for i in range(len(crits)):
        crit = crits[i]
        moves = crit.think(crits)
   
        crit.vel, crit.rvel, crit.pos, crit.rotation = movement(crit.pos, crit.vel, crit.rotation, crit.rvel, deltatime, speed, moves)
        crits[i] = crit

    crits = critreploop(crits)
    #q.put(crits)
    return crits



















global deltatime
realfps = 0
pastfps = 0
gameclock.tick(FPS)
t0 = time.time()
t1=time.time()
frames = 0
while running and __name__ == '__main__': 

    deltatime = t1 - t0
    t0 = time.time()

    #print(deltatime)

    if first:
        #crits = []
        

        crits = critmaker(list(range(200))[:100], True)

        crits += critmaker(list(range(200))[100:], False)
        '''
        with mp.Pool(5) as pl:
            crits.append(pl.map(critmaker, list(range(50))))
            pl.close()
            pl.join()
        '''
        #crits = crits[0]
        #print(crits)


    #for event in p.event.get(): 
    if p.QUIT in [i.type for i in p.event.get()]: 
            running = False
            sys.exit()
        

    #drawcritter(c.blue, (600,600), 270, screen)

    '''
    with mp.Pool(5) as pl:
        bcrits = pl.map(critterize, crits)
        pl.close()
        pl.join()
    '''


    crits = noqueuecritterize(crits, deltatime, speed)

    #bcrits = critterize(crits[0])

    #crits[0] = noqueuecritterize(crits[0], crits, deltatime, speed)

    for i in crits:
        i.draw(screen) 

    realfps = 1.0/(time.time() - t0)
    txt = font.render("FPS: " + str(m.floor(realfps*100)/100), True, c.white)
    preyamount = font.render("Prey: " + str(len([x for x in crits if x.prey])), True, c.white)
    predamount = font.render("Pred: " + str(len([x for x in crits if not x.prey])), True, c.white)



    screen.blit(txt, (0,0))
    screen.blit(preyamount, (0,25))
    screen.blit(predamount, (0,50))




    p.display.flip()
    screen.fill(c.grey)
    
    pastfps = realfps

    t1 = time.time()
    first = False
    frames += 1
