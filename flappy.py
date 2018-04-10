from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

import numpy as np

import neuralNetwork as nn


FPS = 60
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79

#normalizing factors
PIPE_HOLE_OFFSET = (int(BASEY * 0.2) + 0.5*PIPEGAPSIZE) 
PIPE_HOLE_RANGE = int(BASEY * 0.6 - PIPEGAPSIZE)
MAX_PIPE_X = SCREENWIDTH + 10
#BASEY is used for normalizing p.Y
# image and hitmask  dicts
IMAGES, HITMASKS = {}, {}

total_models = 100

models = []

next_pipe_x = -1
next_pipe_hole_y = -1

next_pipe_x = -1
next_pipe_hole_y = -1

generation = 1
last_man_standing = False

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

class Player():

    def __init__(self, net):
        # player velocity, max velocity, downward accleration, accleration on flap
        self.net = net
        self.reset(0, 0)

    def reset(self, x, y):
        # player velocity, max velocity, downward accleration, accleration on flap
        self.VelY    = -9    # player's velocity along Y, default same as playerFlapped
        self.MaxVelY =  10   # max vel along Y, max descend speed
        self.MinVelY =  -8   # min vel along Y, max ascend speed
        self.AccY    =  1   # players downward accleration
        self.FlapAcc =  -9   # players speed on flapping
        self.Flapped = False # True when player flaps
        self.State = True

        self.X = x
        self.Y = y

        self.fitness = 0
        self.index = 0

    def predict_action(self, next_pipe_x, next_pipe_hole_y, next2_pipe_x, next2_pipe_hole_y):
        # Normalized Variables that yielded poor performance
        # n_pipe_x = next_pipe_x / MAX_PIPE_X
        # n_player_y = self.Y/BASEY
        # n_pipe_hole_y = (next_pipe_hole_y - PIPE_HOLE_OFFSET) / PIPE_HOLE_RANGE
       
        # normalized_inputs = [n_pipe_x, n_pipe_hole_y, n_player_y]
        # for i in range(len(normalized_inputs)):
        #     if normalized_inputs[i] > 1:
        #         print("WARNING!!!!!!!!!!!!!! UNNORMALIZED VAR {0}".format(i))

        inputs = []
        inputs.append(self.VelY)
        inputs.append(next_pipe_x)
        inputs.append(self.Y-next_pipe_hole_y)
        inputs.append(next2_pipe_x)
        inputs.append(self.Y - next2_pipe_hole_y)

        r = self.net.evaluate([self.VelY, next_pipe_x, self.Y - next_pipe_hole_y, next2_pipe_x, self.Y - next2_pipe_hole_y])[0] #, next2_pipe_x, next2_pipe_hole_y
        return  1 if r > 0.5 else 0

    def debug_lines(self):
        global next_pipe_x
        global next_pipe_hole_y
        pygame.draw.line(SCREEN, (255, 0, 255), (next_pipe_x, 0), (next_pipe_x, SCREENHEIGHT))
        pygame.draw.line(SCREEN, (255, 0, 255), (0, self.Y), (SCREENWIDTH, self.Y))
        pygame.draw.line(SCREEN, (255, 0, 0), (0, next_pipe_hole_y), (SCREENWIDTH, next_pipe_hole_y))


    def checkCrash(self, upperPipes, lowerPipes, playerIndex):
        """returns True if player collders with base or pipes."""
        width = IMAGES['player'][0].get_width()
        height = IMAGES['player'][0].get_height()
        # if player crashes into ground

        if self.Y + height >= BASEY - 1:
            return True
        playerRect = pygame.Rect(self.X, self.Y,
                      width, height)

        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][playerIndex]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                #self.fitness += 20/(self.Y - next_pipe_hole_y) if next_pipe_hole_y < self.Y and self.Y != next_pipe_hole_y else 20/(self.Y - next_pipe_hole_y + 0.1)
                return True
        return False

def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    
    global models
    models = []
    for i in range(total_models):
        n = nn.neuralNetwork(5, 11, 1)
        models.append(Player(n))

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        mainGame(movementInfo)
        showGameOverScreen() #crashInfo)


def showWelcomeAnimation():
    return {
                'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                'basex': 0,
                'playerIndexGen': cycle([0, 1, 2, 1]),
            }


def mainGame(movementInfo):
    global FPS, last_man_standing
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 3 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    newPipe3 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 10, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 10 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        {'x': SCREENWIDTH + 10 + (SCREENWIDTH), 'y': newPipe3[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 10, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 10 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        {'x': SCREENWIDTH + 10 + (SCREENWIDTH), 'y': newPipe3[1]['y']},
    ]

    global next_pipe_x
    global next_pipe_hole_y

    next_pipe_x = lowerPipes[0]['x']
    next_pipe_hole_y = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()))/2

    global next2_pipe_x
    global next2_pipe_hole_y
    next2_pipe_x = lowerPipes[1]['x']
    next2_pipe_hole_y = (lowerPipes[1]['y'] + (upperPipes[1]['y'] + IMAGES['pipe'][1].get_height()))/2

    pipeVelX = -4

    start_x, start_y = int(SCREENWIDTH * 0.2), movementInfo['playery']
    for p in models:
        p.reset(start_x, start_y)

    alive_players = len(models)

    while True:

        #Kill all players that hit ceiling
        for p in models:
            if p.Y < 0 and p.State == True:
                alive_players -= 1
                p.State = False

        #Check if collided with pipes collisions
        for p in models:
            if p.State == True:
                if p.checkCrash(upperPipes, lowerPipes, playerIndex):
                    alive_players -= 1
                    p.State = False

        #if less than one model alive, stop game
        #if one model is alive, add 10000 to fitness so it is known to have
        #greatest fitness (instead of waiting until it dies - takes too long)
        if alive_players <= (0 if last_man_standing else 1):
            for p in models:
                if p.State:
                    p.fitness += 10000
            return 

        #If model is alive this frame, add 1 to fitness
        for p in models:
            if p.State == True:
                p.fitness += 1

        #Advance pipes for prediction
        next_pipe_x += pipeVelX
        next2_pipe_x += pipeVelX

        #If model's neural net says to flap, flap
        for p in models:
            if p.State == True:
                if p.predict_action(next_pipe_x, next_pipe_hole_y, next2_pipe_x, next2_pipe_hole_y) == 1:
                    if p.Y > -2 * IMAGES['player'][0].get_height():
                        p.VelY = p.FlapAcc
                        p.Flapped = True

        #If quitting, - quit
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_a:
               FPS = 30 if FPS == 240 else 240
            elif event.type == KEYDOWN and event.key == K_c:
               last_man_standing = not last_man_standing

        has_added_score = False
        # check if crossed pipe
        for p in models:
            if p.State == True:
                pipe_idx = 0
                playerMidPos = p.X + IMAGES['player'][playerIndex].get_width()/2
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width()
                    #Passes through pipe
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        #Add to fitness score
                        p.fitness += 25 
                        p.fitness += 50/(p.Y - next_pipe_hole_y) if next_pipe_hole_y < p.Y and p.Y != next_pipe_hole_y else 50/(p.Y - next_pipe_hole_y + 0.1)
                        
                        #get next pipe values
                        next_pipe_x = lowerPipes[pipe_idx+1]['x']
                        next_pipe_hole_y = (lowerPipes[pipe_idx+1]['y'] + (upperPipes[pipe_idx+1]['y'] + IMAGES['pipe'][pipe_idx+1].get_height())) / 2
                        next2_pipe_x = lowerPipes[pipe_idx+2]['x']
                        next2_pipe_hole_y = (lowerPipes[pipe_idx+2]['y'] + (upperPipes[pipe_idx+2]['y'] + IMAGES['pipe'][pipe_idx+1].get_height())) / 2
                        
                        if not has_added_score:
                            score += 1
                            has_added_score = True
                    pipe_idx += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for p in models:
            if p.State == True:
                if p.VelY < p.MaxVelY and not p.Flapped:
                    p.VelY += p.AccY
                if p.Flapped:
                    p.Flapped = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                p.Y += min(p.VelY, BASEY - p.Y - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
                newPipe = getRandomPipe()
                upperPipes.append(newPipe[0])
                lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)
        showGeneration(generation)
        showAlive(alive_players)

        for p in models:
            if p.State == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (p.X, p.Y))

        #Draw debug lines 
        for p in models:
            if p.State:
                p.debug_lines()

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(): #crashInfo):
    """Perform genetic updates here"""
    new_models = []
    global models
    global generation
    models.sort(key= lambda x: x.fitness, reverse=True)

    totalFitness = -10000
    for p in models:
        totalFitness += p.fitness

    output = "\n{0},{1},{2}".format(generation, int(totalFitness), int(models[0].fitness))#, models[0].net)
    writeToFile("generationSummary.csv", [output])
    output = "\nGeneration {0} fittest network:\n{1}".format(generation, models[0].net)
    writeToFile("BestSpecies.txt", [output])
    new_nets = nn.breed(models[0].net, models[1].net, 15)
    new_nets.extend(nn.breed(models[1].net, models[0].net, 15))
    new_nets.extend(nn.breed(models[1].net, models[2].net, 7))
    new_nets.extend(nn.breed(models[2].net, models[1].net, 8))

    new_nets.extend(nn.breed(models[2].net, models[3].net, 5))
    new_nets.extend(nn.breed(models[3].net, models[2].net, 4))
    for n in new_nets:
        nn.mutate(n)

    new_nets.append(models[0].net.clone())
    new_nets.append(models[1].net.clone())
    models = []
    for i in range(len(new_nets)):
        models.append(Player(new_nets[i]))
    generation = generation + 1
    return



def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH * 1.5

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showGeneration(generation):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(generation))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) - 10

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, 10))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showAlive(alive):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(alive))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) - 10

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.9))
        Xoffset += IMAGES['numbers'][digit].get_width()


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def writeToFile(filename, linesToWrite):
    with open(filename, "a") as f:
        f.writelines(linesToWrite)

if __name__ == '__main__':
    main()
