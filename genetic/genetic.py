import time, math, random, bisect, copy
import gym
import numpy as np
import cv2

class NeuralNet : 
    def __init__(self, nodeCount):
        for nci in range(len(nodeCount)):
            nodeCount[nci] = np.prod(nodeCount[nci])
        self.fitness = 0.0
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []
        for i in range(len(nodeCount) - 1):
            norm = 1 / np.sqrt(nodeCount[i])
            self.weights.append(np.random.uniform(low=-norm, high=norm, size=(nodeCount[i], nodeCount[i+1])).tolist() )
            self.biases.append(np.random.uniform(low=0, high=0, size=(nodeCount[i+1])).tolist())
    
    def printWeightsandBiases(self):
        
        print("--------------------------------")
        print("Weights :\n[", end="")
        for i in range(len(self.weights)):
            print("\n [ ", end="")
            for j in range(len(self.weights[i])):
                if j!=0:
                    print("\n   ", end="")
                print("[", end="")
                for k in range(len(self.weights[i][j])):
                    print(" %5.2f," % (self.weights[i][j][k]), end="")
                print("\b],", end="")
            print("\b ],")
        print("\n]")

        print("\nBiases :\n[", end="")
        for i in range(len(self.biases)):
            print("\n [ ", end="")
            for j in range(len(self.biases[i])):
                    print(" %5.2f," % (self.biases[i][j]), end="")
            print("\b],", end="")
        print("\b \n]\n--------------------------------\n")
  
    def getOutput(self, input):
        output = prepare_data(input).flatten()
        for i in range(len(self.nodeCount)-1):
            output = np.reshape( np.matmul(output, self.weights[i]) + self.biases[i], (self.nodeCount[i+1]))
            if i != len(self.nodeCount)-2:
                output = sigmoid(output)
        return output


class Population :
    def __init__(self, populationCount, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ NeuralNet(nodeCount) for i in range(populationCount)]


    def createChild(self, nn1, nn2):
        
        child = NeuralNet(self.nodeCount)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    #if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness + nn2.fitness + 1e-6):
                        child.weights[i][j][k] = nn1.weights[i][j][k]
                    else :
                        child.weights[i][j][k] = nn2.weights[i][j][k]
                    child.weights[i][j][k] += self.m_rate * 2 * (random.random() - 0.5)
                        

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                #if random.random() > self.m_rate:
                if random.random() < nn1.fitness / (nn1.fitness + nn2.fitness + 1e-6):
                    child.biases[i][j] = nn1.biases[i][j]
                else:
                    child.biases[i][j] = nn2.biases[i][j]
                child.biases[i][j] += self.m_rate * 2 * (random.random() - 0.5)

        return child


    def createNewGeneration(self, bestNN):    
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount // 2 + self.popCount % 2):
            #if random.random() < (self.popCount - i) / self.popCount:
            nextGen.append(copy.deepcopy(self.population[i]));
        print('len(nextGen)', len(nextGen))
        
        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit))
        
        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ");
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        
        self.population.clear()
        self.population = nextGen


def sigmoid(x):
    return np.clip(x, 0, 1000)
    x = np.clip(x, -10, 10)
    return 1.0/(1.0 + np.exp(-x))


def replayBestBots(bestNeuralNets, steps, sleep):  
    choice = input("Do you want to watch the replay ?[Y/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation)
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))


def recordBestBots(bestNeuralNets):  
    print("\n Recording Best Bots ")
    print("---------------------")
    env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        for step in range(MAX_STEPS):
            env.render()
            action = bestNeuralNets[i].getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    env.monitor.close()


def uploadSimulation():
    API_KEY = open('/home/dollarakshay/Documents/API Keys/Open AI Key.txt', 'r').read().rstrip()
    gym.upload('Artificial Intelligence/'+GAME, api_key=API_KEY)



def mapRange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.

    return rightMin + (valueScaled * rightSpan)

def normalizeArray(aVal, aMin, aMax): 
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], aMin[i], aMax[i], -1, 1) )
    return res

def scaleArray(aVal, aMin, aMax):   
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], -1, 1, aMin[i], aMax[i]) )
    return res

def prepare_data(x):
    x = x / 255
    #center = (48, 75)
    #cv2.circle(x, center, 4, (1, 1, 1), -1)
    #x = cv2.inRange(x, np.array([0.5, 0, 0]), np.array([0.9, 1, 1]))
    x = cv2.resize(x, (16, 16))
    #cv2.imshow('x', cv2.resize(x, (400, 400)))
    #cv2.waitKey(1)
    return x

GAME = 'CarRacing-v0'
MAX_STEPS = 200
MAX_GENERATIONS = 1000
POPULATION_COUNT = 10
MUTATION_RATE = 0.1
env = gym.make(GAME)
observation = env.reset()

in_dimen = prepare_data(observation).shape
out_dimen = env.action_space.shape
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 10, out_dimen])
bestNeuralNets = []
env.render()

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    minFit =  1000000
    maxFit = -1000000
    maxNeuralNet = None
    for nn in pop.population:
        observation = env.reset()
        totalReward = 0
        for step in range(MAX_STEPS + 10 * (gen + 1)):
            if step % 10 == 0:
                env.render()
                pass
            action = nn.getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break

        nn.fitness = max(0, totalReward)
        print('totalReward', totalReward)
        minFit = min(minFit, nn.fitness)
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            maxNeuralNet = copy.deepcopy(nn);

    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
    pop.createNewGeneration(maxNeuralNet)

recordBestBots(bestNeuralNets)

uploadSimulation()

replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS/10.0))), 0.0625)

