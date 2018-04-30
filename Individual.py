import pandas
import numpy


class Individual:
    def __init__(self, id):
        self.id = id
        self.aircraft = []
        self.proportion = []
        self.runway = []
        self.earliestTime = []
        self.targetTime = []
        self.landingTime = {}
        self.latestTime = []
        self.penaltyCost = []
        self.objectiveFunctionValue = 0
        self.unfitness = 0
        self.createRandom()

    def __repr__(self):
        return repr(
            "The unfitness of individual " + str(self.id) + " is " + str(self.unfitness) + " and obj is: " + str(
                self.objectiveFunctionValue))

    numberOfAircrafts = 10
    numberOfRunways = 1
    directory = "C:\\Users\\janis\\Documents\\Uni\\Master\\TUM\\Airport Operations Management\\Data\\Airland1.xlsx"
    minDistanceBetweenPlane = numpy.zeros((numberOfAircrafts, numberOfAircrafts))

    def createRandom(self):
        import random
        for x in range(self.numberOfAircrafts):
            self.proportion.append(random.random())

        for y in range(self.numberOfAircrafts):
            self.runway.append(random.randint(1, self.numberOfRunways))

        self.calcTime()
        self.calcLandingTime()
        self.calcObjectiveFunction()
        self.calcUnfitnessValue()

    def calcTime(self):
        df = pandas.read_excel(Individual.directory)
        array = df.as_matrix()
        print(list(df))
        for x in range(Individual.numberOfAircrafts):
            self.earliestTime.append(array[x][5])
            self.latestTime.append(array[x][7])
            self.penaltyCost.append(array[x][8])
            self.targetTime.append(array[x][6])
            for y in range(Individual.numberOfAircrafts-1):
                self.minDistanceBetweenPlane[x][y] = array[x][10 + y]

    def calcLandingTime(self):
        for x in range(Individual.numberOfAircrafts):
            earliest = self.earliestTime[x]
            latest = self.latestTime[x]
            self.landingTime[x] = (earliest + self.proportion[x] * (latest - earliest))

    def calcObjectiveFunction(self):
        for x in range(Individual.numberOfAircrafts):
            penalty = abs(self.landingTime[x] - self.targetTime[x]) * self.penaltyCost[x]
            self.objectiveFunctionValue += penalty

    def calcUnfitnessValue(self):
        import operator
        sorted_landingTimes = sorted(self.landingTime.items(), key=operator.itemgetter(1))

        for x in range(0, (Individual.numberOfAircrafts - 1)):
            a = abs(sorted_landingTimes[x][1] - sorted_landingTimes[x + 1][1])
            if a < self.minDistanceBetweenPlane[x][x + 1]:
                self.unfitness += self.minDistanceBetweenPlane[x][x + 1] - a

    def isEqual(self, other):
        return self.id == other.id


