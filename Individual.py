import pandas
import numpy
from AirplaneLandings.Chromosome import Airplane
import operator


class Individual:
    def __init__(self, id):
        self.id = id
        self.fitness = 0.0
        self.unfitness = 0.0
        self.airplanes = []
        self.rank = -1
        self.hasEdgesToAirplanes = set()

        # def __repr__(self):
        #   return repr(
        #      "The unfitness of individual " + str(self.id) + " is " + str(self.unfitness) + " and obj is: " + str(
        #         self.fitness) + " and the rank " + str(self.rank) + " in the population")

    def __repr__(self):
        return "Airplane " + str(self.id)

    numberOfAircrafts = 10
    numberOfRunways = 1
    directory = "C:\\Users\\janis\\Documents\\Uni\\Master\\TUM\\Airport Operations Management\\Bionomic Algorithm\\Data\\Airland1.xlsx"
    minDistanceBetweenPlane = numpy.zeros((numberOfAircrafts, numberOfAircrafts))

    def createRandom(self):
        import random
        for x in range(self.numberOfAircrafts):
            self.airplanes.append(Airplane(x))
            self.airplanes[x].proportion = random.random()
            self.airplanes[x].runwayAllocated = random.randint(1, self.numberOfRunways)

        self.importTimes()
        self.calcLandingTime()
        self.calcObjectiveFunction()
        self.calcUnfitnessValue()

    def createHeuristicIndividual(self, type):
        if type == 1:
            self.createEarliestHeuristic()
        if type == 2:
            self.createTargetHeuristic()
        if type == 3:
            self.createLatesttHeuristic()

    def createEarliestHeuristic(self):
        for x in range(Individual.numberOfAircrafts):
            self.airplanes.append(Airplane(x))
            self.airplanes[x].runwayAllocated = 1

        self.importTimes()
        earliest_Sorted = sorted(self.airplanes, key=lambda airplane: airplane.earliestTime)

        earliest_Sorted[0].landingTime = earliest_Sorted[0].earliestTime
        earliest_Sorted[0].proportion = (earliest_Sorted[0].landingTime - earliest_Sorted[0].earliestTime) / (
                earliest_Sorted[0].latestTime - earliest_Sorted[0].earliestTime)

        for x in range(1, Individual.numberOfAircrafts, 1):
            current = earliest_Sorted[x]
            bufferToPrevious = earliest_Sorted[x - 1].landingTime + \
                               self.minDistanceBetweenPlane[earliest_Sorted[x - 1].aircraftNr - 1][
                                   current.aircraftNr - 1]
            current.landingTime = bufferToPrevious if bufferToPrevious > current.earliestTime else current.earliestTime
            current.proportion = (current.landingTime - current.earliestTime) / (
                    current.latestTime - current.earliestTime)

        self.calcObjectiveFunction()
        self.calcUnfitnessValue()

    def createTargetHeuristic(self):
        for x in range(Individual.numberOfAircrafts):
            self.airplanes.append(Airplane(x))
            self.airplanes[x].runwayAllocated = 1

        self.importTimes()
        target_Sorted = sorted(self.airplanes, key=lambda airplane: airplane.targetTime)

        target_Sorted[0].landingTime = target_Sorted[0].targetTime
        target_Sorted[0].proportion = (target_Sorted[0].landingTime - target_Sorted[0].earliestTime) / (
                target_Sorted[0].latestTime - target_Sorted[0].earliestTime)
        for x in range(1, Individual.numberOfAircrafts, 1):
            current = target_Sorted[x]
            bufferToPrevious = target_Sorted[x - 1].landingTime + \
                               self.minDistanceBetweenPlane[target_Sorted[x - 1].aircraftNr - 1][current.aircraftNr - 1]
            current.landingTime = bufferToPrevious if bufferToPrevious > current.earliestTime else current.earliestTime
            current.proportion = (current.landingTime - current.earliestTime) / (
                    current.latestTime - current.earliestTime)

        self.calcObjectiveFunction()
        self.calcUnfitnessValue()

    def createLatesttHeuristic(self):
        for x in range(Individual.numberOfAircrafts):
            self.airplanes.append(Airplane(x))
            self.airplanes[x].runwayAllocated = 1

        self.importTimes()
        latest_Sorted = sorted(self.airplanes, key=lambda airplane: airplane.latestTime)

        latest_Sorted[0].landingTime = latest_Sorted[0].latestTime
        latest_Sorted[0].proportion = (latest_Sorted[0].landingTime - latest_Sorted[0].earliestTime) / (
                latest_Sorted[0].latestTime - latest_Sorted[0].earliestTime)
        for x in range(1, Individual.numberOfAircrafts, 1):
            current = latest_Sorted[x]
            bufferToPrevious = latest_Sorted[x - 1].landingTime + \
                               self.minDistanceBetweenPlane[latest_Sorted[x - 1].aircraftNr - 1][current.aircraftNr - 1]
            current.landingTime = bufferToPrevious if bufferToPrevious > current.earliestTime else current.earliestTime
            current.proportion = (current.landingTime - current.earliestTime) / (
                    current.latestTime - current.earliestTime)

        self.calcObjectiveFunction()
        self.calcUnfitnessValue()

    def importTimes(self):
        df = pandas.read_excel(Individual.directory)
        array = df.as_matrix()
        for x in range(Individual.numberOfAircrafts):
            self.airplanes[x].earliestTime = (array[x][5])
            self.airplanes[x].latestTime = (array[x][7])
            self.airplanes[x].penaltyCost = (array[x][8])
            self.airplanes[x].targetTime = (array[x][6])
            for y in range(Individual.numberOfAircrafts):
                self.minDistanceBetweenPlane[x][y] = array[x][10 + y]

    def calcLandingTime(self):
        for x in range(Individual.numberOfAircrafts):
            earliest = self.airplanes[x].earliestTime
            latest = self.airplanes[x].latestTime
            self.airplanes[x].landingTime = (earliest + self.airplanes[x].proportion * (latest - earliest))

    def calcObjectiveFunction(self):
        for x in range(Individual.numberOfAircrafts):
            penalty = abs(self.airplanes[x].landingTime - self.airplanes[x].targetTime) * self.airplanes[x].penaltyCost
            self.fitness += penalty

    def calcUnfitnessValue(self):
        landingTimes = {}
        for x in range(Individual.numberOfAircrafts):
            landingTimes[x] = self.airplanes[x].landingTime
        sorted_landingTimes = sorted(landingTimes.items(), key=operator.itemgetter(1))

        for x in range(0, (Individual.numberOfAircrafts - 1)):
            a = abs(sorted_landingTimes[x][1] - sorted_landingTimes[x + 1][1])
            if a < self.minDistanceBetweenPlane[x][x + 1]:
                self.unfitness += self.minDistanceBetweenPlane[x][x + 1] - a

    def isEqual(self, other):
        return self.id == other.id


def enum(**named_values):
    return type('Enum', (), named_values)


HeuristicTypes = enum(EARLIEST='1', TARGET='2', LATEST='3')
