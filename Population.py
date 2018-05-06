from AirplaneLandings.Individual import Individual


class Population:
    def __init__(self, populationSize):
        self.individuals = []
        self.rankedIndividuals = []
        self.populationSize = populationSize
        self.distanceThreshold = Individual.numberOfAircrafts / 10
        for x in range(populationSize - 3):
            self.individuals.append(Individual(x + 1))
            self.individuals[x].createRandom()
        for x in range(3, 0, -1):
            self.individuals.append(Individual(populationSize - x + 1))
            self.individuals[populationSize - x].createHeuristicIndividual(x)

        self.rankIndividuals()
        self.parentSets = []
        self.availableNodes = set()

    def rankIndividuals(self):
        self.rankedIndividuals = sorted(self.individuals,
                                        key=lambda indiv: (indiv.unfitness, indiv.fitness))
        for x in range(self.populationSize):
            self.rankedIndividuals[x].rank = self.populationSize - x


    def calcDistance(self, lhs: Individual, rhs: Individual):
        totaldistance = 0
        for x in range(lhs.numberOfAircrafts):
            if lhs.airplanes[x].runwayAllocated != rhs.airplanes[x].runwayAllocated:
                totaldistance += 1
            else:
                totaldistance += abs(lhs.airplanes[x].proportion - rhs.airplanes[x].proportion)
        return totaldistance

    def determineParentSets(self):
        while self.individualWithRankRemains():
            # recalculation of available set
            self.availableNodes.clear()
            for x in range(self.populationSize):
                if self.individuals[x].rank > 0:
                    self.availableNodes.add(self.individuals[x])

            # determination of edges
            for x in range(self.populationSize):
                self.individuals[x].hasEdgesToAirplanes.clear()
                for y in range(self.populationSize):
                    if x == y:
                        continue
                    distance = self.calcDistance(self.individuals[x], self.individuals[y])
                    if distance < self.distanceThreshold:
                        self.individuals[x].hasEdgesToAirplanes.add(self.individuals[y])

            # adding to parent sets
            self.parentSets.append(set())
            while len(self.availableNodes) > 0:
                element = self.availableNodes.pop()
                element.rank = element.rank - 1
                self.parentSets[len(self.parentSets) - 1].add(element)
                for neighbour in element.hasEdgesToAirplanes:
                    if self.availableNodes.__contains__(neighbour):
                        self.availableNodes.remove(neighbour)

    def individualWithRankRemains(self):
        for x in range(self.populationSize):
            if self.individuals[x].rank > 0:
                return True
        return False


pop = Population(10)
pop.determineParentSets()
print(*pop.parentSets, sep="\n")
