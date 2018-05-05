from AirplaneLandings.Individual import Individual


class Population:
    def __init__(self, populationSize):
        self.individuals = []
        self.rankedIndividuals = []
        self.inclusionValue = []
        for x in range(populationSize - 3):
            self.individuals.append(Individual(x))
            self.individuals[x].createRandom()

    def rankIndividuals(self):
        self.rankedIndividuals = sorted(self.individuals,
                                        key=lambda indiv: (indiv.unfitness, -indiv.fitness))
        for x in range(len(self.rankedIndividuals) - 1, 0, -1):
            self.inclusionValue.append(x)

    def calcDistance(self, lhs: Individual, rhs: Individual):
        totaldistance = 0
        for x in range(lhs.numberOfAircrafts):
            if lhs.airplanes[x].runway != rhs.airplanes[x].runway:
                totaldistance += 1
            else:
                totaldistance += abs(lhs.airplanes[x].proportion - rhs.airplanes[x].proportion)
        return totaldistance
