from AirplaneLandings.Individual import Individual


class Population:
    def __init__(self, populationSize):
        self.individuals = []
        self.rankedIndividuals = []
        self.inclusionValue = [populationSize]
        for x in range(populationSize):
            self.individuals.append(Individual(x))

    def rankIndividuals(self):
        self.rankedIndividuals = sorted(self.individuals,
                                        key=lambda indiv: (indiv.unfitness, -indiv.objectiveFunctionValue))
        for x in range(len(self.rankedIndividuals) - 1, 0, -1):
            self.inclusionValue.append(x)

    def calcDistance(self, lhs: Individual, rhs: Individual):
        totaldistance = 0
        for x in range(lhs.numberOfAircrafts):
            if lhs.runway[x] != rhs.runway[x]:
                totaldistance += 1
            else:
                totaldistance += abs(lhs.proportion[x] - rhs.proportion[x])
        return totaldistance


b = Population(15)

print(b.calcDistance(b.individuals[1], b.individuals[2]))
