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
        for x in range(len(self.rankedIndividuals)-1, 0, -1):
            self.inclusionValue.append(x)

    def calcDistance


b = Population(15)

