class Individual:


    def __init__(self, id):
        self.id = id
        self.aircraft = []
        self.proportion = []
        self.runway = []

    numberOfAircrafts = 20
    numberOfRunways = 4

    def createRandom(self):
        import random
        for x in range(self.numberOfAircrafts):
            self.proportion.append(random.random())

        for y in range(self.numberOfAircrafts):
            self.runway.append(random.randint(1, self.numberOfRunways))



x = Individual(1)
x.createRandom()
print(x.proportion)