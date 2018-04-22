import pandas


class Individual:
    def __init__(self, id):
        self.id = id
        self.aircraft = []
        self.proportion = []
        self.runway = []
        self.earliestTime = []
        self.landingTime = []
        self.latestTime = []

    numberOfAircrafts = 10
    numberOfRunways = 1
    directory = "C:\\Users\\janis\\Documents\\Uni\\Master\\TUM\\Airport Operations Management\\Airland1.xlsx"

    def createRandom(self):
        import random
        for x in range(self.numberOfAircrafts):
            self.proportion.append(random.random())

        for y in range(self.numberOfAircrafts):
            self.runway.append(random.randint(1, self.numberOfRunways))

    def calcTime(self):
        df = pandas.read_excel(Individual.directory)
        array = df.as_matrix()
        for x in range(Individual.numberOfAircrafts):
            self.earliestTime.append(array[x][5])
            self.latestTime.append(array[x][7])

    def calcLandingTime(self):
        for x in range(Individual.numberOfAircrafts):
            earliest = self.earliestTime[x]
            latest = self.latestTime[x]
            self.landingTime.append(earliest + self.proportion[x] * (latest - earliest))


x = Individual(1)
x.createRandom()
x.calcTime()
x.calcLandingTime()
print(x.earliestTime)
print(x.landingTime)

a = [2, 1, 3, 4]
a.sort()
print(a)

