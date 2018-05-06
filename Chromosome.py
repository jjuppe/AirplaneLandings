class Airplane:

    def __init__(self, aircraftNr: int):
        self.aircraftNr = aircraftNr
        self.proportion: float
        self.runwayAllocated: int
        self.earliestTime: int
        self.targetTime: int
        self.latestTime: int
        self.penaltyCost: int
        self.landingTime: int

    def __repr__(self):
        return "airplane: " + str(self.aircraftNr) + " landed at " + str(
            self.landingTime) + ", which equals a proportion of: " + str(self.proportion)
