import statistics as stat
import openpyxl as xl
import datetime

MONITORING = False

class Monitor:
    def __init__(self, nr_planes, nr_runways, objective):
        if MONITORING:
            self._data = {'iteration': [],
                          # population
                          'time_elapsed': [], 'pop-size': [], 'pop-best': [], 'pop-worst': [], 'pop-mean-fitness': [],
                          'pop-mean-unfitness': [],
                          'pop-mean-distance': [], 'pop-stdev-distance': [],
                          # parent sets
                          'theta': [], 'parent-sets-number': [], 'parent-set-mean-size': [], 'edges': [],
                          # children
                          'children-number': [], 'children-mean-fitness': [], 'children-mean-unfitness': [],
                          'children-mean-distance': [], 'children-best': [], 'children-stdev-distance': [],
                          # selected parents
                          'parents-number': [], 'parents-fitness': [], 'parents-unfitness': [], 'parents-mean-distance': [],
                          # selected winners over time
                          'winners-mean-fitness': [], 'winners-mean-unfitness': [], 'winners-mean-distance': []
                          }
            self._wb = xl.Workbook()
            self._sht = self._wb.active

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self._file = "monitor-%s-%d-%d-%s.xlsx" % (objective, nr_planes, nr_runways, timestamp)
            self._sht.append(sorted(self._data.keys()))
            self._wb.save(self._file)
            self._iter = 1

    def evaluate_population(self, members, time_elapsed):
        if MONITORING:
            self._data['time_elapsed'].append(time_elapsed)
            self._data['pop-size'].append(len(members))
            self._data['pop-best'].append(max(members).fitness)
            self._data['pop-worst'].append(min(members).fitness)
            self._data['pop-mean-fitness'].append(mean_fitness(members))
            self._data['pop-mean-unfitness'].append(mean_unfitness(members))
            self._data['pop-mean-distance'].append(mean_distance(members))
            self._data['pop-stdev-distance'].append(stdev_distance(members))

    def evaluate_parent_sets(self, parent_sets, theta, edges):
        if MONITORING:
            self._data['theta'].append(theta)
            self._data['edges'].append(edges)
            self._data['parent-sets-number'].append(len(parent_sets))
            self._data['parent-set-mean-size'].append(mean_size(parent_sets))

    def evaluate_children(self, children):
        if MONITORING:
            self._data['children-number'].append(len(children))
            self._data['children-mean-fitness'].append(mean_fitness(children))
            self._data['children-mean-unfitness'].append(mean_unfitness(children))
            self._data['children-mean-distance'].append(mean_distance(children))
            self._data['children-stdev-distance'].append(stdev_distance(children))
            self._data['children-best'].append(max(children).fitness)

    def evaluate_parents(self, parents):
        if MONITORING:
            self._data['parents-number'].append(len(parents))
            self._data['parents-fitness'].append(mean_fitness(parents))
            self._data['parents-unfitness'].append(mean_unfitness(parents))
            self._data['parents-mean-distance'].append(mean_distance(parents))

    def evaluate_winner(self, winners):
        if MONITORING:
            self._data['winners-mean-fitness'].append(mean_fitness(winners))
            self._data['winners-mean-unfitness'].append(mean_unfitness(winners))
            self._data['winners-mean-distance'].append(mean_distance(winners))

    def write_row(self):
        if MONITORING:
            self._data['iteration'].append(self._iter)
            info = []
            for key in sorted(self._data.keys()):
                try:
                    info.append(self._data[key].pop())
                except IndexError:
                    info.append('n/a')
            self._sht.append(info)
            self._wb.save(self._file)
            self._iter += 1

    def write_phenotype(self, phenotype):
        if MONITORING:
            self._wb.create_sheet('phenotype')
            sht = self._wb.get_sheet_by_name('phenotype')
            for runway in phenotype:
                for i, x_i, rw_i in runway:
                    sht.append([i, x_i, rw_i])


def mean_distance(individuals):
    distances = []
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals):
            if i < j:
                distances.append(ind_i.distance(ind_j))
    if len(distances) > 0:
        return stat.mean(distances)
    else:
        return None


def stdev_distance(individuals):
    distances = []
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals):
            if i < j:
                distances.append(ind_i.distance(ind_j))
    if len(distances) > 1:
        return stat.stdev(distances)
    else:
        return None


def mean_fitness(individuals):
    if len(individuals) > 0:
        return stat.mean([ind.fitness for ind in individuals])
    else:
        return None


def mean_unfitness(individuals):
    if len(individuals) > 0:
        return stat.mean([ind.unfitness for ind in individuals])
    else:
        return None


def mean_size(collections):
    if len(collections) > 0:
        return stat.mean([len(collection) for collection in collections])
    else:
        return None
