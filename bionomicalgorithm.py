import networkx as nx
import random as rd
import math
from functools import total_ordering
from operator import itemgetter
import helpers as hlp
from enum import Enum
import time
from monitor import Monitor


@total_ordering
class Individual:
    Mode = Enum('Mode', 'child earliest_h target_h latest_h random')
    count = 0

    def __init__(self, alp, mode, chromosome=None, parents=None):
        """Represents a solution for the aircraft landing problem.

        Each solution consists of a chromosome, a phenotype, a sequence
        and fitness and unfitness values.

        - Chromosome: List of tuples (i, y_i, rw_i) for plane i
            with time window proportion y_i on runway rw_i. Usually ordered
            by airplane numnber i!
        - Phenotype: List of lists containing tuples (i, x_i, rw_i) for plane i
            with landing time x_i on runway rw_i. Each runway has its own
            list of tuples. Usually ordered runway-wise by landing time x_i!
        - Sequence: List of lists containing ints i. Ordered sequence of
            airplanes i on each runway to allow for easy duplicate check.

        Args:
            alp (alp.ALP): instance of the aircraft landing problem.
            mode (str): generation scheme for the chromosome
            chromosome (list of tuples): representation of
                a solution candidate using tuples (i, y_i, rw_i) for plane i
                with time window proportion y_i on runway rw_i
            parents (list of Individuals): parents used to generate
                this Individual

        """
        # creation info
        self._id = Individual.count
        self._mode = mode
        Individual.count = Individual.count + 1
        self._parents = parents

        # problem instance
        self._alp = alp  # instance of the aircraft landing problem

        # Structure of the solution
        self._chromosome = []
        self._phenotype = []
        self._sequence = []
        self._fitness = 0.0
        self._unfitness = 0.0

        if self._mode == Individual.Mode.random:
            # Generate Individual from a random chromosome
            self._chromosome = self._get_random_chromosome()
            self._phenotype = self._decode()
        elif self._mode == Individual.Mode.child:
            # Generate Individual using a pre-defined chromosome
            self._chromosome = chromosome
            self._phenotype = self._decode()
        else:
            # Generate Individual using a heuristic phenotype
            if self._mode == Individual.Mode.earliest_h:
                times = self._alp.E
            elif self._mode == Individual.Mode.target_h:
                times = self._alp.T
            elif self._mode == Individual.Mode.latest_h:
                times = self._alp.L
            else:
                raise ValueError("Invalid generation scheme given: %d " % self._mode)
            planes = [(i, t_i) for i, t_i in enumerate(times)]
            planes = sorted(planes, key=itemgetter(1))
            self._phenotype = self._get_heuristic_phenotype(planes)
            self._chromosome = self._encode(self._phenotype)

        # sorting is needed for duplicate check and non-linear local improvement
        self._sort()

        # children will be improved after duplicate check
        if self._mode is not Individual.Mode.child:
            self.improve()

    def __eq__(self, other):
        """Tests for identity of two individuals.

        Required for __hash__() implementation.

        Args:
            other (Individual): other individual to be compared
        """
        return self._id == other._id

    def __le__(self, other):
        """Compares if this individual is lesser than another.

        When ranking individuals, they are first compared by
        their unfitness value. For equal unfitness values, 
        the fitness values are compared whereby higher fitness
        values are better.

        Refers to section 5.2 in Pinol & Beasley (2006).

        Args:
            other (Individual): other individual to be compared

        Returns:
            result (boolean): ``True`` if unfitness greater or fitness
            lesser for equal unfitness values, ``False`` else.
        """
        if self._unfitness == other._unfitness:
            # equal fitness leads to arbitrary sorting
            return self._fitness < other._fitness
        else:
            return self._unfitness > other._unfitness

    def __hash__(self):
        """Calculates a hash value for this individual.

        Required for networkx library.

        Returns:
            hash_value (int): unique id of this individual
        """
        return self._id

    def __repr__(self):
        """Returns a string representation for this individual.
        """
        return "[Ind. %d: %d / %d]" % (self._id, self._unfitness, self._fitness)

    def distance(self, other):
        """Calculates the distance to the chromosome of another individual.

        Refers to eq. (19) in Pinol & Beasley (2006).

        The distance is calculated by summing up the absolute
        differences between the proportion value per plane (0 < d < 1).
        If two planes are not assigned to the same runway, a distance
        of 1 is included.

        Args:
            - other (Individual)

        Returns:
            - distance (float)
        """
        dist = 0.0
        for (i, y_i, rw_i), (j, y_j, rw_j) in zip(self.chromosome, other.chromosome):
            if rw_i != rw_j:
                dist += 1
            else:
                dist += math.fabs(y_i - y_j)
        return dist

    def distances(self, other):
        """Calculates the distance to the chromosome of another individual.

        Returns a distance vector where each element equals the distance
        between

        Args:
            - other (Individual)

        Returns:
            - distance (list of floats): vector of airplane-wise distances
        """
        distances = []
        for (i, y_i, rw_i), (j, y_j, rw_j) in zip(self.chromosome, other.chromosome):
            if rw_i != rw_j:
                distances.append(1)
            else:
                distances.append(math.fabs(y_i - y_j))
        return distances

    def duplicate(self, other):
        """Performs a duplicate check.

        Refers to Section 5.7 in Pinol & Beasley (2006).

        Returns:
            duplicate (boolean)
        """
        return self.sequence == other.sequence

    def improve(self):
        """Improves the Individual locally.

        """
        # Get improved phenotype
        self._phenotype = self._alp.improve_solution(phenotype=self._phenotype)

        # Get corresponding chromosome
        self._chromosome = self._encode(self._phenotype)

        # Update solution quality measures
        self._fitness = self._alp.calc_obj_value(phenotype=self._phenotype)
        self._unfitness = self._alp.calc_constr_violation(phenotype=self._phenotype)

        # Maintain sorting to allow for duplicate check
        self._sort()

    def _decode(self):
        """Computes the phenotype of this individual.
        """
        # initialize phenotype
        phenotype = [[] for _ in range(self._alp.nr_runways)]

        for i, y_i, rw_i in self._chromosome:
            x_i = self._alp.calc_time_abs(i, y_i)
            phenotype[rw_i].append((i, x_i, rw_i))

        # exclude empty runways
        phenotype = [runway for runway in phenotype if runway != []]
        return phenotype

    def _encode(self, phenotype):
        """Computes the chromosome corresponding to the given phenotype.
        """

        chromosome = [None] * self._alp.nr_planes
        for runway in phenotype:
            for i, x_i, rw_i in runway:
                y_i = self._alp.calc_time_prop(i, x_i)
                chromosome[i] = (i, y_i, rw_i)

        return chromosome

    def _sort(self):
        """Sorts the phenotype according to the landing times.

        Refers to Section 5.7 in Pinol & Beasley (2006).
        """
        # sort runways by smallest aircraft number on each runway
        self._phenotype = [rw for rw in self._phenotype if len(rw) > 0]
        self._phenotype.sort(key=lambda runway: min(runway, key=lambda plane: plane[0])[0])

        # sort each runway by landing time and airplane number
        for runway in self._phenotype:
            runway.sort(key=lambda plane: (plane[1], plane[0]))

        # store sequence for duplicate check
        self._sequence = [[i for i, x_i, rw_i in runway] for runway in self._phenotype]

    def _get_heuristic_phenotype(self, ordered_planes):
        """Calculates a heuristic chromosome for a given order of planes.

        Refers to section 5.3 in Pinol & Beasley (2006).

        Args:
            ordered_planes (list of ints): list of plane numbers ordered
                for one time attribute (earliest, target or latest time).
        """
        phenotype = [[] for _ in range(self._alp.nr_runways)]
        latest_times = self._alp.L
        sep_times = self._alp.S

        for plane, time in ordered_planes:
            runway = None
            # try empty runway
            for runway_nr, runway_seq in enumerate(phenotype):
                if runway_seq is []:  # no airplanes scheduled yet
                    runway = runway_nr

            if runway is not None:
                # use time that was used for ordering the airplanes
                phenotype[runway].append((plane, time, runway))
                continue

            possible_times = []
            for runway_seq in phenotype:
                earliest_time = time
                for other_plane, landing_time, runway in runway_seq:
                    earliest_time = max(earliest_time, landing_time + sep_times[plane][other_plane])
                possible_times.append(earliest_time)

            runway, scheduled_time = min((idx, val) for idx, val in enumerate(possible_times))

            # make sure that proportion is <= 1
            scheduled_time = min(latest_times[plane], scheduled_time)
            phenotype[runway].append((plane, scheduled_time, runway))

        return phenotype

    def _get_random_chromosome(self):
        """Generates a random chromosome for the given problem instance.

        Returns:
            chromosome (list of tuples): representation of
                a solution candidate using tuples (i, y_i, rw_i) for plane i
                with time window proportion y_i on runway rw_i
        """
        return [(i, rd.random(), rd.randint(0, self._alp.nr_runways - 1)) for i in
                range(self._alp.nr_planes)]

    @property
    def chromosome(self):
        return self._chromosome

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def sequence(self):
        return self._sequence

    @property
    def fitness(self):
        return self._fitness

    @property
    def unfitness(self):
        return self._unfitness


class Population:
    def __init__(self, alp, size=100):
        self._alp = alp  # instance of the aircraft landing problem

        self._members = list()
        for i in range(size):
            if i < size - 3:  # random individuals
                self._members.append(Individual(alp, mode=Individual.Mode.random))
            elif i == size - 3:  # heuristic individuals
                self._members.append(Individual(alp, mode=Individual.Mode.earliest_h))
            elif i == size - 2:
                self._members.append(Individual(alp, mode=Individual.Mode.target_h))
            elif i == size - 1:
                self._members.append(Individual(alp, mode=Individual.Mode.latest_h))
            hlp.progress_bar(current=i + 1, end=size, title=format('[ INIT POP   ]'))

        # Initial sorting according to fitness
        self._members = sorted(self._members)

        print('\r[ INIT POP   ] Size: %d / Best fitness: %d' % (len(self._members), max(self._members).fitness),
              flush=True)

        # Setup graph structure
        # - stores distances below below threshold
        # - makes it easy to derive maximum independent sets (parent selection)
        self._graph = nx.Graph()
        self._threshold = self._alp.nr_planes / 10

        # Add each individual as node to the graph
        for individual in self._members:
            self._graph.add_node(individual)

        # For each pair of individuals that are too close, add an edge to the graph
        relations = [(ind_a, ind_b) for ind_a in self._members for ind_b in self._members if ind_b != ind_a]
        for ind_a, ind_b in relations:
            if not self._graph.has_edge(ind_a, ind_b):
                distance = ind_a.distance(ind_b)
                if distance < self._threshold:
                    self._graph.add_edge(ind_a, ind_b, weight=distance)

    def generate_parent_sets(self):
        """ Generates a list of parent sets for child generation

        Generates a set of parents according section 5.5 in Pinol & Beasley (2006).

        In the parent selection process a distance measure is introduced to keep diversity
        in the parent set high. Nodes whose distance is less than a specified threshold value
        have an edge. When selecting a node for the parent set, each of his neighbours cannot
        be added to the same set. Furthermore, better individuals have higher probability of
        being selected for a parent set because the inclusion frequency corresponds to their
        rank.

        Returns:
            parent_sets (list of sets): list of parent sets, where each parent
                set can have different sizes
        """
        # update sorting to obtain valid ranks
        self._members = sorted(self._members)

        # start with graph that contains all possible
        # nodes and edges and assign ranks according to
        # each individuals fitness
        main_graph = self._graph.copy()
        for rank, individual in enumerate(self._members):
            # asc sorting --> worst individual is assigned
            # lowest rank
            main_graph.node[individual]['rank'] = rank + 1

        # individuals with distance below threshold
        # must have an edge --> others are removed
        # iteratively to reach reasonable number of edges
        max_nr_edges = len(self._members) * (len(self._members) - 1) / 2
        theta = self._threshold

        while main_graph.number_of_edges() > max_nr_edges / 2:
            # get edges to be removed because of too large distance
            edges = [(f, t) for (f, t, w) in main_graph.edges(data='weight') if w >= theta]
            main_graph.remove_edges_from(edges)
            # Further reduce number of edges in next iteration
            theta = theta / 2.0

        total_nr_parents = sum([rank for (parent, rank) in main_graph.nodes(data='rank')])

        # obtain sets of parent individuals while rank
        # (= inclusion frequency) greater than zero
        parent_sets = []
        while len(main_graph) > 0:

            parent_set = set()
            set_graph = main_graph.copy()

            while len(set_graph) > 0:
                # pick random node
                individual = rd.choice(list(set_graph))
                parent_set.add(individual)
                new_rank = main_graph.node[individual]['rank'] - 1
                if new_rank <= 0:
                    # remove node from initial graph
                    main_graph.remove_node(individual)
                else:
                    main_graph.node[individual]['rank'] = new_rank
                neighbors = list(set_graph.neighbors(individual))
                set_graph.remove_node(individual)
                set_graph.remove_nodes_from(neighbors)

            parent_sets.append(parent_set)

            # Print progress
            nr_parents = total_nr_parents - sum([r for (n, r) in main_graph.nodes(data='rank')])
            hlp.progress_bar(nr_parents, total_nr_parents, '[ SELECTION  ]')

        parent_sets = [p_set for p_set in parent_sets if len(p_set) > 1]

        print('\r[ SELECTION  ] Sets generated: %d' % (len(parent_sets)), flush=True)

        return parent_sets, theta

    def generate_children(self, parent_sets):
        """Generates a set of children from a given set of parents

        Generates a set of children from the given parentset according to section 5.6 in Pinol & Beasley (2006).
        Then checks if an individual with the same sequence already exists in the population and removes this
        child in that case according to section 5.7.
        Locally improves every child from the children set according to section 5.8, depending on whether the
        non-linear objective or linear objective is chosen.

        Args:
            parent_sets (list of sets): sets of individuals

        Returns:
            children (list of individuals): list of generated children
        """

        children = []
        for set_nr, parent_set in enumerate(parent_sets):
            if parent_set is None:
                return
            # generate random weights for each parent
            abs_weights = [rd.random() for _ in range(len(parent_set))]
            # normalize weights
            sum_of_weights = sum(abs_weights)
            rel_weights = [w / sum_of_weights for w in abs_weights]

            chromosome = []
            for i in range(self._alp.nr_planes):
                # determine proportion value
                parent_props = [parent.chromosome[i][1] for parent in parent_set]
                child_prop = round(sum([w * p for w, p in zip(rel_weights, parent_props)]), 6)

                # determine runway
                parent_runways = [parent.chromosome[i][2] for parent in parent_set]
                child_rw = rd.choice(parent_runways)

                # add to chromosome
                chromosome.append((i, child_prop, child_rw))

            child = Individual(alp=self._alp, mode=Individual.Mode.child, chromosome=chromosome, parents=parent_set)

            # exclude duplicates with respect to the current population
            if not self._duplicate(child):
                child.improve()
                children.append(child)

            # Print progress
            hlp.progress_bar(current=set_nr + 1, end=len(parent_sets), title=format('[ CROSSOVER  ]'))

        # Print information
        if len(children) > 0:
            print('\r[ CROSSOVER  ] Children generated: %d' % len(children), flush=True)
        else:
            print('\r[ CROSSOVER  ] No children generated', flush=True)

        return children

    def insert_children(self, children):
        """Inserts the best children into the population and eliminates the worst individual

        According to section 5.9 in Pinol & Beasley (2006) only the best child is inserted into
        the population and the worst individual is eliminated from the population.

        Args:
            children (list of individuals): list of children which has been generated

        Returns:
            child (Individual): inserted child
        """
        if len(children) == 0:
            return

        child = max(children)  # get best fitted child
        loser = self._members.pop(0)  # remove least fitted individual
        self._graph.remove_node(loser)
        self._graph.add_node(child)

        # add to distance graph
        for member in self._members:
            dist = child.distance(member)
            if dist < self._threshold:
                self._graph.add_edge(child, member, weight=dist)

        self._members.append(child)
        print('[ INSERTION  ] Child fitness: %d' % child.fitness)
        return child

    def _duplicate(self, other):
        for member in self._members:
            if other.duplicate(member):
                return True
        return False

    @property
    def members(self):
        return self._members


class BionomicAlgorithm():
    def __init__(self, alp):
        self.alp = alp

    def run(self):
        """ runs the bionomic algorithm

        The steps for the bionomic algorithm are corresponding to section 5.10  from Pinol & Beasley (2006).
        1. An initial population is created
        2. Each individual from the initial population is locally improved
        repeating:
        3. Generation of parent sets
        4. Generation of children sets incl. removal of duplicates
        5. Local improvement of every children
        6. Insertion into population of best child and removing of worst individual in population
        until termination

        The algorithm terminates once 40,000 children have been created

        :return: the phenotyoe of the best solution
        """
        monitor = Monitor(self.alp.nr_planes, self.alp.nr_runways, self.alp._objective)
        nr_children_limit = 50000  # termination criterion:
        nr_children_current = 0  # counter
        iter_without_children = 0  # subsequent iterations with duplicates only
        population = Population(alp=self.alp, size=100)
        winners = []  # set of inserted children
        iteration = 0  # iteration counter
        start_time = time.time()
        while nr_children_current < nr_children_limit and iter_without_children < 10:
            monitor.evaluate_population(population.members, time.time() - start_time)
            iteration += 1
            print('- ' * 10)
            print('[ STATUS     ] Iteration %d / Children: %d' % (iteration, nr_children_current))
            parent_sets, theta = population.generate_parent_sets()
            monitor.evaluate_parent_sets(parent_sets, theta, population._graph.number_of_edges())
            children = population.generate_children(parent_sets)
            if len(children) > 0:
                monitor.evaluate_children(children)
                monitor.evaluate_parents(max(children)._parents)
                winner = population.insert_children(children)
                winners.append(winner)
                monitor.evaluate_winner(winners)
                nr_children_current += len(children)
                iter_without_children = 0
            else:
                iter_without_children += 1
            monitor.write_row()

        return population.members.pop().phenotype
