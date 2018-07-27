# -*- coding: utf-8 -*-
""" Contains a methods to create instances of the aircraft landing problem.
"""

import math
import gurobipy as grb
import bionomicalgorithm as ba
from enum import Enum
import copy


class ALP:
    Objective = Enum('Objective', 'min_cost max_util')
    Mode = Enum('Mode', 'normal relaxed')

    def __init__(self, data, objective, nr_runways=1):
        """Represents an instance of the aircraft landing problem.

        - i as aircraft unique id within the problem instance
        - E as vector of earliest landing times
        - T as vector of target landing times
        - L as vector of latest landing times
        - h as vector of penalty costs per time unit before target landing time
        - g as vector of penalty cost per time unit after target landing time

        In addition, an instance can be solved for either one (default)
        or multiple runways which are treated equally.

        The problem can be solved for two objectives:
        - linear: minimize weighted earliness and lateness
        - non-linear: maximize utilization (earliness)

        Args:
            data (list of ints): instance data according to the
                specification of the OR library (see Beasley (1990)).
            objective (ALP.Objective): 'min_cost' for minimized weighted earliness
                and lateness or 'max_util' for maximized earliness
            nr_runways (int): number of equally treated runways
                the instance should be solved for
        """
        # store persistent copy of input data
        self._data = copy.copy(data)

        # Parse input data
        self.E = list()
        self.T = list()
        self.L = list()
        self.S = list()
        self.h = list()
        self.g = list()
        self.nr_planes = int(data.pop(0))

        int(data.pop(0))  # freeze time: discarded
        for i in range(self.nr_planes):
            int(data.pop(0))  # appearance time: discarded
            self.E.append(int(data.pop(0)))  # earliest landing time
            self.T.append(int(data.pop(0)))  # target landing time
            self.L.append(int(data.pop(0)))  # latest landing time
            self.g.append(data.pop(0))  # penalty before target
            self.h.append(data.pop(0))  # penalty after target
            # separation times to preceeding aircraft
            self.S.append([int(data.pop(0)) for j in range(self.nr_planes)])

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(range(self.nr_planes),self.E)
        # plt.plot(range(self.nr_planes),self.T)
        # plt.plot(range(self.nr_planes),self.L)
        # plt.legend(["earliest landing time", "target landing time", "latest landing time"], loc="upper right")
        # plt.xlabel("planes")
        # plt.ylabel("time")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig("Test_data.png")

        if nr_runways <= 0:
            raise ValueError("Illegal number of runways: %d" % nr_runways)
        else:
            self.nr_runways = nr_runways

        # Perform Pre-Processing
        self.U, self.V, self.W = self._preprocess_airplanes()

        # Configure linear program
        self._objective = objective
        self._mode = None

        # model is only solvable for linear objective
        if self._objective is ALP.Objective.min_cost:
            # maintain separate models to avoid re-building if mode is switched

            self._objectives = {ALP.Mode.normal: None, ALP.Mode.relaxed: None}
            self._normal_model = grb.Model()
            self._relaxed_model = grb.Model()

            self._constrs = {self._normal_model: dict(), self._relaxed_model: dict()}  # store all constraints
            self._dvars = {self._normal_model: dict(), self._relaxed_model: dict()}  # store all decision variables

            self._build_dynamic_model(self._normal_model)
            self._enforce_sep_time(self._normal_model)

            self._build_dynamic_model(self._relaxed_model)
            self._relax_sep_time(self._relaxed_model)

    def solve_for_optimality(self, phenotype=None, relax_sep_time=False):
        """Solves this instance for optimality.

        Args:
            phenotype (list of lists containing tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i whose sequence is
                maintained. Optional argument.
            relax_sep_time (boolean): Indicates if the separation time
                constraints are relaxed.

        Returns:
            phenotype (list of lists with tuples): representation of
                an improved solution candidate using tuples (i, x_i, rw_i)
                for plane i with landing time x_i on runway rw_i
        """

        if relax_sep_time:
            model = self._relaxed_model
        else:
            model = self._normal_model

        self._clear_sequence(model)

        if phenotype is not None:
            self._enforce_sequence(model, phenotype)

        # if relax_sep_time:
        #     self._relax_sep_time()
        # else:
        #     self._enforce_sep_time()

        model.optimize()

        if model.status == grb.GRB.Status.INFEASIBLE or model.status == grb.GRB.Status.INF_OR_UNBD:
            return None

        x = [self._dvars[model]['x'][i].X for i in range(self.nr_planes)]
        rw = []
        for i in range(self.nr_planes):
            for r in range(self.nr_runways):
                if self._dvars[model]['z'][i, r].X > 0:
                    rw.append(r)

        phenotype = [[] for _ in range(self.nr_runways)]
        for i, x_i in enumerate(x):
            phenotype[rw[i]].append((i, round(x_i, 5), rw[i]))

        return phenotype

    def solve_heuristically(self):
        """Solves this instance using the Bionomic Algorithm.

        Returns:
            phenotype (list of lists with tuples): representation of
                the best solution found using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i
        """
        bionomic_alg = ba.BionomicAlgorithm(self)
        return bionomic_alg.run()

    def improve_solution(self, phenotype):
        """Locally improves a given landing sequence.

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            phenotype (list of lists with tuples): representation of an
            improved solution candidate using tuples (i, x_i, rw_i) for plane i
            with landing time x_i on runway rw_i
        """
        if self._objective is ALP.Objective.min_cost:
            return self._improve_min_cost_sol(phenotype)
        elif self._objective is ALP.Objective.max_util:
            return self._improve_max_util_sol(phenotype)
        else:
            # should not be reached
            raise ValueError("Invalid objective for local improvement: %s" % self._objective)

    def calc_obj_value(self, phenotype):
        """Calculates the objective value for a given solution.

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            objective_value (float): objective value for the
                instance objective and given solution which is converted
                to suit a maximization problem.
        """
        if self._objective is ALP.Objective.min_cost:
            return self._calc_min_cost_obj(phenotype)
        elif self._objective is ALP.Objective.max_util:
            return self._calc_max_util_obj(phenotype)
        else:
            # should not be reached
            raise ValueError("Invalid objective for calculation of objective value: %s" % self._objective)

    def calc_constr_violation(self, phenotype):
        """Calculates the degree of constraint violation.

        For each plane, the violation is measured by summing up
        positive differences of required separation time and
        actual separation time for succeeding planes. The sum

        Refers to equation (18) in section 5.2 in Pinol & Beasley (2006).

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            violation_measure (float): degree to which the given
                solution violates separation time constraints.
        """
        violation = 0.0
        for runway in phenotype:
            for i, x_i, rw_i in runway:
                for j, x_j, rw_j in runway:
                    if i != j and rw_i == rw_j and x_i <= x_j:
                        violation += max(self.S[i][j] - (x_j - x_i), 0)
        return violation

    def calc_time_prop(self, i, x_i):
        """Calculates the time window proportion for a given landing time.

        Args:
            i (int): number of the airplane
            x_i (float): scheduled landing time of the airplane

        Returns:
            proportion (float): proportion of the allowed time time_window

        Raises:
            ValueError: if not 0 <= proportion <= 1
        """
        time_window = self.L[i] - self.E[i]

        # for E_i == L_i the proportion value does not matter
        if time_window == 0:
            return 0

        # calculcate proportion value
        y_i = (x_i - self.E[i]) / time_window

        if not 0 <= y_i <= 1:
            # should not be reached
            raise ValueError("Illegal time window proportion: %s" % y_i)

        return y_i

    def calc_time_abs(self, i, y_i):
        """Calculates the landing time for a given time window proportion.

        Args:
            i (int): number of the airplane
            y_i (float): proportion of the allowed time window
                where 0 <= proportion <= 1

        Returns:
            x_i (float): scheduled landing time of the airplane

        Raises:
            ValueError: if not 0 <= proportion <= 1
        """
        if y_i > 1 or y_i < 0:
            raise ValueError("Illegal time window proportion: %f" % y_i)

        time_window = self.L[i] - self.E[i]
        return self.E[i] + y_i * time_window

    def _build_base_model(self, model):
        """Builds a MIP for this instance.

        Refers to section 2 in Pinol & Beasley (2006).

        Beforehand, a the Gurobi model object must be initialized.
        The built model does not allow for any adaptions needed for local
        improvement. For this purpose, _build_dynamic_model() must be called.
        """

        # Sets
        # --------------------------------------------------

        P = range(self.nr_planes)  # set of airplanes
        R = range(self.nr_runways)  # set of runways

        # Section 2.1: Decision Variables
        # --------------------------------------------------

        # 1 if aircraft i (in P) lands on runway r (in R), 0 else
        z = self._normal_model.addVars(P, R, vtype=grb.GRB.BINARY, name='z')
        self._dvars[model]['z'] = z

        # 1 if aircraft i (in P) and j (in P) lands on the same runway, 0 else
        gamma = self._normal_model.addVars(P, P, vtype=grb.GRB.BINARY, name='gamma')
        self._dvars[model]['gamma'] = gamma

        # 1 if aircraft i (in P) lands before aircraft j (in P)
        delta = self._normal_model.addVars(P, P, vtype=grb.GRB.BINARY, name='delta')
        self._dvars[model]['delta'] = delta

        # Scheduled landing time for aircraft i (in P)
        x = self._normal_model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='x')
        self._dvars[model]['x'] = x

        # Proportion of the landing time window for aircraft i (in P)
        # Includes Eq. (3)
        y = self._normal_model.addVars(P, vtype=grb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y')
        self._dvars[model]['y'] = y

        # Section 2.2: Constraints
        # --------------------------------------------------

        # 2.2.1: Time window constraints

        # Eq. (1)
        self._constrs[model]['1a'] = self._normal_model.addConstrs(self.E[i] <= x[i] for i in P)
        self._constrs[model]['1b'] = self._normal_model.addConstrs(x[i] <= self.L[i] for i in P)

        # Eq.(2)
        self._constrs[model]['2'] = self._normal_model.addConstrs(
            x[i] == self.E[i] + y[i] * (self.L[i] - self.E[i]) for i in P)

        # 2.2.2: Separation time constraints

        # Eq. (4)
        self._constrs[model]['4'] = self._normal_model.addConstrs(
            delta[i, j] + delta[j, i] == 1 for i in P for j in P if i != j)

        # Eq. (5)
        self._constrs[model]['5'] = self._normal_model.addConstrs(
            gamma[i, j] == gamma[j, i] for i in P for j in P if i != j)

        big_m = 0xFFFFF  # 2^20

        # Eq. (6)
        self._constrs[model]['6'] = self._normal_model.addConstrs(
            x[j] >= x[i] + self.S[i][j] * gamma[i, j] - big_m * delta[j, i] for i in P for j in P if i != j)

        # 2.2.3: Multiple runway constraints

        # Eq. (7)
        self._constrs[model]['7'] = self._normal_model.addConstrs(grb.quicksum(z[i, r] for r in R) == 1 for i in P)

        # Eq. (8)
        self._constrs[model]['8'] = self._normal_model.addConstrs(
            gamma[i, j] >= z[i, r] + z[j, r] - 1 for i in P for j in P for r in R if i < j)

        # Section 2.3.2: Linear Objective
        # --------------------------------------------------

        # Variables to capture earliness and lateness
        alpha = self._normal_model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='alpha')
        self._dvars[model]['alpha'] = alpha

        beta = self._normal_model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='beta')
        self._dvars[model]['beta'] = beta

        obj = grb.quicksum(-1 * (alpha[i] * self.g[i] + beta[i] * self.h[i]) for i in P)
        self._normal_model.setObjective(obj, grb.GRB.MAXIMIZE)
        self._objectives[ALP.Mode.normal] = obj

        # Link alpha and beta to the decision variable x:

        # Eq. (11)
        self._constrs[model]['11'] = self._normal_model.addConstrs(x[i] == self.T[i] - alpha[i] + beta[i] for i in P)

        # Eq. (12)
        self._constrs[model]['12'] = self._normal_model.addConstrs(alpha[i] <= self.T[i] - self.E[i] for i in P)

        # Eq. (13)
        self._constrs[model]['13'] = self._normal_model.addConstrs(alpha[i] >= self.T[i] - x[i] for i in P)

        # Eq. (14)
        self._constrs[model]['14'] = self._normal_model.addConstrs(beta[i] <= self.L[i] - self.T[i] for i in P)

        # Eq. (15)
        self._constrs[model]['15'] = self._normal_model.addConstrs(beta[i] >= x[i] - self.T[i] for i in P)

        self._normal_model.setParam('OutputFlag', 0)

    def _build_dynamic_model(self, model):
        """Builds a MIP for this instance.

        Refers to section 2 in Pinol & Beasley (2006).

        Beforehand, a the Gurobi model object must be initialized.
        The built model allows for adaptions needed for LP-based local
        improvement.
        """

        # Sets
        # --------------------------------------------------

        P = range(self.nr_planes)  # set of airplanes
        R = range(self.nr_runways)  # set of runways

        # Section 2.1: Decision Variables
        # --------------------------------------------------

        # 1 if aircraft i (in P) lands on runway r (in R), 0 else
        z = model.addVars(P, R, vtype=grb.GRB.BINARY, name='z')
        self._dvars[model]['z'] = z

        # 1 if aircraft i (in P) and j (in P) lands on the same runway, 0 else
        gamma = model.addVars(P, P, vtype=grb.GRB.BINARY, name='gamma')
        self._dvars[model]['gamma'] = gamma

        # 1 if aircraft i (in P) lands before aircraft j (in P)
        delta = model.addVars(P, P, vtype=grb.GRB.BINARY, name='delta')
        self._dvars[model]['delta'] = delta

        # Scheduled landing time for aircraft i (in P)
        x = model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='x')
        self._dvars[model]['x'] = x

        # Proportion of the landing time window for aircraft i (in P)
        # Includes Eq. (3)
        y = model.addVars(P, vtype=grb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y')
        self._dvars[model]['y'] = y

        # Violated amount of separation time between plane i in P and plane j in P
        # !! This variable is not directly mentioned in Pinol & Beasley (2006).
        # !! It is introduced to allow for violation of the separation time constraints
        # !! which is needed for the LP-based local improvement procedure.
        v = model.addVars(P, P, vtype=grb.GRB.CONTINUOUS, lb=0.0, name='v')
        self._dvars[model]['v'] = v

        # Section 2.2: Constraints
        # --------------------------------------------------

        # Preset decision variables
        model.addConstrs(delta[i, j] == 1 for i, j in self.V.union(self.W))

        # 2.2.1: Time window constraints

        # Eq. (1)
        self._constrs[model]['1a'] = model.addConstrs(self.E[i] <= x[i] for i in P)
        self._constrs[model]['1b'] = model.addConstrs(x[i] <= self.L[i] for i in P)

        # Eq.(2)
        self._constrs[model]['2'] = model.addConstrs(x[i] == self.E[i] + y[i] * (self.L[i] - self.E[i]) for i in P)

        # 2.2.2: Separation time constraints

        # Eq. (4)
        self._constrs[model]['4'] = model.addConstrs(delta[i, j] + delta[j, i] == 1 for i in P for j in P if i != j)

        # Eq. (5)
        self._constrs[model]['5'] = model.addConstrs(gamma[i, j] == gamma[j, i] for i in P for j in P if i != j)

        bigM = 0xFFFFF  # 2^20

        # Eq. (6)
        # Adaption according to Eq. (7) in Beasley et al. (2000)
        self._constrs[model]['6a'] = model.addConstrs(
            x[j] >= x[i] + (self.S[i][j] - v[i, j]) * gamma[i, j] for i, j in self.V)
        # Adaption according to Eq. (8) and Eq. (11) in Beasley et al. (2000)
        self._constrs[model]['6b'] = model.addConstrs(
            x[j] >= x[i] + (self.S[i][j] - v[i, j]) * gamma[i, j] - (self.L[i] + self.S[i][j] - self.E[j]) * delta[j, i]
            for i, j in self.U)

        # 2.2.3: Multiple runway constraints

        # Eq. (7)
        self._constrs[model]['7'] = model.addConstrs(grb.quicksum(z[i, r] for r in R) == 1 for i in P)

        # Eq. (8)
        self._constrs[model]['8'] = model.addConstrs(
            gamma[i, j] >= z[i, r] + z[j, r] - 1 for i in P for j in P for r in R if i < j)

        # Section 2.3.2: Linear Objective
        # --------------------------------------------------

        # Variables to capture earliness and lateness
        alpha = model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='alpha')
        self._dvars[model]['alpha'] = alpha

        beta = model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='beta')
        self._dvars[model]['beta'] = beta

        obj = grb.quicksum(-1 * (alpha[i] * self.g[i] + beta[i] * self.h[i]) for i in P)
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        self._objectives[ALP.Mode.normal] = obj

        # Link alpha and beta to the decision variable x:

        # Eq. (11)
        self._constrs[model]['11'] = model.addConstrs(x[i] == self.T[i] - alpha[i] + beta[i] for i in P)

        # Eq. (12)
        self._constrs[model]['12'] = model.addConstrs(alpha[i] <= self.T[i] - self.E[i] for i in P)

        # Eq. (13)
        self._constrs[model]['13'] = model.addConstrs(alpha[i] >= self.T[i] - x[i] for i in P)

        # Eq. (14)
        self._constrs[model]['14'] = model.addConstrs(beta[i] <= self.L[i] - self.T[i] for i in P)

        # Eq. (15)
        self._constrs[model]['15'] = model.addConstrs(beta[i] >= x[i] - self.T[i] for i in P)

        # Initialize structures for adaptions to the model

        # to fix the airplane sequence
        self._constrs[model]['fix-rw'] = []
        self._constrs[model]['fix-sq'] = []

        # to relax the separation time constraints
        min_violation_obj = grb.quicksum(-1 * (v[i, j]) for i in P for j in P)
        self._objectives[ALP.Mode.relaxed] = min_violation_obj
        self._constrs[model]['relax'] = dict()

        model.setParam('OutputFlag', 0)

    def _relax_sep_time(self, model):
        """Relaxes the separation times.
        """
        # add constraints relaxing the separation time
        v = self._dvars[model]['v']
        # maintain original sequence using a tiny separation time for equal landing times
        self._constrs[model]['relax'] = model.addConstrs(
            v[i, j] <= self.S[i][j] for i, j in self.U.union(self.V))

        model.setObjective(self._objectives[ALP.Mode.relaxed], grb.GRB.MAXIMIZE)
        model.update()

    def _enforce_sep_time(self, model):
        """Enforces the separation times.
        """
        # add constraints enforcing the separation time
        v = self._dvars[model]['v']
        self._constrs[model]['relax'] = model.addConstrs(v[i, j] == 0.0 for i, j in self.U.union(self.V))

        model.setObjective(self._objectives[ALP.Mode.normal], grb.GRB.MAXIMIZE)
        model.update()

    def _enforce_sequence(self, model, phenotype):
        """Enforces a given sequence of airplanes.

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i
        """
        z = self._dvars[model]['z']
        delta = self._dvars[model]['delta']

        runway_constrs = []
        seq_constrs = []
        for runway in phenotype:
            for i, x_i, rw_i in runway:
                runway_constrs.append(model.addConstr(z[i, rw_i] >= 1.0))
                for j, x_j, rw_j in runway:
                    # only consider (i,j) and not (j,i) and only relevant pairs
                    if i < j and (i, j) in self.U:
                        if x_i < x_j:
                            # plane 1 is landing before plane 2, i.e. delta[i,j] = 1
                            # model sets delta[j,i] = 0 due to eq. (4).
                            constr = model.addConstr(delta[i, j] >= 1.0, name='fix-sq')
                            seq_constrs.append(constr)
                        else:
                            # plane j is landing before plane i, i.e. delta[j,i] = 1
                            # model sets delta[i,j] = 0 due to eq. (4).
                            constr = model.addConstr(delta[j, i] >= 1.0, name='fix-sq')
                            seq_constrs.append(constr)

        self._constrs[model]['fix-rw'] = runway_constrs
        self._constrs[model]['fix-sq'] = seq_constrs
        model.update()

    def _clear_sequence(self, model):
        """Removes an enforced sequence from the model.
        """
        for constr in self._constrs[model]['fix-rw']:
            # allow any runway
            model.remove(constr)
        for constr in self._constrs[model]['fix-sq']:
            # allow any sequence
            model.remove(constr)
        self._constrs[model]['fix-rw'] = []
        self._constrs[model]['fix-sq'] = []
        model.update()

    def _calc_min_cost_obj(self, phenotype):
        """Calculates the objective value for the linear objective.

        Lateness and earliness are weighted with a penalty factor
        per time unit and summed up. For simplification, the objective
        value is multiplied by -1 to make it a maximization problem.

        Refers to eq. (10) in section 2.3.2 in Pinol & Beasley (2006).

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            objective_value (float): objective value for the
                linear objective and given solution
        """
        obj = 0.0
        for runway in phenotype:
            for i, x_i, rw_i in runway:
                alpha_i = max(0, self.T[i] - x_i)
                beta_i = max(0, x_i - self.T[i])
                obj += (alpha_i * self.g[i] + beta_i * self.h[i])
        return -1 * obj

    def _calc_max_util_obj(self, phenotype):
        """Calculates the objective value for the non-linear objective.

        Squared Deviations from the target landing time are multiplied
        by -1 if they are >= 0 (i.e. represent delay) and summed up.

        Refers to eq. (9) in section 2.3.1 in Pinol & Beasley (2006).

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            objective_value (float): objective value for the
                non-linear objective and given solution
        """
        obj = 0.0
        for runway in phenotype:
            for i, x_i, rw_i in runway:
                d_i = x_i - self.T[i]
                if d_i >= 0:
                    # positive delay (lateness) --> penalize
                    obj += -1 * math.pow(d_i, 2)
                else:
                    # negative delay (earliness) --> favor
                    obj += math.pow(d_i, 2)
        return obj

    def _improve_min_cost_sol(self, phenotype):
        """Locally improves a solution for the linear objective.

        Refers to section 5.8 in Pinol & Beasley (2006)

        Args:
            phenotype (list of lists with tuples): representation of
                a solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i

        Returns:
            phenotype (list of lists with tuples): representation of an
                improved solution candidate using tuples (i, x_i, rw_i) for plane i
                with landing time x_i on runway rw_i
        """
        # Solve for given sequence
        improved_phenotype = self.solve_for_optimality(phenotype=phenotype)

        if improved_phenotype is not None:
            # feasible solution found for given sequence.
            return improved_phenotype

        # Solve again with relaxed separation times
        improved_phenotype = self.solve_for_optimality(phenotype=phenotype, relax_sep_time=True)

        if improved_phenotype is not None:
            # feasible solution found for given sequence and relaxed separation times
            return improved_phenotype

        else:
            # Should not be reached
            # Every LP should be feasible when separation time constraints are relaxed
            raise ValueError("Infeasible LP.")

    def _improve_max_util_sol(self, phenotype):
        """Locally improves a solution for the non-linear objective.

        Refers to section 5.8 in Pinol & Beasley (2006)

        Requires a sorted phenotype.

        Args:
            phenotype (list of lists with ints): landing sequence
                representation for each runway using lists of plane
                numbers ordered by landing time.

        Returns:
            phenotype (list of lists with triples): improved solution
                represented by the triple form (plane number,
                landing time, runway number)
        """
        improv_phenotype = list()
        for runway in phenotype:
            new_sequence = []
            for idx, (i, x_i, rw_i) in enumerate(runway):
                if idx == 0:
                    # first plane in runway sequence
                    new_sequence.append((i, self.E[i], rw_i))
                else:
                    # find earliest possible time depending on previous planes
                    lower_bounds = [x_j + self.S[i][j] for j, x_j, rw_j in new_sequence]

                    # add earliest allowed time by current plane
                    lower_bounds.append(self.E[i])

                    # calculate earliest allowed time
                    lower_bound = max(lower_bounds)

                    # get latest allowed time
                    upper_bound = self.L[i]

                    # ensure landing time is within time window
                    # separation times may be violated
                    landing_time = min(lower_bound, upper_bound)
                    new_sequence.append((i, landing_time, rw_i))
            improv_phenotype.append(new_sequence)

        # for runway, sequence in enumerate(phenotype):
        #     for index, (plane, landing_time, r) in enumerate(sequence):
        #         if index == 0:
        #             landing_time = self.E[plane]
        #         else:
        #             earliest_times = []
        #             for pred, landing_time, runway in improv_phenotype[runway]:
        #                 earliest_times.append(landing_time + self.S[plane][pred])
        #             earliest_time = max(earliest_times) if max(earliest_times) > self.E[plane] else \
        #                 self.E[plane]
        #             landing_time = min(self.L[plane], earliest_time)
        #         improv_phenotype[runway].append((plane, landing_time, runway))
        return improv_phenotype

    def _preprocess_airplanes(self):
        """Pre-processes airplane data to improve performance.

        Refers to Section 2 in Beasley et al. (2000).

        For each pair of airplanes with i < j, it is determined
        wether the sequence is already pre-determined from the
        corresponding time window. Additionally, it is checked
        if the separation time is automatically satisfied. This
        saves a great deal of model building which is computationally
        expensive.

        Types of Sets:
        - U: the set of pairs (i, j) of planes for which we are uncertain
            whether plane i lands before plane j
        - V: the set of pairs (i, j) of planes for which i definitely lands
            before j (but for which the separation constraint is not
            automatically satisfied)
        - W: the set of pairs (i, j) of planes for which i definitely lands
            before j (and for which the separation constraint is automatically
            satisfied)

        Returns:
            sets (tuple of sets): U, V, and W
        """
        P = range(self.nr_planes)

        all_pairs = [(i, j) for i in P for j in P if i is not j]

        U, V, W = set(), set(), set()

        for i, j in all_pairs:
            # Eq. (3) in Beasley et al. (2000)
            if self.L[i] < self.E[j] and self.L[i] + self.S[i][j] <= self.E[j]:
                # i will land before j and separation time is automatically satisfied
                W.add((i, j))
            # Eq. (4) in Beasley et al. (2000)
            elif self.L[i] < self.E[j] and self.L[i] + self.S[i][j] > self.E[j]:
                # i will land before j and separation time is **not** automatically satisfied
                V.add((i, j))
            # Eq. (5) in Beasley et al. (2000)
            elif (self.E[j] <= self.E[i] <= self.L[j]) or (  # E_i lies within time window of j
                    self.E[j] <= self.L[i] <= self.L[j]) or (  # L_i lies within the time window of j
                    self.E[i] <= self.E[j] <= self.L[i]) or (  # E_j lies within the time window of i
                    self.E[i] <= self.L[j] <= self.L[i]):  # # L_j lies within the time window of i
                U.add((i, j))

        return U, V, W


def parse_file(file_name):
    """Parses a file from the OR library to a list of floats.

    The specified file should be located in the working directory. Otherwise
    the relative or absolute path must be included. The file extension is
    also mandatory.

    The exact file specification can be found in Beasley (1990).

    Args:
        file_name (str): name of the file

    Returns:
        list of floats: file content parsed to numeric values

    Raises:
        ValueError: if the file contains other elements than numeric
            values parseable to floats.
        FileNotFoundError: if the specified file cannot be located.
    """
    try:
        with open(file_name) as f:
            # get input as list of string where each item represents a row
            data = f.read()
    except FileNotFoundError:
        print("The passed file '%s' was not found." % file_name)
        raise FileNotFoundError

    # split file content after each whitespace, tab or newline
    elements = data.split()

    try:
        # parse as float values
        return [float(x) for x in elements]
    except ValueError:
        print("Not every element of the read file '%s' cannot be converted to float" % file_name)
        raise ValueError


def solve_instances(objective, instances, optimal=True):
    for data_set, runways in instances:
        try:
            input_data = parse_file("data/airland%d.txt" % data_set)
        except:
            print("Errors occured while reading data set: %d" % data_set)
            continue;
        print("################################################################")
        print("[ START      ] DATA SET=%d, RUNWAYS=%d, OBJECTIVE=%s" % (data_set, runways, objective))
        alp = ALP(data=input_data, objective=objective, nr_runways=runways)
        if optimal:
            phenotype = alp.solve_for_optimality()
        else:
            phenotype = alp.solve_heuristically()
        yield alp, phenotype
