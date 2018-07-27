from alp import ALP, solve_instances
import datetime as dt
import openpyxl as xl

###############################################################################
# CONFIGURATION ###############################################################
###############################################################################

# Where results are stored
SOLUTION_FILE = "results.xlsx"

# Which instances are run?
DATA_SETS = [1, 2, 3]  # add airland files in [1, ..., 13]
RUNWAYS = [1, 2]  # add runways

# overwrite previous choice to solve specific instances
INSTANCES = [(9, 1)]  # tuple of (data set, nr runways)

# Which objective is set?
# OBJECTIVE = ALP.Objective.min_cost  # linear
OBJECTIVE = ALP.Objective.max_util  # non-linear

# Which method is used?
APPLY_SOLVER = False  # True: Gurobi, False: Bionomic Algorithm

###############################################################################
# NO CHANGES NECESSARY AFTER THIS POINT #######################################
###############################################################################

# Main method
if __name__ == '__main__':
    # Setup instances to solve
    if not INSTANCES:
        for data_set in DATA_SETS:
            for runway in RUNWAYS:
                INSTANCES.append((data_set, runway))

    if APPLY_SOLVER:
        sheet_name = "solver"
    else:
        sheet_name = "heuristic"

    # Solve instances
    for instance, phenotype in solve_instances(objective=OBJECTIVE, instances=INSTANCES, optimal=APPLY_SOLVER):
        # Collect solution info
        obj_val = instance.calc_obj_value(phenotype)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        info = [dt.datetime.now(), str(OBJECTIVE), instance.nr_planes, instance.nr_runways, obj_val, str(phenotype)]
        # Write to results excel file
        wb = xl.load_workbook(SOLUTION_FILE)
        sht = wb[sheet_name]
        sht.append(info)
        wb.save(SOLUTION_FILE)
