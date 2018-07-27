# A Bionomic Algorithm for the Aircraft Landing Problem
## Info

This project contains a Bionomic Algorithm and a Mixed-integer Linear Program for the Aircraft Landing Problem.

Authors: Janis Juppe & Robert zur Bonsen

## Problem Setting
In this project we implemented a Bionomic Algorithm for the Aircraft Landing Problem in a seminar at TUM. The seminar was about Airport Operations Management. A relevant problem many airports face is to determine the landing sequence and scheduling of incoming airlanes on the available runways while satisfying separation time constraints due to air turbulences. This becomes especially challenging with increasing air traffic. 

## Solution
We implemented a Bionomic Algorithm (BA) as presented in Pinol & Beasley (2006) to solve the problem. The BA is based on the principles of Genetic Algorithms but employs more structured procedures. As such, a local improvement step aims to improve every individual by using a MILP model. Furthermore, parent selection uses an elaborate graph representation to create parent sets of high diversity. 

## Installation

This project requires Python >3.6. Install required packages by running the following command in the root directory:

`pip install -r requirements.txt`

To solve the MILP we use Gurobipy v. 8.0.0.

---

## Usage

Run the main method contained in `main.py` to solve ALP instances. All necessary parameters can be configured there.

Results are stored in results.xlsx by default. For advanced monitoring, set `MONITORING` to `True`in `monitoring.py`.

To collect information about computation time, use the decorator `@timethis` from `helpers.py` for any method. The
result will be printed on the console.
