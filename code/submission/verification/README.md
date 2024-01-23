The code relies on VenMAS, a tool for verification of closed-loop systems with neural network components, reproduced here.
 
The files specific to the submission can be found in
 * resources/guarding/
 * test/pnis_guarding.py
 * test/run_guarding_experiments.sh

To reproduce the experiments, run 

    ./run_guarding_experiments.sh

In order to run this code, you will need to have installed 

* Python >= 3.5
* Gurobi 10.0

To be able to use Gurobi, you will need to have a Gurobi license installed on your machine.

