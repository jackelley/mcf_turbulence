#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:10:00

module load cray-mpich/8.1.30
module load python
srun -n 4 python /global/homes/j/jackk/repos/mcf_turbulence/src/par_read_2.py
srun python /global/homes/j/jackk/repos/mcf_turbulence/src/plot_g1_phase_space_par.py
srun python /global/homes/j/jackk/repos/mcf_turbulence/src/plot_g1_phase_space.py

