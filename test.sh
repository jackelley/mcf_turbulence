#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:05:00

module load cray-mpich/8.1.30
module load python
srun -n 4 python par_read_g1.py
srun python plot_g1_phase_space.py
srun python plot_g1_phase_space_par.py

