#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:10:00

module load cray-mpich/8.1.30
module load python
srun python src/read_g1.py
srun -n 4 python src/read_parallel.py \
    --sim_index 1 \
    --base_dir "/global/cfs/cdirs/m3586/parametric_ETG_ROM/training_folders" \
    --output_dir "/pscratch/sd/j/jackk/mcf_turbulence/" \
    --grid 3 168 32 8 \
    --precision float64
srun python /global/homes/j/jackk/repos/mcf_turbulence/src/plot_g1_phase_space_par.py
srun python /global/homes/j/jackk/repos/mcf_turbulence/src/plot_g1_phase_space.py

