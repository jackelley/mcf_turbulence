import numpy as np
import os
import re
from mpi4py import MPI

nx0, nky0, nz0, nv0, nw0 = 3, 1, 168, 32, 8  # n_kx x n_z x n_v x n_mu grid size
data_size = (nx0, nz0, nv0, nw0)
N = np.prod(data_size)

base_dir = "/global/cfs/cdirs/m3586/parametric_ETG_ROM/training_folders"


def parse_num_steps_from_parameters(param_path: str) -> int:
    """
    Search parameters.dat for a line containing 'number of computed time steps'
    and return the integer found on that line. Flexible to spaces, '=', ':'.
    """
    with open(param_path, "r") as f:
        for line in f:
            if "number of computed time steps" in line.lower():
                # Extract the last integer on the line
                m = re.findall(r"(-?\d+)", line)
                if not m:
                    raise ValueError(
                        f"Found the key line but no integer: {line.strip()}"
                    )
                return int(m[-1])
    raise FileNotFoundError(
        f"Could not find 'number of computed time steps' in {param_path}"
    )


def main():
    # --- MPI Initialization ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Select a single simulation to process in parallel ---
    sim_idx = 1
    sim_dir = os.path.join(base_dir, f"ETG_sim_{sim_idx}")
    param_path = os.path.join(sim_dir, "parameters.dat")
    input_filename = os.path.join(sim_dir, "g1.dat")

    # error handling
    try:
        num_steps = parse_num_steps_from_parameters(param_path)
        n_time = (num_steps + 1)
        if rank == 0:
            print(f"Parsed 'num_steps' = {num_steps}")
            print(f"Calculated 'n_time' = {n_time}")
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"[ERROR] Could not determine n_time: {e}")
        comm.Abort()
        return

    # split up workload
    all_time_indices = np.arange(0, num_steps + 1, 10)
    n_time = len(all_time_indices)

    local_time_indices = np.array_split(all_time_indices, size)[rank]

    num_local_times = len(local_time_indices)
    local_times = np.empty(num_local_times, dtype=np.float64)
    local_data = np.empty(
        data_size + (num_local_times,), dtype=np.complex128, order="F"
    )

    with open(input_filename, "rb") as fstream:
        # enumerate to get local index (0, 1, 2...) for storing data
        for local_t, global_t in enumerate(local_time_indices):
            offset = global_t * (2 * N + 1)

            # read time
            fstream.seek(offset * 8)
            local_times[local_t] = np.fromfile(fstream, dtype=np.float64, count=1)[0]

            # read data
            fstream.seek((offset + 1) * 8)
            data_block = np.fromfile(fstream, dtype=np.float64, count=2 * N)

            # reshape and combine into complex numbers
            data_real = data_block[0::2].reshape(data_size, order="F")
            data_imag = data_block[1::2].reshape(data_size, order="F")
            local_data[..., local_t] = data_real + 1j * data_imag

    # reshape
    local_data_X = local_data.reshape((N, num_local_times), order="F")

    # form buffers on rank 0
    counts = comm.gather(num_local_times, root=0)
    full_data_X = None
    full_times = None

    if rank == 0:
        print(f"\nRoot gathering data from all {size} processes...")

        # Prepare buffers to receive all the data
        full_times = np.empty(n_time, dtype=np.float64)
        full_data_X = np.empty((N, n_time), dtype=np.complex128, order="F")

        data_counts = [c * N for c in counts]
    else:
        full_times = None
        full_data_X = None
        counts = None
        data_counts = None

    comm.Gatherv(sendbuf=local_times, recvbuf=(full_times, counts), root=0)
    comm.Gatherv(sendbuf=local_data_X, recvbuf=(full_data_X, data_counts), root=0)

    # --- Write Output ---
    if rank == 0:
        # output_raw_data_filename = os.path.join(sim_dir, "g1_mpi_data_copy.npy")
        # output_times_filename = os.path.join(sim_dir, "g1_mpi_times_copy.npy")
        
        # output_raw_data_filename = os.path.join(
        #     "/global/homes/j/jackk/repos/mcf_turbulence/example_data", "g1_mpi_data.npy"
        # )
        # output_times_filename = os.path.join(
        #     "/global/homes/j/jackk/repos/mcf_turbulence/example_data",
        #     "g1_mpi_times.npy",
        # )

        np.save("/pscratch/sd/j/jackk/mcf_turbulence/g1_mpi_data.npy", full_data_X)
        np.save("/pscratch/sd/j/jackk/mcf_turbulence/g1_mpi_times.npy", full_times)

        print(
            f"[OK] Root process successfully gathered and processed {n_time} time slices."
        )
        print(f"Final data shape: {full_data_X.shape}")
        print(f"Final times shape: {full_times.shape}")


if __name__ == "__main__":
    main()
