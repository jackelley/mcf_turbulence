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
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    base_size_per_rank = N // size
    remainder_size_per_rank = N % size

    counts_N = np.full(size, base_size_per_rank, dtype=int)
    counts_N[-1] += remainder_size_per_rank
    
    displs_N = np.zeros(size, dtype=int)
    displs_N[1:] = np.cumsum(counts_N)[:-1]

    size_per_rank = counts_N[rank]
    # this rank's starting index for each time step
    start_idx_complex = displs_N[rank]

    idx = 3
    sim_dir = os.path.join(base_dir, f"ETG_sim_{idx}")
    param_path = os.path.join(sim_dir, "parameters.dat")
    n_time = 0
    if rank == 0:
        try:
            num_steps = parse_num_steps_from_parameters(param_path)
            n_time = (num_steps + 1)
        except Exception as e:
            print(f"Rank 0 failed to read parameters: {e}")
            n_time = -1 # signal error
    
    n_time = comm.bcast(n_time, root=0)

    if n_time == -1:
        if rank == 0:
            print("Aborting due to error on rank 0.")
        comm.Abort(1)

    input_filename = os.path.join(base_dir, f"ETG_sim_{idx}", "g1.dat")
    output_raw_data_filename = "/pscratch/sd/j/jackk/mcf_turbulence/par_output_data.npy"
    output_times_filename = "/pscratch/sd/j/jackk/mcf_turbulence/par_output_times.npy"

    local_times = np.empty(n_time, dtype=np.float64)
    local_data = np.empty(size_per_rank * 2 * n_time, dtype=np.float64, order="F")

    local_data_2d = local_data.reshape((size_per_rank * 2, n_time), order="F")

    with open(input_filename, "rb") as fstream:
        for t in range(n_time):
            offset = t * (2 * N + 1)  # offset is in units of float64
            time_pointer = offset * 8  # byte offset
            fstream.seek(time_pointer)
            local_times[t] = np.fromfile(fstream, dtype=np.float64, count=1)[0]

            data_pointer = (offset + 1 + (start_idx_complex * 2)) * 8  # byte offset
            fstream.seek(data_pointer)
            data_block = np.fromfile(fstream, dtype=np.float64, count = 2 * size_per_rank)

            # data_real_flat = data_block[0::2]  # 0 to the end, every 2
            # data_imag_flat = data_block[1::2]  # 1 to the end, every 2

            # cant reshape here, data is not for a full time step

            # local_data[0, t] = data_real_flat
            # local_data[1, t] = data_imag_flat

            local_data_2d[:, t] = data_block
    
    full_times = local_times  # all ranks already have this

    full_data_flat = None
    if rank == 0:
        print(f"\nRoot gathering data from all {size} processes...")
        full_data_flat = np.empty(2 * N * n_time, dtype=np.float64, order="F")

    recvcounts_float = (counts_N * 2 * n_time).astype(int)
    recvdispls_float = (displs_N * 2 * n_time).astype(int)

    comm.Gatherv(sendbuf=local_data, 
                 recvbuf=[full_data_flat, recvcounts_float, recvdispls_float, MPI.DOUBLE], 
                 root=0)
    
    if rank == 0:
        print("Gather complete. Reshaping data...")
        
        # full_data_flat is (2 * N * n_time) F-ordered
        full_data_2d = full_data_flat.reshape((2 * N, n_time), order="F")

        # de-interleave real and imaginary parts
        # this slicing creates two (N, n_time) F-ordered arrays
        data_real_flat = full_data_2d[0::2, :]
        data_imag_flat = full_data_2d[1::2, :]

        # combine into complex array (N, n_time)
        # this is the 'data_X' from your example
        data_X = data_real_flat + 1j * data_imag_flat

        # reshape to final physical + time dimensions
        # data_size = (nx0, nz0, nv0, nw0)
        data = data_X.reshape((*data_size, n_time), order="F")

        print(f"Final data shape (data): {data.shape}")
        print(f"Final data_X shape (data_X): {data_X.shape}")
        
        np.save(output_raw_data_filename, data_X)
        np.save(output_times_filename, local_times)
        print(f"Successfully processed {n_time} time steps.")
        print(f"[OK] sim {idx}: wrote {n_time} time slices")

if __name__ == "__main__":
    main()
