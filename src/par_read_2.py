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
    start_idx_complex = displs_N[rank]

    idx = 4
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

    local_sum_D = 0.0

    # float buffer to read the interleaved data
    local_data_block_float = np.empty(size_per_rank * 2, dtype=np.float64)
    # complex buffer to hold the transformed data (this is what we send)
    local_data_block_complex = np.empty(size_per_rank, dtype=np.complex128)
    
    # allocate final arrays
    full_times = None
    data_X = None
    
    if rank == 0:
        print(f"Rank 0 allocating memory for {n_time} time steps...")
        full_times = np.empty(n_time, dtype=np.float64)
        data_X = np.empty((N, n_time), dtype=np.complex128, order="F")

    recvcounts_complex = counts_N.astype(int)
    recvdispls_complex = displs_N.astype(int)

    with open(input_filename, "rb") as fstream:
        for t in range(n_time):
            if (t % 2000 == 0 or t == n_time - 1):
                print(f"Processing time step {t}/{n_time-1} on rank {rank}...")
                
            offset = t * (2 * N + 1)  # offset is in units of float64
            
            time_pointer = offset * 8  # byte offset
            fstream.seek(time_pointer)
            time_val = np.fromfile(fstream, dtype=np.float64, count=1)[0]
            if rank == 0:
                full_times[t] = time_val

            data_pointer = (offset + 1 + (start_idx_complex * 2)) * 8  # byte offset
            fstream.seek(data_pointer)
            
            fstream.readinto(local_data_block_float.data)

            local_real = local_data_block_float[0::2]
            local_imag = local_data_block_float[1::2]
            local_data_block_complex[:] = local_real + 1j * local_imag

            # checksum
            local_sum_D += np.sum(np.abs(local_data_block_complex)**2)
            
            # we gather directly into the correct column of the final complex array
            recvbuf_t = [data_X[:, t], recvcounts_complex, recvdispls_complex, MPI.COMPLEX16] if rank == 0 else None

            # send the complex buffer
            comm.Gatherv(sendbuf=local_data_block_complex, 
                         recvbuf=recvbuf_t, 
                         root=0)
    
    total_D_parallel = comm.reduce(local_sum_D, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"PARALLEL Checksum: {total_D_parallel}")

        print("All time steps gathered. Reshaping and saving...")
        
        # reshape to final physical + time dimensions
        data = data_X.reshape((*data_size, n_time), order="F")

        print(f"Final data shape (data): {data.shape}")
        print(f"Final data_X shape (data_X): {data_X.shape}")
        
        np.save(output_raw_data_filename, data_X)
        np.save(output_times_filename, full_times)
        print(f"Successfully processed {n_time} time steps.")
        print(f"[OK] wrote {n_time} time slices")

if __name__ == "__main__":
    main()
