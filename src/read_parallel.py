import numpy as np
import os
import re
import argparse
from mpi4py import MPI
from typing import Tuple, Optional, Any

# Helper Functions (Data/File Parsing)

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


def distribute_work(
    N: int, rank: int, size: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calculates the workload (counts and displacements) for MPI ranks.
    
    Distributes N items among 'size' ranks.
    
    Returns:
        - counts (np.ndarray): Array of item counts for each rank.
        - displs (np.ndarray): Array of item start displacements for each rank.
        - local_size (int): The item count for the current rank.
        - local_start (int): The item start index for the current rank.
    """
    base_size = N // size
    remainder = N % size
    
    # dump remainder on the last rank
    counts = np.full(size, base_size, dtype=int)
    counts[-1] += remainder
    
    displs = np.zeros(size, dtype=int)
    displs[1:] = np.cumsum(counts)[:-1]
    
    local_size = counts[rank]
    local_start = displs[rank]
    
    return counts, displs, local_size, local_start

# data reading function

def read_distribution_parallel(
    comm: MPI.Comm,
    input_filename: str,
    param_path: str,
    grid_size: Tuple[int, ...],
    precision: type = np.float64,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reads a the dataset in parallel, with each rank taking a slice of each timestep
    
    Args:
        comm: The MPI.Comm object.
        input_filename: Path to the binary data file (e.g., "g1.dat").
        param_path: Path to the "parameters.dat" file to find n_time.
        grid_size: A tuple of the grid dimensions, e.g., (nx, nz, nv, nw).
        precision: The floating-point precision of the data on disk 
                     (e.g., np.float64 or np.float32).

    Returns:
        A tuple (data, times) on rank 0, or (None, None) on other ranks.
        - data: Reshaped complex data array with shape (*grid_size, n_time).
        - times: Array of time values.
    """
    rank = comm.rank
    size = comm.size
    
    # determine datatypes
    if precision == np.float64:
        complex_dtype = np.complex128
        mpi_complex_dtype = MPI.COMPLEX16
        item_size = 8
    elif precision == np.float32:
        complex_dtype = np.complex64
        mpi_complex_dtype = MPI.COMPLEX8
        item_size = 4
    else:
        raise ValueError("precision must be np.float64 or np.float32")

    N = np.prod(grid_size)

    # distribute indices to ranks
    counts_N, displs_N, size_per_rank, start_idx_complex = (
        distribute_work(N, rank, size)
    )

    # rank 0 gets the number of timesteps
    n_time = 0
    if rank == 0:
        print(f"Rank 0 parsing parameters from {param_path}")
        try:
            num_steps = parse_num_steps_from_parameters(param_path)
            n_time = num_steps + 1
        except Exception as e:
            print(f"Rank 0 failed to read parameters: {e}")
            n_time = -1  # signal error
    
    n_time = comm.bcast(n_time, root=0)

    if n_time <= 0:
        if rank == 0:
            print("Aborting due to error in reading time steps.")
        comm.Abort(1)

    # allocate buffers
    local_sum_Qstar_Q = 0.0
    
    # float buffer to read the interleaved data
    local_data_block_float = np.empty(size_per_rank * 2, dtype=precision)
    # complex buffer to hold the transformed data (this is what we send)
    local_data_block_complex = np.empty(size_per_rank, dtype=complex_dtype)
    
    # allocate final arrays on root
    full_times = None
    data_X = None
    
    if rank == 0:
        print(f"Rank 0 allocating memory for {n_time} time steps...")
        # time value is always float64 in the file format
        full_times = np.empty(n_time, dtype=np.float64) 
        data_X = np.empty((N, n_time), dtype=complex_dtype, order="F")

    recvcounts_complex = counts_N.astype(int)
    recvdispls_complex = displs_N.astype(int)

    # open file and loop over timesteps
    if rank == 0:
        print(f"Opening file {input_filename} to read {n_time} steps...")
        
    with open(input_filename, "rb") as fstream:
        for t in range(n_time):
            if rank == 0 and (t % 2000 == 0 or t == n_time - 1):
                print(f"Processing time step {t}/{n_time-1}...")
            
            # read time on rank 0
            # offset is in units of *items* (floats)
            offset_items = t * (2 * N + 1)
            if rank == 0:
                # time is float64, so its item size is 8
                time_pointer_bytes = offset_items * 8 
                fstream.seek(time_pointer_bytes)
                time_val = np.fromfile(fstream, dtype=np.float64, count=1)[0]
                full_times[t] = time_val

            # all ranks seek to their respective data blocks
            data_pointer_bytes = (offset_items + 1 + (start_idx_complex * 2)) * item_size
            
            fstream.seek(data_pointer_bytes)
            fstream.readinto(local_data_block_float.data)

            # de-interleave data
            local_real = local_data_block_float[0::2]
            local_imag = local_data_block_float[1::2]
            local_data_block_complex[:] = local_real + 1j * local_imag

            # checksum Q^* Q
            local_sum_Qstar_Q += np.vdot(
                local_data_block_complex, local_data_block_complex
            ).real
            
            # gather data to root
            recvbuf_t = (
                [data_X[:, t], recvcounts_complex, recvdispls_complex, mpi_complex_dtype] 
                if rank == 0 
                else None
            )

            comm.Gatherv(sendbuf=local_data_block_complex, 
                         recvbuf=recvbuf_t, 
                         root=0)
    
    # finalize and return
    total_Qstar_Q_parallel = comm.reduce(local_sum_Qstar_Q, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"All time steps gathered.")
        print(f"PARALLEL Checksum (Sum(Q^H * Q)): {total_Qstar_Q_parallel}")
        
        # reshape to final physical + time dimensions
        data_final = data_X.reshape((*grid_size, n_time), order="F")
        print(f"Final data shape: {data_final.shape}")
        
        return data_final, full_times
    
    return None, None

def parse_arguments():
    """
    Parses command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Parallel reader for large plasma datasets."
    )
    
    parser.add_argument(
        "-i", "--sim_index",
        type=int,
        required=True,
        help="Simulation index (e.g., 4)."
    )
    
    parser.add_argument(
        "-b", "--base_dir",
        type=str,
        required=True,
        help="Base directory containing simulation folders (e.g., /path/to/training_folders)."
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output .npy files."
    )
    
    parser.add_argument(
        "-g", "--grid",
        nargs='+',
        type=int,
        required=True,
        help="Grid dimensions, e.g., --grid nx0 nz0 nv0 nw0"
    )
    
    parser.add_argument(
        "-p", "--precision",
        type=str,
        choices=["float64", "float32"],
        default="float64",
        help="Precision of the data file (default: float64)."
    )
    
    return parser.parse_args()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    args = None
    if rank == 0:
        args = parse_arguments()
        
    # broadcast the parsed arguments object to all ranks
    args = comm.bcast(args, root=0)
    
    # all ranks configure their settings based on broadcasted args
    idx = args.sim_index
    grid_size = tuple(args.grid)
    data_precision = np.float64 if args.precision == "float64" else np.float32

    # define file paths
    sim_dir = os.path.join(args.base_dir, f"ETG_sim_{idx}")
    param_path = os.path.join(sim_dir, "parameters.dat")
    input_filename = os.path.join(sim_dir, "g1.dat")
    
    # define output file paths
    # ensure output directory exists (only rank 0 needs to write)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
    output_raw_data_filename = os.path.join(args.output_dir, f"par_output_data_sim{idx}.npy")
    output_times_filename = os.path.join(args.output_dir, f"par_output_times_sim{idx}.npy")
    
    if rank == 0:
        print("--- Configuration ---")
        print(f"Sim Index:   {idx}")
        print(f"Sim Dir:     {sim_dir}")
        print(f"Grid Size:   {grid_size}")
        print(f"Precision:   {args.precision}")
        print(f"Output Dir:  {args.output_dir}")
        print("---------------------")

    # do reading
    data, times = read_distribution_parallel(
        comm,
        input_filename,
        param_path,
        grid_size,
        precision=data_precision
    )
    
    # save data
    if rank == 0:
        if data is not None and times is not None:
            print(f"Successfully read data. Shape: {data.shape}")
            
            # reshape back to (N, n_time) for saving
            n_time = data.shape[-1]
            N = np.prod(data.shape[:-1])
            data_X_to_save = data.reshape(N, n_time, order="F")

            print(f"Saving data_X with shape {data_X_to_save.shape} to {output_raw_data_filename}")
            np.save(output_raw_data_filename, data_X_to_save)
            
            print(f"Saving times with shape {times.shape} to {output_times_filename}")
            np.save(output_times_filename, times)
            
            print(f"[OK] wrote {n_time} time slices for sim {idx}")
        else:
            print("Error: Rank 0 did not receive data.")

if __name__ == "__main__":
    main()