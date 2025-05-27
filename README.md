# Parallel Image Convolution

This project demonstrates high-performance Gaussian blur convolution on images using a hybrid parallel approach combining **OpenMP** (shared memory, multi-core parallelism) and **MPI** (distributed memory, multi-node parallelism). The implementation is in C++ and leverages `stb_image` and `stb_image_write` for image I/O.

## Features

- Efficient Gaussian blur convolution
- Hybrid parallelism: **MPI** for distributing work across processes/nodes, **OpenMP** for accelerating inner loops on each process
- Scalable performance across cores and nodes

## Performance Comparison

The following execution times were measured for a standard image convolution task:

| Configuration                                     | Execution Time (Seconds) |
|---------------------------------------------------|-------------------------|
| Without OpenMP (serial)                           | 432.363                 |
| Without MPI (OpenMP only, 2 threads)              | 95.836                  |
| Hybrid (4 MPI processes × 2 OpenMP threads = 8 threads) | 82.6                    |

### Execution Time Comparison

![results_comparison](https://github.com/user-attachments/assets/89c16505-0b4a-4677-870b-381e1a15455c)

- **75% Faster with OpenMP**
- **8 Threads:** 4 MPI Processes × 2 OpenMP Threads

By combining OpenMP and MPI, the hybrid system achieves significant time savings and scalability, enabling efficient parallel image processing.

## How It Works

- **OpenMP** accelerates the nested convolution loops within each MPI process.
- **MPI** distributes image chunks among processes and manages halo (border) communication.
- The hybrid approach results in both intra-node (multi-core) and inter-node (multi-process) acceleration.

## Usage Example

Set the number of OpenMP threads and run with MPI:

```bash
set OMP_NUM_THREADS=2
mpiexec -n 4 ./Parallel_Image_Convolution2.exe input.png output.png
```
- This launches 4 MPI processes, each using 2 OpenMP threads, for a total of 8 threads.

## Requirements

- **OpenMP**: Enables multi-threading within each process ([Download Guide](https://www.openmp.org/resources/openmp-compilers-tools/))
- **MPI**: Enables multi-process distributed computation ([Download MPICH](https://www.mpich.org/downloads/) or [OpenMPI](https://www.open-mpi.org/software/))
- **stb_image.h**: Header-only image loader ([Download](https://github.com/nothings/stb/blob/master/stb_image.h))
- **stb_image_write.h**: Header-only image writer ([Download](https://github.com/nothings/stb/blob/master/stb_image_write.h))

Ensure that `stb_image.h` and `stb_image_write.h` are included in your project directory.

## Building

1. Install an MPI implementation (e.g., MPICH or OpenMPI) and a C++ compiler supporting OpenMP.
2. Place `stb_image.h` and `stb_image_write.h` in your source directory.
3. Compile with MPI and OpenMP flags, e.g.:
   ```bash
   mpicxx -fopenmp -o Parallel_Image_Convolution2.exe main.cpp
   ```

## References

- [OpenMP Official Site](https://www.openmp.org/)
- [MPI Standard](https://www.mpi-forum.org/)
- [stb Image Library](https://github.com/nothings/stb)

---
Results are made on windows laptop running intel core i5-10300H with 4 cores and 8 threads.<br>
**Your results may vary depending on your CPU**
