# ScaleTucker

ScaleTucker: A scalable Tucker decomposition method with storage-based operations

ScaleTucker is a scalable and robust Tucker decomposition system designed for billion-scale sparse tensors on a single commodity machine.
It divides the input tensor into appropriately sized chunks and updates the factor matrices chunk by chunk with storage-based out-of-core operations, while integrating advanced vectorization with Advanced Vector Extensions 512 (AVX-512) instructions to enhance computational performance.

Unlike conventional in-memory CPU, GPU, and distributed approaches that suffer from intermediate data explosion and memory overflow, ScaleTucker provides a unified co-design of partitioning, AVX-512-vectorized computation, and asynchronous I/O scheduling, enabling reliable large-scale tensor decomposition on a single CPU-storage node under a strict memory budget.

> **Source code availability.** The paper describing ScaleTucker is currently under review. Until the paper is accepted, this repository provides **prebuilt binaries only**; the full source code (headers and implementation files) will be released here **after the paper is accepted**.

## Features

- **Storage-Based Out-of-Core Processing**: Handles tensors far larger than main memory by storing tensor chunks as fixed-length records on storage and caching them under a user-defined memory budget (`-m`).
- **Chunk-Based Computation via Tensor Partitioning**: The input tensor is partitioned into chunks, and factor matrices are updated chunk by chunk with row-wise update rules.
- **Partitioning Optimization**: Automatically selects the partition parameters by a cost model that accounts for I/O, computation, factor traffic, and scheduling overhead; manual override via `-p`.
- **Asynchronous I/O–Computation Pipeline**: Asynchronous pipeline overlaps chunk loading with delta/factor computation, hiding storage latency.
- **AVX-512 Vectorization**: SIMD kernels for the dominant delta and reconstruction computations, combined with OpenMP multi-threading

## License

This project is licensed under the terms of the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details.

### Disclaimer

**THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.** EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

## Prerequisites

- **Linux on x86-64** (the AVX binaries require an AVX-512-capable CPU).
- **Boost runtime libraries**: `boost_program_options`, `boost_filesystem`, `boost_system`.
- **OpenMP runtime**: libgomp (gxx builds) or Intel OpenMP (icpx builds).

## Getting the Binaries

The source code is withheld until the paper is accepted (see the note above), so the repository cannot be built from source yet. Instead, prebuilt executables are provided in `bin/`:

| Binary                 | Compiler          | Kernels        |
| ---------------------- | ----------------- | -------------- |
| `ScaleTucker-gxx`      | g++               | Scalar, OpenMP |
| `ScaleTucker-AVX-gxx`  | g++               | AVX-512        |
| `ScaleTucker-icpx`     | Intel oneAPI icpx | Scalar, OpenMP |
| `ScaleTucker-AVX-icpx` | Intel oneAPI icpx | AVX-512        |

Before running, adjust the intermediate-file output paths in `config.properties` for your environment. Once the source code is released, the four targets above can be rebuilt with `make gxx_build | avx_gxx_build | icpx_build | avx_icpx_build`.

## Usage

Run the executable with the required arguments.

```bash
./ScaleTucker-AVX-icpx -i <input_file> -o <order> [options]
```

### Command Line Options

| Option              | Short  | Description                                                                                                                                 | Default            |
| ------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `--help`          | `-h` | Display help menu.                                                                                                                          | -                  |
| `--input`         | `-i` | Path to the input tensor file.                                                                                                              | **Required** |
| `--order`         | `-o` | Order (number of modes) of the tensor.                                                                                                      | **Required** |
| `--rank`          | `-r` | Tucker rank for the decomposition.                                                                                                          | 10                 |
| `--num_threads`   | `-t` | Number of CPU threads.                                                                                                                      | 1                  |
| `--partition`     | `-p` | Manual partition parameters, comma-separated with one value per mode (e.g.,`-p 2,2,4`). Omit to let the cost-model optimizer choose them. | Auto (cost model)  |
| `--memory`        | `-m` | Host memory budget for the chunk cache plus resident decomposition data (suffix K/M/G/T; a bare number means GiB).                          | 128G               |
| `--max_iters`     | `-x` | Maximum number of ALS iterations.                                                                                                           | 1                  |
| `--no_early_stop` | -      | Disable the convergence early stop (fit change ≤ 0.001); run exactly`--max_iters` iterations.                                            | False              |
| `--orthogonalize` | -      | After the ALS loop, orthogonalize all factor matrices by thin QR and fold the R factors into the core tensor, then re-evaluate the fit.     | False              |

### Example

```bash
./ScaleTucker-AVX-icpx -i ~/datasets/nell-2.tns -o 3 -r 10 -t 64 -m 128G -x 30 --no_early_stop -p 2,2,2
```

This command runs Tucker decomposition on `nell-2.tns` (order 3) with rank 10, using 64 threads, a 128 GB host memory budget, 30 fixed iterations, and a 2×2×2 manual partitioning.

## Input Format

The input file should be in a coordinate format, where each line represents a non-zero element:

```
<index_1> <index_2> ... <index_N> <value>

1    1    1    4.0
1    2    1    5.5
2    1    1    3.2
2    2    1    2.8
3    2    1    7.3
1    1    2    1.1
1    2    2    6.8
2    1    2    2.9
2    2    2    4.4
...
```

- Indices are 1-based.
- Indices and value are tab- or space-separated.

Real-world tensor datasets are available in `scripts/datasets.sh`. For more datasets, refer to [FROSTT](http://frostt.io/).

## Comparison Methods

The methods compared in the paper are available at the following repositories:

| Method    | Repository                                                           |
| --------- | -------------------------------------------------------------------- |
| P-Tucker  | https://github.com/sejoonoh/P-Tucker                                 |
| S-HOT     | https://github.com/jinohoh/WSDM17_shot                               |
| VeST      | https://github.com/leesael/VeST                                      |
| FTcom     | https://github.com/donalee/ftcom                                     |
| GTA       | https://github.com/sejoonoh/GTA-Tensor                               |
| GPUTucker | https://github.com/DevelopNumericalLibraryForSupercomputer/GPUTucker |

## Directory Structure

- `bin/`: Prebuilt executables.
- `config.properties`: Paths for the intermediate files ScaleTucker writes at runtime.
- `scripts/`: Dataset download script (`datasets.sh`).
- `test/`: Deterministic smoke-test input (`tiny.tns`).
- `baselines/`: Binaries of the comparison methods used in the experiments.

The source tree (`include/`, `source/`, `main.cpp`, `Makefile`, and the bundled Eigen in `lib/`) will be added when the source code is released after paper acceptance.
