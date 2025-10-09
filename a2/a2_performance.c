/*
 * Assignment 2: 2D Convolution Performance Analysis
 * Compares Sequential, OpenMP, MPI, and Hybrid (MPI+OpenMP) implementations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

// Read matrix from file
int read_matrix(const char *filename, float **data, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return 0;
    }
    
    *data = (float*)malloc((*rows) * (*cols) * sizeof(float));
    for (int i = 0; i < (*rows) * (*cols); i++) {
        if (fscanf(fp, "%f", &(*data)[i]) != 1) {
            fclose(fp);
            free(*data);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

// Write matrix to file
void write_matrix(const char *filename, float *data, int rows, int cols) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.3f", data[i * cols + j]);
            if (j < cols - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Generate random matrix in memory
float* generate_random_matrix(int rows, int cols, int seed) {
    srand(seed);
    float *data = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
    return data;
}

// Function to allocate 2D array with contiguous memory (optimized for cache performance)
float **allocate_2d_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float *));
    if (!array) return NULL;
    
    // Allocate contiguous block for better cache locality and MPI communication
    array[0] = (float *)malloc(rows * cols * sizeof(float));
    if (!array[0]) {
        free(array);
        return NULL;
    }
    
    // Set up row pointers to point into the contiguous block
    for (int i = 1; i < rows; i++) {
        array[i] = array[0] + i * cols;
    }
    return array;
}

// Function to free 2D array
void free_2d_array(float **array) {
    if (array) {
        free(array[0]);  // Free the contiguous data block
        free(array);     // Free the row pointers
    }
}

// Function to generate random array using 2D interface
void generate_random_array(float **array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

// Memory-efficient matrix allocation for MPI processes
// Only allocates memory needed for this process's portion
float* allocate_mpi_local_matrix(int total_rows, int total_cols, int rank, int size, int *local_rows) {
    int rows_per_proc = total_rows / size;
    int extra = total_rows % size;
    
    *local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    
    return (float*)malloc((*local_rows) * total_cols * sizeof(float));
}

// NUMA-aware allocation for hybrid MPI+OpenMP
// Allocates memory on the local NUMA node for better thread access
float* allocate_numa_aware_matrix(int rows, int cols) {
    float *data = NULL;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Allocate on the NUMA node where the master thread runs
            data = (float*)malloc(rows * cols * sizeof(float));
        }
    }
    
    // Touch memory pages to ensure they're allocated on the correct NUMA node
    if (data) {
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) {
            data[i] = 0.0f;
        }
    }
    
    return data;
}

// Sequential 2D Convolution with SAME padding and strides
void conv2d_sequential(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW, float *output, int *outH, int *outW) {
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;
    
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;
    
    for (int out_i = 0; out_i < *outH; out_i++) {
        for (int out_j = 0; out_j < *outW; out_j++) {
            float sum = 0.0f;
            int center_i = out_i * sH;
            int center_j = out_j * sW;
            
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int in_i = center_i + ki - pad_h;
                    int in_j = center_j + kj - pad_w;
                    
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        sum += f[in_i * W + in_j] * g[ki * kW + kj];
                    }
                }
            }
            output[out_i * (*outW) + out_j] = sum;
        }
    }
}

// OpenMP 2D Convolution with SAME padding and strides
void conv2d_openmp(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW, float *output, int *outH, int *outW) {
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;
    
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;

    #pragma omp parallel for schedule(guided) proc_bind(close) collapse(2)
    for (int out_i = 0; out_i < *outH; out_i++) {
        for (int out_j = 0; out_j < *outW; out_j++) {
            float sum = 0.0f;
            int center_i = out_i * sH;
            int center_j = out_j * sW;
            
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int in_i = center_i + ki - pad_h;
                    int in_j = center_j + kj - pad_w;
                    
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        sum += f[in_i * W + in_j] * g[ki * kW + kj];
                    }
                }
            }
            output[out_i * (*outW) + out_j] = sum;
        }
    }
}

// MPI 2D Convolution with SAME padding and strides
void conv2d_mpi(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW, float *output, int *outH, int *outW, int rank, int size) {
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;
    
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;
    
    int rows_per_proc = (*outH) / size;
    int extra = (*outH) % size;
    int start_row = rank * rows_per_proc + (rank < extra ? rank : extra);
    int end_row = start_row + rows_per_proc + (rank < extra ? 1 : 0);
    int my_rows = end_row - start_row;
    
    float *local_output = (float*)malloc(my_rows * (*outW) * sizeof(float));
    
    for (int out_i = 0; out_i < my_rows; out_i++) {
        int global_out_i = start_row + out_i;
        for (int out_j = 0; out_j < *outW; out_j++) {
            float sum = 0.0f;
            int center_i = global_out_i * sH;
            int center_j = out_j * sW;
            
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int in_i = center_i + ki - pad_h;
                    int in_j = center_j + kj - pad_w;
                    
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        sum += f[in_i * W + in_j] * g[ki * kW + kj];
                    }
                }
            }
            local_output[out_i * (*outW) + out_j] = sum;
        }
    }
    
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_proc + (r < extra ? r : extra);
            int r_rows = rows_per_proc + (r < extra ? 1 : 0);
            recvcounts[r] = r_rows * (*outW);
            displs[r] = r_start * (*outW);
        }
    }
    
    MPI_Gatherv(local_output, my_rows * (*outW), MPI_FLOAT,
                output, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    free(local_output);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Hybrid MPI+OpenMP 2D Convolution with SAME padding and strides
void conv2d_hybrid(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW, float *output, int *outH, int *outW, int rank, int size) {
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;
    
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;
    
    int rows_per_proc = (*outH) / size;
    int extra = (*outH) % size;
    int start_row = rank * rows_per_proc + (rank < extra ? rank : extra);
    int end_row = start_row + rows_per_proc + (rank < extra ? 1 : 0);
    int my_rows = end_row - start_row;
    
    float *local_output = (float*)malloc(my_rows * (*outW) * sizeof(float));
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int out_i = 0; out_i < my_rows; out_i++) {
        for (int out_j = 0; out_j < *outW; out_j++) {
            int global_out_i = start_row + out_i;
            float sum = 0.0f;
            int center_i = global_out_i * sH;
            int center_j = out_j * sW;
            
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int in_i = center_i + ki - pad_h;
                    int in_j = center_j + kj - pad_w;
                    
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
                        sum += f[in_i * W + in_j] * g[ki * kW + kj];
                    }
                }
            }
            local_output[out_i * (*outW) + out_j] = sum;
        }
    }
    
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_proc + (r < extra ? r : extra);
            int r_rows = rows_per_proc + (r < extra ? 1 : 0);
            recvcounts[r] = r_rows * (*outW);
            displs[r] = r_start * (*outW);
        }
    }
    
    MPI_Gatherv(local_output, my_rows * (*outW), MPI_FLOAT,
                output, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    free(local_output);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Function to estimate and report memory usage
void report_memory_usage(int H, int W, int kH, int kW, int outH, int outW, int rank, int size, const char* mode) {
    if (rank == 0) {
        double input_mb = (H * W * sizeof(float)) / (1024.0 * 1024.0);
        double kernel_mb = (kH * kW * sizeof(float)) / (1024.0 * 1024.0);
        double output_mb = (outH * outW * sizeof(float)) / (1024.0 * 1024.0);
        double total_mb = input_mb + kernel_mb + output_mb;
        
        printf("Memory Usage (%s mode):\n", mode);
        printf("  Input matrix:  %.2f MB (%dx%d)\n", input_mb, H, W);
        printf("  Kernel matrix: %.2f MB (%dx%d)\n", kernel_mb, kH, kW);
        printf("  Output matrix: %.2f MB (%dx%d)\n", output_mb, outH, outW);
        printf("  Total per process: %.2f MB\n", total_mb);
        
        if (strcmp(mode, "mpi") == 0 || strcmp(mode, "hybrid") == 0) {
            double local_output_mb = (outH * outW * sizeof(float)) / (size * 1024.0 * 1024.0);
            printf("  Local output per MPI process: %.2f MB\n", local_output_mb);
            printf("  Total memory across %d processes: %.2f MB\n", size, total_mb * size);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 6) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <mode> <f_file|H> <g_file|kH> <sH> <sW> [o_file]\n", argv[0]);
            fprintf(stderr, "Modes: seq, omp, mpi, hybrid\n");
            fprintf(stderr, "For synthetic data: use numbers for H and kH (W=H, kW=kH)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    char *mode = argv[1];
    char *f_file = argv[2];
    char *g_file = argv[3];
    int sH = atoi(argv[4]);
    int sW = atoi(argv[5]);
    char *o_file = (argc > 6) ? argv[6] : NULL;
    
    float *f = NULL, *g = NULL;
    int H, W, kH, kW;
    
    // Check if using synthetic data (numeric arguments)
    int is_synthetic = (atoi(f_file) > 0 && atoi(g_file) > 0);
    
    if (is_synthetic) {
        // Generate synthetic matrices
        H = W = atoi(f_file);
        kH = kW = atoi(g_file);
        if (rank == 0) {
            printf("Generating synthetic %dx%d input and %dx%d kernel\n", H, W, kH, kW);
        }
        f = generate_random_matrix(H, W, 42 + rank);
        g = generate_random_matrix(kH, kW, 123 + rank);
    } else {
        // Read from files
        if (!read_matrix(f_file, &f, &H, &W)) {
            if (rank == 0) fprintf(stderr, "Error reading %s\n", f_file);
            MPI_Finalize();
            return 1;
        }
        
        if (!read_matrix(g_file, &g, &kH, &kW)) {
            if (rank == 0) fprintf(stderr, "Error reading %s\n", g_file);
            free(f);
            MPI_Finalize();
            return 1;
        }
    }
    
    int outH, outW;
    float *output = NULL;
    
    // Pre-calculate output dimensions for memory estimation
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;
    outH = (H + sH - 1) / sH;
    outW = (W + sW - 1) / sW;
    
    // Report memory usage
    report_memory_usage(H, W, kH, kW, outH, outW, rank, size, mode);
    
    // Optimized memory allocation based on mode
    if (rank == 0 || strcmp(mode, "mpi") == 0 || strcmp(mode, "hybrid") == 0) {
        if (strcmp(mode, "hybrid") == 0) {
            // Use NUMA-aware allocation for hybrid mode
            output = allocate_numa_aware_matrix(outH, outW);
        } else {
            // Standard allocation for other modes
            output = (float*)malloc(outH * outW * sizeof(float));
        }
        
        if (!output) {
            if (rank == 0) fprintf(stderr, "Error: Failed to allocate output matrix\n");
            free(f);
            free(g);
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    if (strcmp(mode, "seq") == 0) {
        if (rank == 0) {
            conv2d_sequential(f, H, W, g, kH, kW, sH, sW, output, &outH, &outW);
        }
    } else if (strcmp(mode, "omp") == 0) {
        if (rank == 0) {
            conv2d_openmp(f, H, W, g, kH, kW, sH, sW, output, &outH, &outW);
        }
    } else if (strcmp(mode, "mpi") == 0) {
        conv2d_mpi(f, H, W, g, kH, kW, sH, sW, output, &outH, &outW, rank, size);
    } else if (strcmp(mode, "hybrid") == 0) {
        conv2d_hybrid(f, H, W, g, kH, kW, sH, sW, output, &outH, &outW, rank, size);
    } else {
        if (rank == 0) {
            fprintf(stderr, "Unknown mode: %s\n", mode);
        }
        free(f);
        free(g);
        if (output) free(output);
        MPI_Finalize();
        return 1;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("Mode: %s, Time: %.6f seconds\n", mode, end - start);
        if (o_file && output) {
            write_matrix(o_file, output, outH, outW);
        }
    }
    
    free(f);
    free(g);
    if (output) free(output);
    
    MPI_Finalize();
    return 0;
}
