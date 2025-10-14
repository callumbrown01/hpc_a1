/*
 * Assignment 2: 2D Convolution Performance Analysis
 * Compares Sequential, OpenMP, MPI, and Hybrid (MPI+OpenMP) implementations
 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/*  OpenMP helpers  */
void setup_threading() {
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    omp_set_nested(1);

    printf("OpenMP Configuration:\n");
    printf("  Max threads available: %d\n", max_threads);
    printf("  Threads set to: %d\n", omp_get_max_threads());
    printf("  Number of processors: %d\n", omp_get_num_procs());

    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    printf("  Actual threads in parallel region: %d\n", actual_threads);

    char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        printf("  OMP_NUM_THREADS environment variable: %s\n\n", omp_threads);
    } else {
        printf("  OMP_NUM_THREADS not set - using all available cores\n\n");
    }
}

/* 2D array utils  */
float **allocate_2d_array(int rows, int cols) {
    float **array = (float **)malloc((size_t)rows * sizeof(float *));
    if (!array) return NULL;

    array[0] = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    if (!array[0]) { free(array); return NULL; }

    for (int i = 1; i < rows; i++) array[i] = array[0] + (size_t)i * (size_t)cols;
    return array;
}

void free_2d_array(float **array) {
    if (array) {
        free(array[0]);
        free(array);
    }
}

/* IO & generators */
float* generate_random_matrix(int rows, int cols, int seed) {
    srand(seed);
    float *data = (float*)malloc((size_t)rows * (size_t)cols * sizeof(float));
    if (!data) return NULL;
    for (size_t i = 0, n=(size_t)rows*(size_t)cols; i < n; i++) data[i] = (float)rand() / RAND_MAX;
    return data;
}

/*  contiguous reader */
int read_matrix_flat(const char *filename, float **data, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;

    if (fscanf(fp, "%d %d", rows, cols) != 2) { fclose(fp); return 0; }

    size_t n = (size_t)(*rows) * (size_t)(*cols);
    *data = (float*)malloc(n * sizeof(float));
    if (!*data) { fclose(fp); return 0; }

    for (size_t i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &(*data)[i]) != 1) {
            fclose(fp);
            free(*data); *data = NULL;
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

void write_matrix(const char *filename, float *data, int rows, int cols) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Failed to open %s for writing\n", filename); return; }
    fprintf(fp, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.3f", data[(size_t)i * (size_t)cols + (size_t)j]);
            if (j < cols - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/*  MPI-aware allocation  */
float* allocate_mpi_local_matrix(int total_rows, int total_cols, int rank, int size, int *local_rows) {
    int rows_per_proc = total_rows / size;
    int extra = total_rows % size;
    *local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    return (float*)malloc((size_t)(*local_rows) * (size_t)total_cols * sizeof(float));
}

/* NUMA-aware local allocation for hybrid */
float* allocate_numa_aware_matrix(int rows, int cols) {
    float *data = NULL;
    #pragma omp parallel
    {
        #pragma omp single
        data = (float*)malloc((size_t)rows * (size_t)cols * sizeof(float));
    }
    if (data) {
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) data[i] = 0.0f;
    }
    return data;
}

/*  Broadcast helpers  */
static void mpi_bcast_matrix(float **buf, int *H, int *W, int root) {
    MPI_Bcast(H, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(W, 1, MPI_INT, root, MPI_COMM_WORLD);
    int n = (*H) * (*W);
    if (*buf == NULL) *buf = (float*)malloc((size_t)n * sizeof(float));
    MPI_Bcast(*buf, n, MPI_FLOAT, root, MPI_COMM_WORLD);
}

/*  Convolution kernels  */
void conv2d_sequential(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW,
                       float *output, int *outH, int *outW) {
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
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                        sum += f[(size_t)in_i * (size_t)W + (size_t)in_j] * g[(size_t)ki * (size_t)kW + (size_t)kj];
                }
            }
            output[(size_t)out_i * (size_t)(*outW) + (size_t)out_j] = sum;
        }
    }
}

void conv2d_openmp(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW,
                   float *output, int *outH, int *outW) {
    int pad_h = (kH - 1) / 2;
    int pad_w = (kW - 1) / 2;

    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;

    #pragma omp parallel for schedule(guided) proc_bind(close)
    for (int out_i = 0; out_i < *outH; out_i++) {
        for (int out_j = 0; out_j < *outW; out_j++) {
            float sum = 0.0f;
            int center_i = out_i * sH;
            int center_j = out_j * sW;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int in_i = center_i + ki - pad_h;
                    int in_j = center_j + kj - pad_w;
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                        sum += f[(size_t)in_i * (size_t)W + (size_t)in_j] * g[(size_t)ki * (size_t)kW + (size_t)kj];
                }
            }
            output[(size_t)out_i * (size_t)(*outW) + (size_t)out_j] = sum;
        }
    }
}

/* MPI kernel using preallocated local_output and gathering into root */
void conv2d_mpi(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW,
                float *output_root, int *outH, int *outW, int rank, int size,
                float *local_output) {
    int pad_h = (kH - 1) / 2, pad_w = (kW - 1) / 2;
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;

    int rows_per_proc = (*outH) / size;
    int extra         = (*outH) % size;
    int start_row     = rank * rows_per_proc + (rank < extra ? rank : extra);
    int my_rows       = rows_per_proc + (rank < extra ? 1 : 0);

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
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                        sum += f[(size_t)in_i * (size_t)W + (size_t)in_j] * g[(size_t)ki * (size_t)kW + (size_t)kj];
                }
            }
            local_output[(size_t)out_i * (size_t)(*outW) + (size_t)out_j] = sum;
        }
    }

    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc((size_t)size * sizeof(int));
        displs     = (int*)malloc((size_t)size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_proc + (r < extra ? r : extra);
            int r_rows  = rows_per_proc + (r < extra ? 1 : 0);
            recvcounts[r] = r_rows * (*outW);
            displs[r]     = r_start * (*outW);
        }
    }

    MPI_Gatherv(local_output, my_rows * (*outW), MPI_FLOAT,
                output_root, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) { free(recvcounts); free(displs); }
}

/* HYBRID kernel using OpenMP within each rank, gather into root */
void conv2d_hybrid(float *f, int H, int W, float *g, int kH, int kW, int sH, int sW,
                   float *output_root, int *outH, int *outW, int rank, int size,
                   float *local_output) {
    int pad_h = (kH - 1) / 2, pad_w = (kW - 1) / 2;
    *outH = (H + sH - 1) / sH;
    *outW = (W + sW - 1) / sW;

    int rows_per_proc = (*outH) / size;
    int extra         = (*outH) % size;
    int start_row     = rank * rows_per_proc + (rank < extra ? rank : extra);
    int my_rows       = rows_per_proc + (rank < extra ? 1 : 0);

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
                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                        sum += f[(size_t)in_i * (size_t)W + (size_t)in_j] * g[(size_t)ki * (size_t)kW + (size_t)kj];
                }
            }
            local_output[(size_t)out_i * (size_t)(*outW) + (size_t)out_j] = sum;
        }
    }

    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc((size_t)size * sizeof(int));
        displs     = (int*)malloc((size_t)size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_proc + (r < extra ? r : extra);
            int r_rows  = rows_per_proc + (r < extra ? 1 : 0);
            recvcounts[r] = r_rows * (*outW);
            displs[r]     = r_start * (*outW);
        }
    }

    MPI_Gatherv(local_output, my_rows * (*outW), MPI_FLOAT,
                output_root, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) { free(recvcounts); free(displs); }
}

/*  Memory reporting  */
void report_memory_usage(int H, int W, int kH, int kW, int outH, int outW, int rank, int size, const char* mode) {
    if (rank == 0) {
        double input_mb  = ((double)H * (double)W * sizeof(float)) / (1024.0 * 1024.0);
        double kernel_mb = ((double)kH * (double)kW * sizeof(float)) / (1024.0 * 1024.0);
        double output_mb = ((double)outH * (double)outW * sizeof(float)) / (1024.0 * 1024.0);
        double total_mb  = input_mb + kernel_mb + output_mb;

        printf("Memory Usage (%s mode):\n", mode);
        printf("  Input matrix:  %.2f MB (%dx%d)\n", input_mb, H, W);
        printf("  Kernel matrix: %.2f MB (%dx%d)\n", kernel_mb, kH, kW);
        printf("  Output matrix: %.2f MB (%dx%d)\n", output_mb, outH, outW);
        printf("  Total per process: %.2f MB\n", total_mb);

        if (strcmp(mode, "mpi") == 0 || strcmp(mode, "hybrid") == 0) {
            double local_output_mb = output_mb / (double)size;
            printf("  Local output per MPI process: %.2f MB\n", local_output_mb);
            printf("  Total memory across %d processes: %.2f MB\n", size, total_mb * size);
        }
        printf("\n");
    }
}

/*  Main. */
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

    char *mode  = argv[1];
    char *f_arg = argv[2];
    char *g_arg = argv[3];
    int sH = atoi(argv[4]);
    int sW = atoi(argv[5]);
    char *o_file = (argc > 6) ? argv[6] : NULL;

    float *f = NULL, *g = NULL;
    int H=0, W=0, kH=0, kW=0;

    int is_synthetic = (atoi(f_arg) > 0 && atoi(g_arg) > 0);

    /* Generate or Read on root, then broadcast to all ranks */
    if (is_synthetic) {
        if (rank == 0) {
            H = W = atoi(f_arg);
            kH = kW = atoi(g_arg);
            printf("Generating synthetic %dx%d input and %dx%d kernel\n", H, W, kH, kW);
            f = generate_random_matrix(H, W, 42);   /* fixed seed on root */
            g = generate_random_matrix(kH, kW, 123);
        }
        mpi_bcast_matrix(&f, &H, &W, 0);
        mpi_bcast_matrix(&g, &kH, &kW, 0);
    } else {
        if (rank == 0) {
            if (!read_matrix_flat(f_arg, &f, &H, &W)) {
                fprintf(stderr, "Error reading %s\n", f_arg);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (!read_matrix_flat(g_arg, &g, &kH, &kW)) {
                fprintf(stderr, "Error reading %s\n", g_arg);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        mpi_bcast_matrix(&f, &H, &W, 0);
        mpi_bcast_matrix(&g, &kH, &kW, 0);
    }

    int outH = (H + sH - 1) / sH;
    int outW = (W + sW - 1) / sW;

    if ((strcmp(mode,"omp")==0 || strcmp(mode,"hybrid")==0) && rank==0) {
        setup_threading();
    }

    report_memory_usage(H, W, kH, kW, outH, outW, rank, size, mode);

    float *output_root = NULL;   /* root buffer for mpi/hybrid (gathered result) */
    float **output2d   = NULL;   /* for omp (2D view + contiguous) */
    float *output_seq  = NULL;   /* for seq */
    float *local_output = NULL;  /* per-rank local slice for mpi/hybrid */

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (strcmp(mode, "seq") == 0) {
        if (rank == 0) {
            output_seq = (float*)malloc((size_t)outH * (size_t)outW * sizeof(float));
            if (!output_seq) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            conv2d_sequential(f, H, W, g, kH, kW, sH, sW, output_seq, &outH, &outW);
        }
    } else if (strcmp(mode, "omp") == 0) {
        if (rank == 0) {
            output2d = allocate_2d_array(outH, outW);
            if (!output2d) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            conv2d_openmp(f, H, W, g, kH, kW, sH, sW, output2d[0], &outH, &outW);
        }
    } else if (strcmp(mode, "mpi") == 0) {
        int local_rows = 0;
        local_output = allocate_mpi_local_matrix(outH, outW, rank, size, &local_rows);
        if (!local_output) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (rank == 0) {
            output_root = (float*)malloc((size_t)outH * (size_t)outW * sizeof(float));
            if (!output_root) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        conv2d_mpi(f, H, W, g, kH, kW, sH, sW,
                   output_root, &outH, &outW, rank, size, local_output);
    } else if (strcmp(mode, "hybrid") == 0) {
        int rows_per_proc = outH / size, extra = outH % size;
        int my_rows = rows_per_proc + (rank < extra ? 1 : 0);
        local_output = allocate_numa_aware_matrix(my_rows, outW);
        if (!local_output) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (rank == 0) {
            output_root = (float*)malloc((size_t)outH * (size_t)outW * sizeof(float));
            if (!output_root) { fprintf(stderr, "Alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        conv2d_hybrid(f, H, W, g, kH, kW, sH, sW,
                      output_root, &outH, &outW, rank, size, local_output);
    } else {
        if (rank == 0) fprintf(stderr, "Unknown mode: %s\n", mode);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Mode: %s, Time: %.6f seconds\n", mode, end - start);
        if (o_file) {
            if (strcmp(mode,"seq")==0 && output_seq)                      write_matrix(o_file, output_seq, outH, outW);
            else if (strcmp(mode,"omp")==0 && output2d)                  write_matrix(o_file, output2d[0], outH, outW);
            else if ((strcmp(mode,"mpi")==0 || strcmp(mode,"hybrid")==0) && output_root)
                                                                          write_matrix(o_file, output_root, outH, outW);
        }
    }

    free(f); free(g);
    if (output_seq) free(output_seq);
    if (output2d)   free_2d_array(output2d);
    if (local_output) free(local_output);
    if (output_root) free(output_root);

    MPI_Finalize();
    return 0;
}
