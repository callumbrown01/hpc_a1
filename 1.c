#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

// Function to set maximum number of threads
void setup_threading() {
    // Set to maximum available threads
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    
    // Set thread affinity for better performance
    omp_set_nested(1);  // Enable nested parallelism if needed
    
    printf("OpenMP Configuration:\n");
    printf("  Max threads available: %d\n", max_threads);
    printf("  Threads set to: %d\n", omp_get_max_threads());
    printf("  Number of processors: %d\n", omp_get_num_procs());
    
    // Test that threads are actually working
    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    printf("  Actual threads in parallel region: %d\n", actual_threads);
    
    // Check environment variables
    char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        printf("  OMP_NUM_THREADS environment variable: %s\n", omp_threads);
    } else {
        printf("  OMP_NUM_THREADS not set - using all available cores\n");
    }
    printf("\n");
}

// Function to allocate 2D array
float **allocate_2d_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float *));
    array[0] = (float *)malloc(rows * cols * sizeof(float));
    for (int i = 1; i < rows; i++) {
        array[i] = array[0] + i * cols;
    }
    return array;
}

// Function to free 2D array
void free_2d_array(float **array) {
    free(array[0]);
    free(array);
}

// Function to generate random array
void generate_random_array(float **array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

// Function to read array from file
int read_array_from_file(const char *filename, float ***array, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;

    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return 0;
    }

    *array = allocate_2d_array(*rows, *cols);

    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(fp, "%f", &((*array)[i][j])) != 1) {
                fclose(fp);
                return 0;
            }
        }
    }
    fclose(fp);
    return 1;
}

// Function to write array to file
void write_array_to_file(const char *filename, float **array, int rows, int cols) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.3f ", array[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Sequential implementation of 2D convolution
void conv2d_sequential(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki <= pad_h; ki++) {
                for (int kj = -pad_w; kj <= pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Parallel implementation 1: Basic parallel for with static
void conv2d_parallel_basic(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    // Verify thread usage
    static int first_call = 1;
    if (first_call) {
        int threads_used = 0;
        #pragma omp parallel
        {
            #pragma omp single
            threads_used = omp_get_num_threads();
        }
        printf("Basic parallel implementation using %d threads\n", threads_used);
        first_call = 0;
    }

    #pragma omp parallel for  schedule(static) proc_bind(close)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki <= pad_h; ki++) {
                for (int kj = -pad_w; kj <= pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Parallel implementation 2: Dynamic scheduling
void conv2d_parallel_dynamic(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    #pragma omp parallel for schedule(dynamic) proc_bind(close)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki <= pad_h; ki++) {
                for (int kj = -pad_w; kj <= pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Parallel implementation 3: Guided scheduling
void conv2d_parallel_guided(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    #pragma omp parallel for schedule(guided) proc_bind(close)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki <= pad_h; ki++) {
                for (int kj = -pad_w; kj <= pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}



// Parallel implementation 4: collapse approach
void conv2d_parallel_tiled(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;
    const int tile_size = 16;

    #pragma omp parallel for collapse(2) proc_bind(close)
    for (int ti = 0; ti < H; ti += tile_size) {
        for (int tj = 0; tj < W; tj += tile_size) {
            int i_end = (ti + tile_size < H) ? ti + tile_size : H;
            int j_end = (tj + tile_size < W) ? tj + tile_size : W;
            
            for (int i = ti; i < i_end; i++) {
                for (int j = tj; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int ki = -pad_h; ki <= pad_h; ki++) {
                        for (int kj = -pad_w; kj <= pad_w; kj++) {
                            int ii = i + ki;
                            int jj = j + kj;
                            if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                                sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                            }
                        }
                    }
                    output[i][j] = sum;
                }
            }
        }
    }
}

// Wrapper for backward compatibility
void conv2d_parallel(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    conv2d_parallel_basic(f, H, W, g, kH, kW, output);
}

// Wrapper function that calls either sequential or parallel implementation
void conv2d(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    #ifdef USE_PARALLEL
        conv2d_parallel(f, H, W, g, kH, kW, output);
    #else
        conv2d_sequential(f, H, W, g, kH, kW, output);
    #endif
}

int main(int argc, char *argv[]) {
    // Setup threading first
    setup_threading();
    
    int H = 0, W = 0, kH = 0, kW = 0;
    char *f_file = NULL, *g_file = NULL, *o_file_seq = NULL, *o_file_par = NULL;
    int opt;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "H:W:f:g:s:p:k:")) != -1) {
        switch (opt) {
            case 'H': H = atoi(optarg); break;
            case 'W': W = atoi(optarg); break;
            case 'f': f_file = optarg; break;
            case 'g': g_file = optarg; break;
            case 's': o_file_seq = optarg; break; // sequential output file
            case 'p': o_file_par = optarg; break; // parallel output file
            case 'k': kH = kW = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-H height] [-W width] [-f input_file] [-g kernel_file] [-s seq_output_file] [-p par_output_file] [-k kernel_size]\n", argv[0]);
                exit(1);
        }
    }

    float **f = NULL, **g = NULL, **output = NULL;
    srand((unsigned int)time(NULL));

    // Handle input matrix
    if (f_file) {
        if (!read_array_from_file(f_file, &f, &H, &W)) {
            fprintf(stderr, "Error reading input matrix\n");
            exit(1);
        }
    } else if (H > 0 && W > 0) {
        f = allocate_2d_array(H, W);
        generate_random_array(f, H, W);
    } else {
        fprintf(stderr, "Error Must specify either input file or dimensions\n");
        exit(1);
    }

    // Handle kernel matrix
    if (g_file) {
        if (!read_array_from_file(g_file, &g, &kH, &kW)) {
            fprintf(stderr, "Error reading kernel matrix\n");
            exit(1);
        }
    } else if (kH > 0 && kW > 0) {
        g = allocate_2d_array(kH, kW);
        generate_random_array(g, kH, kW);
    } else {
        fprintf(stderr, "Error Must specify either kernel file or kernel size\n");
        exit(1);
    }

    // Allocate output matrices for all implementations
    float **output_seq = allocate_2d_array(H, W);
    float **output_par1 = allocate_2d_array(H, W);
    float **output_par2 = allocate_2d_array(H, W);
    float **output_par3 = allocate_2d_array(H, W);
    float **output_par4 = allocate_2d_array(H, W);
    float **output_par5 = allocate_2d_array(H, W);

    printf("Running convolution performance tests on %dx%d matrix with %dx%d kernel...\n", H, W, kH, kW);
    printf("Number of OpenMP threads: %d\n\n", omp_get_max_threads());

    // Sequential convolution (baseline)
    double start_seq = omp_get_wtime();
    conv2d_sequential(f, H, W, g, kH, kW, output_seq);
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;
    printf("Sequential convolution time: %.9f seconds\n", seq_time);

    // Parallel implementation 1: Basic static
    double start_par1 = omp_get_wtime();
    conv2d_parallel_basic(f, H, W, g, kH, kW, output_par1);
    double end_par1 = omp_get_wtime();
    double par1_time = end_par1 - start_par1;
    printf("Parallel basic static time: %.9f seconds (%.2fx speedup)\n", 
           par1_time, seq_time / par1_time);

    // Parallel implementation 2: Dynamic scheduling
    double start_par2 = omp_get_wtime();
    conv2d_parallel_dynamic(f, H, W, g, kH, kW, output_par2);
    double end_par2 = omp_get_wtime();
    double par2_time = end_par2 - start_par2;
    printf("Parallel dynamic scheduling time: %.9f seconds (%.2fx speedup)\n", 
           par2_time, seq_time / par2_time);

    // Parallel implementation 3: Guided scheduling
    double start_par3 = omp_get_wtime();
    conv2d_parallel_guided(f, H, W, g, kH, kW, output_par3);
    double end_par3 = omp_get_wtime();
    double par3_time = end_par3 - start_par3;
    printf("Parallel guided scheduling time: %.9f seconds (%.2fx speedup)\n", 
           par3_time, seq_time / par3_time);


    // Parallel implementation 4: collapse approach
    double start_par5 = omp_get_wtime();
    conv2d_parallel_tiled(f, H, W, g, kH, kW, output_par5);
    double end_par5 = omp_get_wtime();
    double par5_time = end_par5 - start_par5;
    printf("Parallel tiled approach time: %.9f seconds (%.2fx speedup)\n", 
           par5_time, seq_time / par5_time);

    // Find the fastest implementation
    double fastest_time = par1_time;
    int fastest_idx = 1;
    const char* fastest_name = "Basic static";
    
    if (par2_time < fastest_time) { fastest_time = par2_time; fastest_idx = 2; fastest_name = "Dynamic scheduling"; }
    if (par3_time < fastest_time) { fastest_time = par3_time; fastest_idx = 3; fastest_name = "Guided scheduling"; }
    if (par5_time < fastest_time) { fastest_time = par5_time; fastest_idx = 5; fastest_name = "Tiled approach"; }

    printf("\nFastest parallel implementation: %s (%.9f seconds, %.2fx speedup)\n", 
           fastest_name, fastest_time, seq_time / fastest_time);

    // Write outputs if requested (uses sequential and fastest parallel)
    if (o_file_seq) {
        write_array_to_file(o_file_seq, output_seq, H, W);
    }
    if (o_file_par) {
        // Write the fastest parallel implementation result
        switch (fastest_idx) {
            case 1: write_array_to_file(o_file_par, output_par1, H, W); break;
            case 2: write_array_to_file(o_file_par, output_par2, H, W); break;
            case 3: write_array_to_file(o_file_par, output_par3, H, W); break;
            case 4: write_array_to_file(o_file_par, output_par4, H, W); break;
            case 5: write_array_to_file(o_file_par, output_par5, H, W); break;
        }
    }

    // Clean up
    free_2d_array(f);
    free_2d_array(g);
    free_2d_array(output_seq);
    free_2d_array(output_par1);
    free_2d_array(output_par2);
    free_2d_array(output_par3);
    free_2d_array(output_par4);
    free_2d_array(output_par5);

    return 0;
}