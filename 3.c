#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

// Function to set maximum number of threads
void setup_threading() {
    // Check OMP_NUM_THREADS environment variable first
    char* omp_threads_env = getenv("OMP_NUM_THREADS");
    int requested_threads = 0;
    
    if (omp_threads_env) {
        requested_threads = atoi(omp_threads_env);
        printf("OpenMP Configuration:\n");
        printf("  OMP_NUM_THREADS environment variable: %s\n", omp_threads_env);
        printf("  Requested threads: %d\n", requested_threads);
        
        // Use the environment variable setting
        omp_set_num_threads(requested_threads);
    } else {
        // Set to maximum available threads only if no environment variable
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        printf("OpenMP Configuration:\n");
        printf("  No OMP_NUM_THREADS set - using max available\n");
        printf("  Max threads available: %d\n", max_threads);
    }
    
    printf("  Max threads available: %d\n", omp_get_max_threads());
    printf("  Number of processors: %d\n", omp_get_num_procs());
    
    // Test that threads are actually working
    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    printf("  Actual threads in parallel region: %d\n", actual_threads);
    
    // Check environment variables - but don't print if already handled
    if (!omp_threads_env) {
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

// Function to compare two arrays for correctness verification
int compare_arrays(float **array1, float **array2, int rows, int cols) {
    const float tolerance = 1e-6f; // Small tolerance for floating point comparison
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = array1[i][j] - array2[i][j];
            if (diff < 0) diff = -diff; // absolute value
            if (diff > tolerance) {
                printf("Difference found at [%d][%d]: %.6f vs %.6f (diff: %.6f)\n", 
                       i, j, array1[i][j], array2[i][j], diff);
                return 0; // Arrays are different
            }
        }
    }
    return 1; // Arrays are identical (within tolerance)
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
void conv2d_parallel_collapse(
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
    // Handle -t flag for thread testing
    if (argc == 2 && strcmp(argv[1], "-t") == 0) {
        setup_threading();
        printf("Thread test completed successfully.\n");
        return 0;
    }
    
    // Setup threading first
    setup_threading();
    
    int H = 0, W = 0, kH = 0, kW = 0;
    char *f_file = NULL, *g_file = NULL, *o_file = NULL;
    int skip_sequential = 0;  // Flag to skip sequential computation
    int opt;

    // Parse command line arguments - handle kH and kW separately first
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-kH") == 0 && i + 1 < argc) {
            kH = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-kW") == 0 && i + 1 < argc) {
            kW = atoi(argv[i + 1]);
        }
    }

    // Parse remaining command line arguments
    while ((opt = getopt(argc, argv, "H:W:f:g:o:k:s")) != -1) {
        switch (opt) {
            case 'H': H = atoi(optarg); break;
            case 'W': W = atoi(optarg); break;
            case 'f': f_file = optarg; break;
            case 'g': g_file = optarg; break;
            case 'o': o_file = optarg; break;
            case 's': skip_sequential = 1; break;  // Skip sequential computation
            case 'k': 
                // Only set if kH and kW weren't already set separately
                if (kH == 0 && kW == 0) {
                    kH = kW = atoi(optarg); 
                }
                break;
            default:
                fprintf(stderr, "Usage: %s -H height -W width [-f input_file] [-g kernel_file] [-o output_file] [-k kernel_size | -kH height -kW width] [-s]\n", argv[0]);
                fprintf(stderr, "Examples:\n");
                fprintf(stderr, "  %s -H 1000 -W 1000 -kH 3 -kW 3\n", argv[0]);
                fprintf(stderr, "  %s -H 1000 -W 1000 -kH 3 -kW 3 -s  (skip sequential)\n", argv[0]);
                fprintf(stderr, "  %s -H 1000 -W 1000 -kH 3 -kW 3 -f f.txt -g g.txt -o o.txt\n", argv[0]);
                exit(1);
        }
    }

    float **f = NULL, **g = NULL, **output = NULL;
    srand((unsigned int)time(NULL));

    // Handle input matrix
    if (H > 0 && W > 0) {
        // Generate matrix first
        f = allocate_2d_array(H, W);
        generate_random_array(f, H, W);
        printf("Generated random input matrix: %dx%d\n", H, W);
        
        // Save generated matrix if filename provided
        if (f_file) {
            write_array_to_file(f_file, f, H, W);
            printf("Saved input matrix to %s\n", f_file);
        }
    } else if (f_file) {
        // Try to read from file if no dimensions provided
        if (!read_array_from_file(f_file, &f, &H, &W)) {
            fprintf(stderr, "Error reading input matrix\n");
            exit(1);
        }
        printf("Read input matrix from %s: %dx%d\n", f_file, H, W);
    } else {
        fprintf(stderr, "Error: Must specify either input file or dimensions\n");
        exit(1);
    }

    // Handle kernel matrix
    if (kH > 0 && kW > 0) {
        // Generate kernel first
        g = allocate_2d_array(kH, kW);
        generate_random_array(g, kH, kW);
        printf("Generated random kernel matrix: %dx%d\n", kH, kW);
        
        // Save generated matrix if filename provided
        if (g_file) {
            write_array_to_file(g_file, g, kH, kW);
            printf("Saved kernel matrix to %s\n", g_file);
        }
    } else if (g_file) {
        // Try to read from file if no kernel size provided
        if (!read_array_from_file(g_file, &g, &kH, &kW)) {
            fprintf(stderr, "Error reading kernel matrix\n");
            exit(1);
        }
        printf("Read kernel matrix from %s: %dx%d\n", g_file, kH, kW);
    } else {
        fprintf(stderr, "Error: Must specify either kernel file or kernel size\n");
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

    double seq_time = 0.0;  // Default value when skipped
    
    if (!skip_sequential) {
        // Sequential convolution (baseline)
        double start_seq = omp_get_wtime();
        conv2d_sequential(f, H, W, g, kH, kW, output_seq);
        double end_seq = omp_get_wtime();
        seq_time = end_seq - start_seq;
        printf("Sequential convolution time: %.9f seconds\n", seq_time);
    } else {
        printf("Sequential convolution: SKIPPED\n");
    }

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
    conv2d_parallel_collapse(f, H, W, g, kH, kW, output_par5);
    double end_par5 = omp_get_wtime();
    double par5_time = end_par5 - start_par5;
    printf("Parallel collapse approach time: %.9f seconds (%.2fx speedup)\n", 
           par5_time, seq_time / par5_time);

    // Find the fastest implementation
    double fastest_time = par1_time;
    int fastest_idx = 1;
    const char* fastest_name = "Basic static";
    
    if (par2_time < fastest_time) { fastest_time = par2_time; fastest_idx = 2; fastest_name = "Dynamic scheduling"; }
    if (par3_time < fastest_time) { fastest_time = par3_time; fastest_idx = 3; fastest_name = "Guided scheduling"; }
    if (par5_time < fastest_time) { fastest_time = par5_time; fastest_idx = 5; fastest_name = "Collapse approach"; }

    printf("\nFastest parallel implementation: %s (%.9f seconds, %.2fx speedup)\n", 
           fastest_name, fastest_time, seq_time / fastest_time);

    // Verify algorithm correctness by comparing sequential vs parallel results
    printf("\n=== ALGORITHM CORRECTNESS VERIFICATION ===\n");
    printf("Comparing sequential output with parallel implementations...\n");
    
    int all_correct = 1;
    
    printf("Sequential vs Parallel Basic Static:    ");
    if (compare_arrays(output_seq, output_par1, H, W)) {
        printf("✓ IDENTICAL\n");
    } else {
        printf("✗ DIFFERENT\n");
        all_correct = 0;
    }
    
    printf("Sequential vs Parallel Dynamic:         ");
    if (compare_arrays(output_seq, output_par2, H, W)) {
        printf("✓ IDENTICAL\n");
    } else {
        printf("✗ DIFFERENT\n");
        all_correct = 0;
    }
    
    printf("Sequential vs Parallel Guided:          ");
    if (compare_arrays(output_seq, output_par3, H, W)) {
        printf("✓ IDENTICAL\n");
    } else {
        printf("✗ DIFFERENT\n");
        all_correct = 0;
    }
    
    printf("Sequential vs Parallel Collapse:        ");
    if (compare_arrays(output_seq, output_par5, H, W)) {
        printf("✓ IDENTICAL\n");
    } else {
        printf("✗ DIFFERENT\n");
        all_correct = 0;
    }
    
    if (all_correct) {
        printf("\n🎉 ALL TESTS PASSED: All parallel implementations produce identical results!\n");
        printf("   Algorithm correctness verified.\n");
    } else {
        printf("\n❌ ALGORITHM ERROR: Some parallel implementations produce different results!\n");
        printf("   Please check the implementation for bugs.\n");
    }

    // Write output if requested (uses fastest parallel implementation)
    if (o_file) {
        // Write the fastest parallel implementation result
        switch (fastest_idx) {
            case 1: write_array_to_file(o_file, output_par1, H, W); break;
            case 2: write_array_to_file(o_file, output_par2, H, W); break;
            case 3: write_array_to_file(o_file, output_par3, H, W); break;
            case 4: write_array_to_file(o_file, output_par4, H, W); break;
            case 5: write_array_to_file(o_file, output_par5, H, W); break;
        }
        printf("Saved output matrix to %s\n", o_file);
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