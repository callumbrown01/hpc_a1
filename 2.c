#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

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

// Parallel implementation of 2D convolution using OpenMP
void conv2d_parallel(
    float **f, int H, int W,
    float **g, int kH, int kW,
    float **output
) {
    int pad_h = kH / 2;
    int pad_w = kW / 2;
 
    #pragma omp parallel for collapse(2)
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
    int H = 0, W = 0, kH = 0, kW = 0;
    char *f_file = NULL, *g_file = NULL, *o_file = NULL;
    int opt;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "H:W:f:g:o:k:")) != -1) {
        switch (opt) {
            case 'H': H = atoi(optarg); break;
            case 'W': W = atoi(optarg); break;
            case 'f': f_file = optarg; break;
            case 'g': g_file = optarg; break;
            case 'o': o_file = optarg; break;
            case 'k': kH = kW = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage %s [-H height] [-W width] [-f input_file] [-g kernel_file] [-o output_file] [-k kernel_size]\n", argv[0]);
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

    // Allocate separate output matrices
    float **output_seq = allocate_2d_array(H, W);
    float **output_par = allocate_2d_array(H, W);

    // Set num threads
    int max_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", max_threads);
    printf("Problem size: %dx%d matrix with %dx%d kernel\n", H, W, kH, kW);
    printf("Total operations: %d (matrix elements) x %d (kernel elements) = %d\n", 
           H*W, kH*kW, H*W*kH*kW);
    omp_set_num_threads(max_threads);

    // Time the sequential performance
    double start_seq = omp_get_wtime();
    conv2d_sequential(f, H, W, g, kH, kW, output_seq);
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;
    
    // Time the parallel performance
    double start_par = omp_get_wtime();
    conv2d_parallel(f, H, W, g, kH, kW, output_par);
    double end_par = omp_get_wtime();
    double par_time = end_par - start_par;
    
    // Calculate speedup
    double speedup = 0.0;
    if (par_time > 0 && seq_time > 0) {
        speedup = seq_time / par_time;
    }
    
    // Output timing data in consistent format for parsing
    printf("Sequential convolution time: %.6f seconds\n", seq_time);
    printf("Parallel convolution time: %.6f seconds\n", par_time);
    printf("Speedup: %.2fx\n", speedup);
    
    // Check if problem is too small for parallelization
    int total_work = H * W * kH * kW;
    if (total_work < 100000) {
        printf("WARNING: Problem size may be too small for effective parallelization\n");
        printf("Consider using matrices larger than 100x100 for better parallel speedup\n");
    }

    // Write output if requested (use parallel result)
    if (o_file) {
        write_array_to_file(o_file, output_par, H, W);
    }

    // Clean up
    free_2d_array(f);
    free_2d_array(g);
    free_2d_array(output_seq);
    free_2d_array(output_par);

    return 0;
}