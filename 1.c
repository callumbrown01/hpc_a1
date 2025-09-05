#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

/*
  Parallelisation approach:
  - We parallelise ONLY the outer image row loop (i) with `#pragma omp parallel for schedule(static)`.
  - Each thread owns a disjoint set of full rows of the output. This avoids multiple threads
    writing to adjacent elements in the same cache line (false sharing) and minimises cache-coherency traffic.
  - We deliberately do NOT use collapse(2) on (i,j) because flattening can interleave adjacent
    output elements across threads and slow things down due to false sharing and extra scheduling overhead.

  Memory representation:
  - Each 2D array (f, g, output) is allocated as a single contiguous block for all elements,
    with a row-pointer table indexing into that block: array[0] points to the base, and array[r]
    points to array[0] + r * cols. This keeps rows contiguous in memory and improves spatial locality.

  Cache considerations:
  - Writing entire rows per thread gives contiguous stores, improving write-combine and cache line utilisation.
  - The convolution kernel g is typically small and benefits from cache reuse as we sweep across a row.
  - Keeping the inner loops (ki, kj) tight and the j loop innermost maintains sequential access over row-major data.
*/

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

    // Parallelise across rows only to keep each thread's writes contiguous (avoid false sharing)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki <= pad_h; ki++) {
                int ii = i + ki;
                if (ii < 0 || ii >= H) continue;  // quick bounds check on ii once per ki
                for (int kj = -pad_w; kj <= pad_w; kj++) {
                    int jj = j + kj;
                    if (jj >= 0 && jj < W) {
                        sum += f[ii][jj] * g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
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

    float **f = NULL, **g = NULL, **output_seq = NULL, **output_par = NULL;
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

    // Allocate output matrices
    output_seq = allocate_2d_array(H, W);
    output_par = allocate_2d_array(H, W);

    // Run sequential
    double start_time = omp_get_wtime();
    conv2d_sequential(f, H, W, g, kH, kW, output_seq);
    double end_time = omp_get_wtime();
    printf("Sequential Convolution time %.9f seconds\n", end_time - start_time);

    // Run parallel
    start_time = omp_get_wtime();
    conv2d_parallel(f, H, W, g, kH, kW, output_par);
    end_time = omp_get_wtime();
    printf("Parallel Convolution time %.9f seconds\n", end_time - start_time);

    // Write output if requested (write parallel version by default)
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