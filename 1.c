#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

// Function to allocate 2D array
float allocate_2d_array(int rows, int cols) {
    float array = (float)malloc(rows  sizeof(float));
    array[0] = (float)malloc(rows  cols  sizeof(float));
    for (int i = 1; i  rows; i++) {
        array[i] = array[0] + i  cols;
    }
    return array;
}

// Function to free 2D array
void free_2d_array(float array) {
    free(array[0]);
    free(array);
}

// Function to generate random array
void generate_random_array(float array, int rows, int cols) {
    for (int i = 0; i  rows; i++) {
        for (int j = 0; j  cols; j++) {
            array[i][j] = (float)rand()  RAND_MAX;
        }
    }
}

// Function to read array from file
int read_array_from_file(const char filename, float array, int rows, int cols) {
    FILE fp = fopen(filename, r);
    if (!fp) return 0;
    
    fscanf(fp, %d %d, rows, cols);
    array = allocate_2d_array(rows, cols);
    
    for (int i = 0; i  rows; i++) {
        for (int j = 0; j  cols; j++) {
            fscanf(fp, %f, &(array)[i][j]);
        }
    }
    fclose(fp);
    return 1;
}

// Function to write array to file
void write_array_to_file(const char filename, float array, int rows, int cols) {
    FILE fp = fopen(filename, w);
    fprintf(fp, %d %dn, rows, cols);
    for (int i = 0; i  rows; i++) {
        for (int j = 0; j  cols; j++) {
            fprintf(fp, %.3f , array[i][j]);
        }
        fprintf(fp, n);
    }
    fclose(fp);
}
// Sequential implementation of 2D convolution
void conv2d_sequential(
    float f, int H, int W,
    float g, int kH, int kW,
    float output
) {
    int pad_h = kH  2;
    int pad_w = kW  2;

    for (int i = 0; i  H; i++) {
        for (int j = 0; j  W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki = pad_h; ki++) {
                for (int kj = -pad_w; kj = pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii = 0 && ii  H && jj = 0 && jj  W) {
                        sum += f[ii][jj]  g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Parallel implementation of 2D convolution using OpenMP
void conv2d_parallel(
    float f, int H, int W,
    float g, int kH, int kW,
    float output
) {
    int pad_h = kH  2;
    int pad_w = kW  2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i  H; i++) {
        for (int j = 0; j  W; j++) {
            float sum = 0.0f;
            for (int ki = -pad_h; ki = pad_h; ki++) {
                for (int kj = -pad_w; kj = pad_w; kj++) {
                    int ii = i + ki;
                    int jj = j + kj;
                    if (ii = 0 && ii  H && jj = 0 && jj  W) {
                        sum += f[ii][jj]  g[ki + pad_h][kj + pad_w];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// Wrapper function that calls either sequential or parallel implementation
void conv2d(
    float f, int H, int W,
    float g, int kH, int kW,
    float output
) {
    #ifdef USE_PARALLEL
        conv2d_parallel(f, H, W, g, kH, kW, output);
    #else
        conv2d_sequential(f, H, W, g, kH, kW, output);
    #endif
}

int main(int argc, char argv) {
    int H = 0, W = 0, kH = 0, kW = 0;
    char f_file = NULL, g_file = NULL, o_file = NULL;
    int opt;
    
    //Parse command line arguments
    while ((opt = getopt(argc, argv, HWfgok)) != -1) {
        switch (opt) {
            case 'H' H = atoi(optarg); break;
            case 'W' W = atoi(optarg); break;
            case 'f' f_file = optarg; break;
            case 'g' g_file = optarg; break;
            case 'o' o_file = optarg; break;
            case 'k' kH = kW = atoi(optarg); break;
            default
                fprintf(stderr, Usage %s [-H height] [-W width] [-f input_file] [-g kernel_file] [-o output_file] [-k kernel_size]n, argv[0]);
                exit(1);
        }
    }

    float f = NULL, g = NULL, output = NULL;
    srand(time(NULL));

     Handle input matrix
    if (f_file) {
        
        read_array_from_file(f_file, &f, &H, &W);
    } else if (H  0 && W  0) {
        f = allocate_2d_array(H, W);
        generate_random_array(f, H, W);
    } else {
        fprintf(stderr, Error Must specify either input file or dimensionsn);
        exit(1);
    }

     Handle kernel matrix
    if (g_file) {
        read_array_from_file(g_file, &g, &kH, &kW);
    } else if (kH  0 && kW  0) {
        g = allocate_2d_array(kH, kW);
        generate_random_array(g, kH, kW);
    } else {
        fprintf(stderr, Error Must specify either kernel file or kernel sizen);
        exit(1);
    }

    // Allocate output matrix
    output = allocate_2d_array(H, W);

     Perform convolution and time it
    double start_time = omp_get_wtime();
    conv2d(f, H, W, g, kH, kW, output);
    double end_time = omp_get_wtime();
    printf(Convolution time %.9f secondsn, end_time - start_time);

    // Write output if requested
    if (o_file) {
        write_array_to_file(o_file, output, H, W);
    }

     Clean up
    free_2d_array(f);
    free_2d_array(g);
    free_2d_array(output);

    return 0;
}