#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
//load image
unsigned char* load_image(const char* filename, int* width, int* height, int* channels) {
    unsigned char* image = stbi_load(filename, width, height, channels, 0);
    if (image == nullptr) {
        cerr << "Error loading image: " << stbi_failure_reason() << endl;
        return nullptr;
    }
    return image;
}

//apply convolution
void apply_convolution(unsigned char* image, unsigned char* output, int width, int height, int channels, const float* kernel, int kernel_size) {
    int k_half = kernel_size / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -k_half; ky <= k_half; ++ky) {
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        int ix = min(max(x + kx, 0), width - 1);
                        int iy = min(max(y + ky, 0), height - 1);
                        sum += image[(iy * width + ix) * channels + c] * kernel[(ky + k_half) * kernel_size + (kx + k_half)];
                    }
                }
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(clamp(sum, 0.0f, 255.0f));
            }
        }
    }
}

//save image
void save_image(const char* filename, unsigned char* image, int width, int height, int channels) {
    if (!stbi_write_png(filename, width, height, channels, image, width * channels)) {
        cerr << "Error saving image: " << stbi_failure_reason() << endl;
    }
}

//main function
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_image> <output_image> <kernel_size>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int kernel_size = stoi(argv[3]);
    float* kernel = new float[kernel_size * kernel_size];

    // Initialize kernel (example: Gaussian blur)
    float sigma = kernel_size / 4.0f; // Adjust this factor as needed, e.g., kernel_size / 6.0f
    if (sigma < 0.5f) sigma = 0.5f; // Ensure sigma is not too small
    float sum = 0.0f;
    int k_half = kernel_size / 2;
    for (int y = -k_half; y <= k_half; ++y) {
        for (int x = -k_half; x <= k_half; ++x) {
            float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + k_half) * kernel_size + (x + k_half)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernel[i] /= sum;
    }
    int width, height, channels;
    unsigned char* image = load_image(input_filename, &width, &height, &channels);
    if (image == nullptr) {
        MPI_Finalize();
        return 1;
    }

    unsigned char* output = new unsigned char[width * height * channels];

    double start_time = 0.0, end_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    apply_convolution(image, output, width, height, channels, kernel, kernel_size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Convolution applied in " << (end_time - start_time) << " seconds by rank 0." << endl;
    }

    if (rank == 0) {
        save_image(argv[2], output, width, height, channels);
    }
    delete[] image;
    delete[] output;
    delete[] kernel;

    MPI_Finalize();
    return 0;
}
