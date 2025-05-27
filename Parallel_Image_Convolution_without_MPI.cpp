#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector> 
#include <numeric> 
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

// Modified apply_convolution to work on a local chunk
void apply_convolution_local(
    const unsigned char* local_image_chunk_with_halo, // Input data for this rank (includes halo rows)
    unsigned char* local_output_chunk,           // Output buffer for this rank's rows
    int full_image_width,
    int local_chunk_height_with_halo,          // Height of local_image_chunk_with_halo
    int channels,
    const float* kernel,
    int kernel_size,
    int output_start_y_in_local_chunk,         // The 'y' index in local_image_chunk_with_halo that corresponds to the first row of output
    int num_rows_to_output_for_this_rank)      // Number of rows this rank should write to local_output_chunk
{
    int k_half = kernel_size / 2;

    
    for (int y_out_idx = 0; y_out_idx < num_rows_to_output_for_this_rank; ++y_out_idx) {
        for (int x = 0; x < full_image_width; ++x) {
            // y_conv_center is the y-coordinate in the local_image_chunk_with_halo
            // that is the center of the kernel for the current output pixel.
            int y_conv_center = output_start_y_in_local_chunk + y_out_idx;

            for (int c = 0; c < channels; ++c) {
                float sum_val = 0.0f;
                for (int ky = -k_half; ky <= k_half; ++ky) {
                    for (int kx = -k_half; kx <= k_half; ++kx) {
                        // ix and iy are coordinates within local_image_chunk_with_halo
                        int ix = std::min(std::max(x + kx, 0), full_image_width - 1); // x is relative to full width
                        int iy = std::min(std::max(y_conv_center + ky, 0), local_chunk_height_with_halo - 1);

                        sum_val += local_image_chunk_with_halo[(iy * full_image_width + ix) * channels + c] *
                                   kernel[(ky + k_half) * kernel_size + (kx + k_half)];
                    }
                }
                local_output_chunk[(y_out_idx * full_image_width + x) * channels + c] =
                    static_cast<unsigned char>(std::clamp(sum_val, 0.0f, 255.0f));
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
    int kernel_size_arg = stoi(argv[3]);
    float* kernel_data = new float[kernel_size_arg * kernel_size_arg];
    int k_half = kernel_size_arg / 2;

    if (rank == 0) { 
        float sigma = kernel_size_arg / 4.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        float sum_kernel = 0.0f;
        for (int y = -k_half; y <= k_half; ++y) {
            for (int x = -k_half; x <= k_half; ++x) {
                float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel_data[(y + k_half) * kernel_size_arg + (x + k_half)] = value;
                sum_kernel += value;
            }
        }
        for (int i = 0; i < kernel_size_arg * kernel_size_arg; ++i) {
            kernel_data[i] /= sum_kernel;
        }
    }
    // Broadcast kernel data to all processes
    MPI_Bcast(kernel_data, kernel_size_arg * kernel_size_arg, MPI_FLOAT, 0, MPI_COMM_WORLD);


    int img_width = 0, img_height = 0, img_channels = 0;
    unsigned char* full_image_data = nullptr;
    unsigned char* full_output_data = nullptr;

    if (rank == 0) {
        full_image_data = load_image(input_filename, &img_width, &img_height, &img_channels);
        if (full_image_data == nullptr) {
            // Signal error to other processes
            int error_code = 1;
            for(int i=1; i<size; ++i) MPI_Send(&error_code, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Finalize();
            return 1;
        }
        // Signal success
        int error_code = 0;
        for(int i=1; i<size; ++i) MPI_Send(&error_code, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        full_output_data = new unsigned char[img_width * img_height * img_channels];
    } else {
        int error_code;
        MPI_Recv(&error_code, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (error_code == 1) {
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast image dimensions
    MPI_Bcast(&img_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate row distribution for each process
    std::vector<int> rows_per_rank(size);
    std::vector<int> start_row_for_rank_output(size + 1, 0); // Global start row for output
    int base_rows = img_height / size;
    int remainder_rows = img_height % size;
    for (int i = 0; i < size; ++i) {
        rows_per_rank[i] = base_rows + (i < remainder_rows ? 1 : 0);
        start_row_for_rank_output[i+1] = start_row_for_rank_output[i] + rows_per_rank[i];
    }

    int my_num_output_rows = rows_per_rank[rank];
    int my_global_start_row_for_output = start_row_for_rank_output[rank];

    // Determine the slice of the input image needed (with halo)
    int input_slice_start_row = std::max(0, my_global_start_row_for_output - k_half);
    int input_slice_end_row = std::min(img_height - 1, my_global_start_row_for_output + my_num_output_rows - 1 + k_half);
    int local_input_chunk_height = input_slice_end_row - input_slice_start_row + 1;
    
    unsigned char* local_input_chunk = nullptr;
    if (local_input_chunk_height > 0 && my_num_output_rows > 0) { // Process only if there are rows to process
        local_input_chunk = new unsigned char[local_input_chunk_height * img_width * img_channels];
    }
    unsigned char* local_output_chunk = nullptr;
    if (my_num_output_rows > 0) {
         local_output_chunk = new unsigned char[my_num_output_rows * img_width * img_channels];
    }
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            if (rows_per_rank[r] == 0) continue; // Skip ranks with no rows

            int r_input_start_row = std::max(0, start_row_for_rank_output[r] - k_half);
            int r_input_end_row = std::min(img_height - 1, start_row_for_rank_output[r] + rows_per_rank[r] - 1 + k_half);
            int r_chunk_height = r_input_end_row - r_input_start_row + 1;
            
            if (r_chunk_height > 0) {
                long long offset = static_cast<long long>(r_input_start_row) * img_width * img_channels;
                long long count = static_cast<long long>(r_chunk_height) * img_width * img_channels;
                if (r == 0) {
                    if (local_input_chunk && count > 0) { // Check if local_input_chunk is allocated
                         memcpy(local_input_chunk, full_image_data + offset, count);
                    }
                } else {
                    MPI_Send(full_image_data + offset, count, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        if (local_input_chunk_height > 0 && my_num_output_rows > 0) { // Check if expecting data
            long long count_recv = static_cast<long long>(local_input_chunk_height) * img_width * img_channels;
            MPI_Recv(local_input_chunk, count_recv, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    double start_time = 0.0, end_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    if (my_num_output_rows > 0 && local_input_chunk != nullptr && local_output_chunk != nullptr) { // Ensure buffers are valid
        int output_start_y_in_this_chunk = my_global_start_row_for_output - input_slice_start_row;
        apply_convolution_local(local_input_chunk, local_output_chunk, img_width, local_input_chunk_height, img_channels,
                                kernel_data, kernel_size_arg, output_start_y_in_this_chunk, my_num_output_rows);
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // Sync before gather
    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Convolution applied in " << (end_time - start_time) << " seconds." << endl;
    }

    // Gather results using MPI_Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = rows_per_rank[i] * img_width * img_channels;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }

    long long my_output_bytes = static_cast<long long>(my_num_output_rows) * img_width * img_channels;

    MPI_Gatherv(my_num_output_rows > 0 ? local_output_chunk : MPI_IN_PLACE, // Send buffer (or MPI_IN_PLACE if rank 0 and its chunk is already in full_output_data)
                my_output_bytes, MPI_UNSIGNED_CHAR,           // Send count and type
                full_output_data, recvcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, // Recv buffer, counts, displs, type (only used by root)
                0, MPI_COMM_WORLD);                           // Root and communicator

    if (rank == 0) {
        save_image(output_filename, full_output_data, img_width, img_height, img_channels);
        delete[] full_image_data;
        delete[] full_output_data;
    }

    delete[] kernel_data;
    if (local_input_chunk) delete[] local_input_chunk;
    if (local_output_chunk) delete[] local_output_chunk;

    MPI_Finalize();
    return 0;
}