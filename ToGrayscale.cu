#include <iostream>

__global__ void toGrayscale(float* pictureIn, int height, int width, float* grayscaleOut) {
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) {
        return;
    }

    size_t offsetGrayscale = row * width + col;
    size_t offset = offsetGrayscale * 3; // 3 channels
    float r = pictureIn[offset];
    float g = pictureIn[offset + 1];
    float b = pictureIn[offset + 2];

    grayscaleOut[offsetGrayscale] = r * 0.299f + g * 0.587f + b * 0.114f;
}

int main() {
    constexpr int height = 72;
    constexpr int width = 128;
    constexpr int channels = 3; // R, G, B
    float picture [height][width][channels];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                picture[y][x][c] = rand() % 256;
            }
        }
    }
    float grayscale[height][width];

    float *picture_d, *grayscale_d;
    cudaMalloc(&picture_d, height * width * channels * sizeof(float));
    cudaMalloc(&grayscale_d, height * width * sizeof(float));

    for (int y = 0; y < height; ++y) {
        cudaMemcpy(picture_d + y * width * channels, picture[y], width * channels * sizeof(float), cudaMemcpyHostToDevice);
    }

    size_t batch = 16;
    toGrayscale<<<dim3((width + batch - 1) / batch, (height + batch - 1) / batch, 1), dim3(batch, batch, 1)>>>(picture_d, height, width, grayscale_d);

    for (int y = 0; y < height; ++y) {
        cudaError_t err = cudaMemcpy(grayscale[y], grayscale_d + y * width, width * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << err << "\n";
    }

    cudaFree(picture_d);
    cudaFree(grayscale_d);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << grayscale[y][x] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}