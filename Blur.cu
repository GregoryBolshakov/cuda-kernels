#include <iostream>

__global__ void blurPicture(float* pictureIn, unsigned height, unsigned width, unsigned blurSize, float* pictureOut) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {return;}

    float areaSum = 0.f;
    unsigned count = 0;
    for (int yOffset = -(int)blurSize; yOffset <= (int)blurSize; ++yOffset) {
        for (int xOffset = -(int)blurSize; xOffset <= (int)blurSize; ++xOffset) {
            int curRow = row + yOffset;
            int curCol = col + xOffset;
            if (curRow < 0 || (unsigned)curRow >= height || curCol < 0 || (unsigned)curCol >= width) {
                continue;
            }

            areaSum += pictureIn[curRow * width + curCol];
            count++;
        }
    }

    pictureOut[row * width + col] = areaSum / count;
}

int main() {
    constexpr unsigned height = 18;
    constexpr unsigned width = 24;
    float picture [height][width]; // Assume only one channel
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            picture[y][x] = rand() % 256;
        }
    }
    float blurred[height][width];

    float *picture_d, *blurred_d;
    cudaMalloc(&picture_d, height * width * sizeof(float));
    cudaMalloc(&blurred_d, height * width * sizeof(float));

    for (unsigned y = 0; y < height; ++y) {
        cudaMemcpy(picture_d, &picture[0][0], height * width * sizeof(float), cudaMemcpyHostToDevice);
    }

    size_t batch = 16;
    blurPicture<<<dim3((width + batch - 1) / batch, (height + batch - 1) / batch, 1), dim3(batch, batch, 1)>>>(picture_d, height, width, 1, blurred_d);

    cudaMemcpy(&blurred[0][0], blurred_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(blurred_d);
    cudaFree(picture_d);

    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            std::cout << blurred[y][x] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}