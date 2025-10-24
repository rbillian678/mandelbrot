#include <cstdint>
#include <complex>
#include <vector>
#include <chrono>
#include <arm_neon.h>
#include <iostream>
#include <fstream>

std::vector<std::vector<uint32_t>> mandelbrot_neon(float xmin, float xmax, float ymin, float ymax, uint32_t w, uint32_t h, uint32_t max_iter)
{
    // auto t0 = std::chrono::steady_clock::now();
    float deltaX = (xmax - xmin) / w;
    float deltaY = (ymax - ymin) / h;
    std::vector<uint32_t> Z(h * w, 0); // maybe uint32_t
    for (uint32_t i = 0; i < h; i++)
    {
        for (uint32_t j = 0; j < w; j += 4)
        {
            float32x4_t real = {xmin + i * deltaX, xmin + i * deltaX, xmin + i * deltaX, xmin + i * deltaX};
            float32x4_t imag = {ymin + j * deltaY, ymin + (j + 1) * deltaY, ymin + (j + 2) * deltaY, ymin + (j + 3) * deltaY};
            float32x4_t zReal = vdupq_n_f32(0);
            float32x4_t zImag = vdupq_n_f32(0);
            uint32x4_t it = vdupq_n_u32(max_iter);
            for (uint16_t k = 0; k < max_iter; k++)
            {
                // z = z * z + c;
                // (real+imag*i)^2 = real^2 - imag^2 + 2*real*imag*i
                float32x4_t sqReal = vmulq_f32(zReal, zReal);
                float32x4_t sqImag = vmulq_f32(zImag, zImag);
                float32x4_t aSqPlusBsq = vsubq_f32(sqReal, sqImag);
                float32x4_t imaginary = vmulq_f32(zReal, zImag);
                imaginary = vmulq_n_f32(imaginary, 2.0f);
                aSqPlusBsq = vaddq_f32(aSqPlusBsq, real);
                imaginary = vaddq_f32(imaginary, imag);
                zReal = aSqPlusBsq;
                zImag = imaginary;

                float32x4_t c = vmulq_f32(zReal, zReal);
                float32x4_t d = vmulq_f32(zImag, zImag);
                c = vaddq_f32(c, d);

                float32x4_t b = vdupq_n_f32(4.0f);
                uint32x4_t mask = vcgtq_f32(c, b); // if abs(z)^2 > 4 ? 1:0
                uint32x4_t temp = vbslq_u32(mask, vdupq_n_u32(k), vdupq_n_u32(max_iter));
                it = vminq_u32(it, temp);
                // // bellow step
                // if (std::abs(z) > 2)
                // {
                //     return i;
                // }
                // Combine all lanes with AND — true only if all lanes are 0xFFFFFFFF
                uint64x2_t pair = vreinterpretq_u64_u32(it);
                uint64_t combined = vgetq_lane_u64(pair, 0) & vgetq_lane_u64(pair, 1);

                bool all_equal = (combined == 0xFFFFFFFFFFFFFFFFull);
                if (all_equal)
                {
                    break;
                }
            }
            vst1q_u32(&Z[j * w + i], it);
            // Z[j][i] = maxIterj
        }
    }
    std::vector<std::vector<uint32_t>> ans(h, std::vector<uint32_t>(w, 0));
    int idx{0};
    for (uint32_t i = 0; i < h; i++)
    {
        for (uint32_t j = 0; j < w; j++)
        {
            ans[i][j] = Z[idx++];
        }
    }
    // auto t1 = std::chrono::steady_clock::now();
    // std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
    return ans;
}

int main()
{
    float xmin = -2.0f;
    float xmax = 1.0f;
    float ymin = -1.0f;
    float ymax = 1.0f;
    uint32_t w = 10;
    uint32_t h = 10;
    uint32_t max_iter = 100;
    std::vector<std::vector<uint32_t>> Z = mandelbrot_neon(xmin, xmax, ymin, ymax, w, h, max_iter);

    // write z to file.txt
    std::ofstream file("file.txt");
    for (uint32_t i = 0; i < h; i++)
    {
        for (uint32_t j = 0; j < w; j++)
        {
            file << Z[i][j] << " ";
        }
        file << std::endl
             << std::endl;
    }
    file.close();

    return 0;
}