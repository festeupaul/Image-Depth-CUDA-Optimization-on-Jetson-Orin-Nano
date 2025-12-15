/*
 * Scale Recovery Geometric pe Jetson Orin Nano - BENCHMARK VERSION
 * Masoara timpul de executie pe GPU si timpul total al algoritmului.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono> // Pentru masurarea timpului pe CPU/Total

// --- CONFIGURARE PARAMETRI CAMERĂ ---
#define H_REF 375.0f
#define FX_REF 718.856f
#define CY_REF 185.21f
#define CAMERA_HEIGHT 1.65f 

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "\n[CRITIC] Eroare CUDA la linia " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (Cod: " << err << ")\n"; \
            exit(1); \
        } \
    } while (0)

__global__ void scale_kernel(float* depth_map, int total_pixels, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < total_pixels; i += blockDim.x * gridDim.x) {
        depth_map[i] *= scale_factor;
    }
}

float compute_median(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    float median = v[n];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n - 1, v.end());
        median = (v[n - 1] + median) / 2.0f;
    }
    return median;
}

int main(int argc, char** argv) {
    std::cout << "--- Jetson Orin Nano Scale Recovery (Benchmark) ---\n";

    if (argc < 4) {
        std::cerr << "Utilizare: ./scale_app <input_file.txt> <width> <height>\n";
        return 1;
    }

    std::string input_file = argv[1];
    int W = std::atoi(argv[2]);
    int H = std::atoi(argv[3]);
    int num_pixels = W * H;

    std::cout << "Imagine: " << W << " x " << H << " (" << num_pixels << " pixeli)\n";

    // Alocare Memorie Unificată
    float* d_depth_map;
    CUDA_CHECK(cudaMallocManaged(&d_depth_map, num_pixels * sizeof(float)));

    // Citire Date (Nu includem asta in benchmark-ul algoritmului)
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "[EROARE] Nu pot deschide fișierul: " << input_file << "\n";
        cudaFree(d_depth_map);
        return 1;
    }
    
    for (int i = 0; i < num_pixels; ++i) {
        infile >> d_depth_map[i];
    }
    infile.close();
    std::cout << "[CPU] Date incarcate. Incepem Benchmark-ul...\n";

    // --- START CRONOMETRU TOTAL (Algoritm) ---
    auto start_algo = std::chrono::high_resolution_clock::now();

    // 1. Calcul Median (CPU Logic)
    float fy = FX_REF * ((float)H / H_REF);
    float cy = CY_REF * ((float)H / H_REF);

    int row_start = (int)(H * 0.80);
    int row_end = (int)(H * 0.95);
    int col_start = (int)(W * 0.40);
    int col_end = (int)(W * 0.60);

    std::vector<float> geo_depths;
    geo_depths.reserve(row_end - row_start);
    for (int r = row_start; r < row_end; ++r) {
        float pixel_offset = (float)r - cy;
        if (pixel_offset < 1.0f) pixel_offset = 1.0f;
        geo_depths.push_back((fy * CAMERA_HEIGHT) / pixel_offset);
    }
    float median_geo = compute_median(geo_depths);

    std::vector<float> pred_patch;
    pred_patch.reserve((row_end - row_start) * (col_end - col_start));
    for (int r = row_start; r < row_end; ++r) {
        for (int c = col_start; c < col_end; ++c) {
            pred_patch.push_back(d_depth_map[r * W + c]);
        }
    }
    float median_pred = compute_median(pred_patch);

    float scale_factor = 1.0f;
    if (median_pred > 0.001f) {
        scale_factor = median_geo / median_pred;
    }

    // --- EXECUȚIE GPU (Benchmark Kernel) ---
    
    // Configurare Events pentru masurare precisa GPU
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Masuram doar executia kernelului
    CUDA_CHECK(cudaEventRecord(start_gpu));
    scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_depth_map, num_pixels, scale_factor);
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    
    // Asteptam sa termine GPU
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));
    
    // Calculam timpul GPU
    float milliseconds_gpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));

    // --- STOP CRONOMETRU TOTAL ---
    auto end_algo = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_total = end_algo - start_algo;

    std::cout << "-> Scale Factor: " << scale_factor << "\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "[BENCHMARK] Timp Kernel GPU (Strict Scalare): " << milliseconds_gpu << " ms\n";
    std::cout << "[BENCHMARK] Timp Total Algoritm (Calcul Factor + GPU): " << duration_total.count() << " ms\n";
    std::cout << "--------------------------------------------------\n";

    // Curatare Events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Salvare Rezultat
    std::ofstream outfile("metric_depth_cuda.txt");
    outfile.precision(4);
    outfile << std::fixed;

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            outfile << d_depth_map[r * W + c];
            if (c < W - 1) outfile << " ";
        }
        outfile << "\n";
    }
    outfile.close();

    std::cout << "[SUCCES] Fisier salvat: metric_depth_cuda.txt\n";

    cudaFree(d_depth_map);
    return 0;
}