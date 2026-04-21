#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "../src/preprocess/preprocess_cuda.cuh"

int main() {
  using namespace neugn::preprocess;

  std::vector<int64_t> h_src{0, 1, 2, 3, 1};
  std::vector<int64_t> h_dst{1, 2, 3, 4, 3};
  int64_t N = 5;
  int64_t E = static_cast<int64_t>(h_src.size());

  int64_t *d_src = nullptr, *d_dst = nullptr;
  cudaMalloc(&d_src, sizeof(int64_t) * E);
  cudaMalloc(&d_dst, sizeof(int64_t) * E);
  cudaMemcpy(d_src, h_src.data(), sizeof(int64_t) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst, h_dst.data(), sizeof(int64_t) * E, cudaMemcpyHostToDevice);

  float* d_rwse = nullptr;
  cudaMalloc(&d_rwse, sizeof(float) * N * 4);
  ComputeRWSECUDA(d_src, d_dst, E, N, 4, d_rwse);

  int64_t* d_spd = nullptr;
  cudaMalloc(&d_spd, sizeof(int64_t) * N * N);
  ComputeSubgraphSPDCUDA(d_src, d_dst, E, N, 3, d_spd);

  std::vector<int64_t> h_nodes{1, 3, 4};
  int64_t *d_nodes = nullptr, *d_sub_src = nullptr, *d_sub_dst = nullptr, *d_sub_ec = nullptr;
  cudaMalloc(&d_nodes, sizeof(int64_t) * h_nodes.size());
  cudaMemcpy(d_nodes, h_nodes.data(), sizeof(int64_t) * h_nodes.size(), cudaMemcpyHostToDevice);
  cudaMalloc(&d_sub_src, sizeof(int64_t) * E);
  cudaMalloc(&d_sub_dst, sizeof(int64_t) * E);
  cudaMalloc(&d_sub_ec, sizeof(int64_t));
  SimpleNodeSubgraphCUDA(d_src, d_dst, E, N, d_nodes, static_cast<int64_t>(h_nodes.size()),
                         nullptr, nullptr, d_sub_src, d_sub_dst, d_sub_ec, nullptr, nullptr);

  int64_t *d_path_src = nullptr, *d_path_dst = nullptr, *d_path_len = nullptr;
  cudaMalloc(&d_path_src, sizeof(int64_t) * (E + N));
  cudaMalloc(&d_path_dst, sizeof(int64_t) * (E + N));
  cudaMalloc(&d_path_len, sizeof(int64_t));
  Graph2PathV2PureDeterministicCUDA(d_src, d_dst, E, N, d_path_src, d_path_dst, d_path_len);

  cudaDeviceSynchronize();

  std::vector<float> h_rwse(N * 4);
  std::vector<int64_t> h_spd(N * N), h_path_len(1), h_sub_ec(1);
  cudaMemcpy(h_rwse.data(), d_rwse, sizeof(float) * N * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_spd.data(), d_spd, sizeof(int64_t) * N * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_path_len.data(), d_path_len, sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sub_ec.data(), d_sub_ec, sizeof(int64_t), cudaMemcpyDeviceToHost);

  std::cout << "rwse[0,*]=";
  for (int i = 0; i < 4; ++i) std::cout << " " << h_rwse[i];
  std::cout << "\nspd[0,*]=";
  for (int64_t j = 0; j < N; ++j) std::cout << " " << h_spd[j];
  std::cout << "\nsubgraph edges=" << h_sub_ec[0];
  std::cout << "\npath_len=" << h_path_len[0] << "\n";

  cudaFree(d_src); cudaFree(d_dst); cudaFree(d_rwse); cudaFree(d_spd);
  cudaFree(d_nodes); cudaFree(d_sub_src); cudaFree(d_sub_dst); cudaFree(d_sub_ec);
  cudaFree(d_path_src); cudaFree(d_path_dst); cudaFree(d_path_len);
  return 0;
}
