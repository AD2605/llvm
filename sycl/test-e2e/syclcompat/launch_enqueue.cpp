// Launches a kernel with a Cluster Group Property
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_90 -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/sycl.hpp>

#include <syclcompat.hpp>

void checkClusterDims(int *correct_result_flag, char *a) {
  uint32_t cluster_dim_x, cluster_dim_y, cluster_dim_z;
// Temporary solution till cluster group class is implemented
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__) &&            \
    (__SYCL_CUDA_ARCH__ >= 900)
  asm volatile("\n\t"
               "mov.u32 %0, %%cluster_nctaid.x; \n\t"
               "mov.u32 %1, %%cluster_nctaid.y; \n\t"
               "mov.u32 %2, %%cluster_nctaid.z; \n\t"
               : "=r"(cluster_dim_z), "=r"(cluster_dim_y), "=r"(cluster_dim_x));
#endif
  auto ptr_a = reinterpret_cast<int *>(a);
  ptr_a[syclcompat::local_id::x()] = syclcompat::local_id::x();
  if (cluster_dim_z != 1 || cluster_dim_y != 2 || cluster_dim_x != 2) {
    *correct_result_flag = 0;
  }
}

int main() {
  using namespace sycl::ext::oneapi::experimental;

  sycl::queue queue;

  cluster_size cluster_dims(sycl::range<3>(2, 2, 1));
  properties cluster_launch_property{cluster_dims};

  int *correct_result_flag = sycl::malloc_device<int>(1, queue);
  queue.memset(correct_result_flag, 1, sizeof(int)).wait();
  syclcompat::dim3 grid_dim(1, 2, 2);
  syclcompat::dim3 block_dim(1, 32, 32);

  syclcompat::experimental::launch<checkClusterDims>(
      grid_dim, block_dim, 8192, cluster_launch_property, correct_result_flag);
  syclcompat::wait_and_throw();

  int correct_result_flag_host = 1;
  syclcompat::memcpy(&correct_result_flag_host, correct_result_flag,
                     sizeof(int));

  if (!correct_result_flag_host) {
    std::cerr << "Cluster Dimensions did not match " << std::endl;
  }

  return !correct_result_flag_host;
}
