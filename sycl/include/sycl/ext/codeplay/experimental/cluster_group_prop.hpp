//==--- cluster_group_prop.hpp --- SYCL extension for cuda cluster group ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental {

namespace cuda {
template <int Dim>
struct cluster_size
    : ::sycl::ext::oneapi::experimental::detail::run_time_property_key<
          ::sycl::ext::oneapi::experimental::detail::ClusterLaunch> {
  cluster_size(const range<Dim> &size) : size(size) {}
  sycl::range<Dim> get_cluster_size() { return size; }

private:
  range<Dim> size;
};
template <int Dim> using cluster_size_key = cluster_size<Dim>;
} // namespace cuda

namespace detail {
template <typename PropertiesT> constexpr std::size_t getClusterDim() {
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::codeplay::experimental::cuda::cluster_size_key<
                        1>>()) {
    return 1;
  }
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::codeplay::experimental::cuda::cluster_size_key<
                        2>>()) {
    return 2;
  }
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::codeplay::experimental::cuda::cluster_size_key<
                        3>>()) {
    return 3;
  }
  return 0;
}
} // namespace detail
} // namespace ext::codeplay::experimental

} // namespace _V1
} // namespace sycl

namespace sycl_cp = sycl::ext::codeplay::experimental;
namespace sycl_ex = sycl::ext::oneapi::experimental;

template <>
struct sycl_ex::is_property_key<sycl_cp::cuda::cluster_size_key<1>>
    : std::true_type {};
template <>
struct sycl_ex::is_property_key<sycl_cp::cuda::cluster_size_key<2>>
    : std::true_type {};
template <>
struct sycl_ex::is_property_key<sycl_cp::cuda::cluster_size_key<3>>
    : std::true_type {};

template <typename T>
struct sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<1>, T>
    : std::true_type {};

template <typename T>
struct sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<2>, T>
    : std::true_type {};

template <typename T>
struct sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<3>, T>
    : std::true_type {};

template <>
struct sycl_ex::is_property_value<sycl_cp::cuda::cluster_size_key<1>>
    : is_property_key<sycl_cp::cuda::cluster_size_key<1>> {};
template <>
struct sycl_ex::is_property_value<sycl_cp::cuda::cluster_size_key<2>>
    : is_property_key<sycl_cp::cuda::cluster_size_key<2>> {};
template <>
struct sycl_ex::is_property_value<sycl_cp::cuda::cluster_size_key<3>>
    : is_property_key<sycl_cp::cuda::cluster_size_key<3>> {};

template <typename O>
struct sycl_ex::is_property_value_of<sycl_cp::cuda::cluster_size_key<1>, O>
    : sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<1>, O> {};

template <typename O>
struct sycl_ex::is_property_value_of<sycl_cp::cuda::cluster_size_key<2>, O>
    : sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<2>, O> {};

template <typename O>
struct sycl_ex::is_property_value_of<sycl_cp::cuda::cluster_size_key<3>, O>
    : sycl_ex::is_property_key_of<sycl_cp::cuda::cluster_size_key<3>, O> {};
