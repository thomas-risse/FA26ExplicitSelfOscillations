#ifndef EIGEN_UTILITY_H
#define EIGEN_UTILITY_H

#include <Eigen/Dense>

template <typename Derived, typename ftype>
auto ClipEigen(Eigen::Ref<const Eigen::ArrayX<Derived>> array, const ftype& min,
               const ftype& max) {
  return array.cwiseMin(max).cwiseMax(min);
}

template <typename Derived, typename ftype>
auto ClipEigen(const Eigen::ArrayBase<Derived>& array, const ftype& min,
               const ftype& max) {
  return array.cwiseMin(max).cwiseMax(min);
}

template <typename Derived1, typename Derived2>
auto SafeSetEigen(Eigen::ArrayBase<Derived1>& array,
                  Eigen::Ref<const Eigen::ArrayX<Derived2>> array2) {
  int minDim = std::min(array.size(), array2.size());
  array.head(minDim) = array2.head(minDim);
}

template <typename Derived1, typename Derived2>
auto SafeSetEigen(Eigen::ArrayBase<Derived1>& array,
                  const Eigen::ArrayBase<Derived2>& array2) {
  int minDim = std::min(array.size(), array2.size());
  array.head(minDim) = array2.head(minDim);
}

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

#endif  // EIGEN_UTILITY_H