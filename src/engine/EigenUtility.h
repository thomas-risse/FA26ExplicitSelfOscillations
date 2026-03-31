#ifndef EIGEN_UTILITY_H
#define EIGEN_UTILITY_H

#include <Eigen/Dense>

template <typename derived, typename ftype>
auto smoothRampMatrix(const Eigen::MatrixBase<derived>& x, ftype epsilon = 1) {
  const auto t = ((x.array()) / (epsilon)).cwiseMax(0).cwiseMin(1);
  return (x.array() > epsilon)
      .select(x.array(), (-t * t * t + 2 * t * t) * epsilon)
      .matrix();
}

template <typename ftype>
ftype softplus(const ftype& x, ftype epsilon = 1, ftype threshold = 40) {
  if (x / epsilon > threshold) {
    return x;
  } else {
    return epsilon * log1p(exp(x / epsilon));
  }
}

template <typename ftype>
ftype softplusDerivative(const ftype& x, ftype epsilon = 1,
                         ftype threshold = 40) {
  if (x / epsilon > threshold) {
    return 1;
  } else {
    return exp(x / epsilon) / (1 + exp(x / epsilon));
  }
}

template <typename derived, typename ftype>
auto softplusMatrix(const Eigen::MatrixBase<derived>& x, ftype epsilon = 1,
                    ftype threshold = 40) {
  return ((x / epsilon).array() > threshold)
      .select(x, (x / epsilon).array().exp().log1p().matrix() * epsilon);
}

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