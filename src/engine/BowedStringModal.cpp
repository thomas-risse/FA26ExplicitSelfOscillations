#include "BowedStringModal.h"

#include <iostream>

template <typename ftype>
BowedStringModal<ftype>::BowedStringModal(float sampleRate,
                                          std::shared_ptr<Bow<ftype>> bow,
                                          int Nmodes, int dissipationMode) {
  this->bow = bow;
  this->Nmodes = Nmodes;
  ReinitDsp(sampleRate);
  this->dissipationMode = dissipationMode;
};

template <typename ftype>
void BowedStringModal<ftype>::ReinitDsp(float sampleRate) {
  sr = sampleRate;
  dt = 1 / (float(sr));

  // Reinit state
  qnow = Eigen::Array<ftype, -1, 1>::Zero(Nmodes);
  qnext = qnow;
  qlast = qnow;

  RHS = LHS = qnow;

  // Reinit system matrices
  M = Eigen::Array<ftype, -1, 1>::Ones(Nmodes);
  K = M;
  R0 = M;

  Amps = M;
  Omega = 2 * M_PI * 100 * Amps;
  Decays = M;

  // Compute excitation vector
  exProj = Eigen::Array<ftype, -1, 1>::Ones(Nmodes);
  computeInputVector();

  // Pre-compute fixed intermediate vectors
  updateIntermediateQuantities();
};

template <typename ftype>
void BowedStringModal<ftype>::updateIntermediateQuantities() {
  A0_inv = dt * dt / (M + dt * R0 / 2);
  A0_inv_phi = A0_inv * exProj;
  phi_A0_inv_phi = (A0_inv_phi * exProj).sum();

  M_qnow = -K + 2 * M / (dt * dt);
  M_qlast = -(M / (dt * dt) - R0 / (2 * dt));
}

template <typename ftype>
void BowedStringModal<ftype>::computeInputVector() {
  for (int i = 0; i < Nmodes; i++) {
    exProj(i) = sin(exPos * (i + 1) * M_PI);
  }
}

template <typename ftype>
void BowedStringModal<ftype>::setExPos(ftype pos) {
  this->exPos = std::clamp(pos, ftype(0), ftype(1));
  computeInputVector();
  updateIntermediateQuantities();
}

template <typename ftype>
void BowedStringModal<ftype>::setPhysicalParameters(VecRef M, VecRef K,
                                                    VecRef R) {
  this->K = K;
  this->M = M;
  this->R0 = R;

  updateIntermediateQuantities();
};

template <typename ftype>
void BowedStringModal<ftype>::computePhysicalParameters() {
  R0 = Omega.cwiseInverse();
  M = 1 / (2 * 6.9) * Decays.cwiseProduct(R0);
  K = M.cwiseProduct(Omega).cwiseProduct(Omega);

  updateIntermediateQuantities();
};

template <typename ftype>
void BowedStringModal<ftype>::setLinearParameters(VecRef Amps, VecRef Freqs,
                                                  VecRef Decays) {
  SafeSetEigen(this->Amps, Amps);
  // Omega is clamped to ensure stability of the scheme
  SafeSetEigen(this->Omega,
               ClipEigen(2 * M_PI * Freqs, ftype(0), ftype(2.0 * sr)).eval());
  SafeSetEigen(this->Decays, Decays);
  computePhysicalParameters();
};

template <typename ftype>
void BowedStringModal<ftype>::setAmps(VecRef Amps) {
  SafeSetEigen(this->Amps, Amps);
  computePhysicalParameters();
};

template <typename ftype>
void BowedStringModal<ftype>::setFreqs(VecRef Freqs) {
  SafeSetEigen(this->Omega,
               ClipEigen(2 * M_PI * Freqs, ftype(0), ftype(2.0 * sr)).eval());
  computePhysicalParameters();
};

template <typename ftype>
void BowedStringModal<ftype>::setDecays(VecRef Decays) {
  SafeSetEigen(this->Decays, Decays);
  computePhysicalParameters();
};

template <typename ftype>
std::tuple<Eigen::ArrayX<ftype>, Eigen::ArrayX<ftype>>
BowedStringModal<ftype>::process(ftype vBow, ftype FBow) {
  // Evaluate dissipation matrix
  // Trick for initialization
  if (std::isnan(vBowLast)) {
    vBowLast = vBow;
  }
  if (std::isnan(FBowLast)) {
    FBowLast = FBow;
  }

  // if (dissipationMode == 0) {
  vrel = (exProj * (qnow - qlast)).sum() / dt - (vBow + vBowLast) / 2;
  phival = bow->phi(vrel);
  Fb_v = phival * (FBow + FBowLast) / 2 / (vrel + std::copysign(NUM_EPS, vrel));

  if (dissipationMode == 1) {
    // Right hand side
    RHS = M_qnow * qnow + M_qlast * qlast
          + exProj * Fb_v / (2 * dt) * (exProj * qlast).sum()
          + exProj * Fb_v * (vBow + vBowLast) / 2;

    // Solve using Shermann-Morrisson
    qnext = A0_inv * (RHS)-A0_inv_phi * Fb_v * (A0_inv_phi * RHS).sum()
            / (2 * dt + Fb_v * phi_A0_inv_phi);

    vrel = (exProj * (qnext - qlast) / (2 * dt)).sum() - (vBow + vBowLast) / 2;
    phival = bow->phi(vrel);
    Fb_v = phival * (FBow + FBowLast) / 2
           / (vrel + std::copysign(NUM_EPS, vrel));
  }

  // Right hand side
  RHS = M_qnow * qnow + M_qlast * qlast
        + exProj * Fb_v / (2 * dt) * (exProj * qlast).sum()
        + exProj * Fb_v * (vBow + vBowLast) / 2;

  // Solve using Shermann-Morrisson
  qnext = A0_inv * (RHS)-A0_inv_phi * Fb_v * (A0_inv_phi * RHS).sum()
          / (2 * dt + Fb_v * phi_A0_inv_phi);

  qlast = qnow;
  qnow = qnext;

  vBowLast = vBow;
  FBowLast = FBow;

  // out = (qnow-qlast) / dt;
  return {(qnow + qlast) / 2, (qnow - qlast) / dt * M};
};

template class BowedStringModal<double>;
template class BowedStringModal<float>;
