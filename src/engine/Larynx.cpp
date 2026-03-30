#include "Larynx.h"

#include <iostream>

template <typename ftype>
Larynx<ftype>::Larynx(float samplerate) {
  sr = samplerate;
  dt = 1 / sr;

  resonator = std::make_shared<WebsterFDTD<ftype>>(sr);

  A0_inv = dt;
  massMatrixInv = masses.inverse();

  // clang-format off
  elongationMatrix << 1, 0, -1,
                      0, 1, -1,
                      0, 0, 1,
                      1, -1, 0;
  // clang-format on
  stiffnessMatrix
      = elongationMatrix.transpose() * stiffnesses * elongationMatrix;

  dissipationCoefficients.diagonal()(0)
      = 2 * xi * sqrt(stiffnesses.diagonal()(0) * masses.diagonal()(0));
  dissipationCoefficients.diagonal()(1)
      = 2 * xi * sqrt(stiffnesses.diagonal()(1) * masses.diagonal()(1));
  dissipationCoefficients.diagonal()(2)
      = xi * sqrt(stiffnesses.diagonal()(2) * masses.diagonal()(2));
  dissipationCoefficients.diagonal()(3) = 0;
  dissipationMatrix = elongationMatrix.transpose() * dissipationCoefficients
                      * elongationMatrix;

  std::cout << masses.diagonal() << std::endl;

  p.setZero();
  q.setZero();
  r.setZero();
  PsubCentered.setZero();
  Psub.setZero();

  kineticEnergy.setZero();
  potentialEnergy.setZero();
};

template <typename ftype>
void Larynx<ftype>::fillMassesInterpenetrationsAndAreas() {
  massesInterpenetrations = q(idxNext, Eigen::all).transpose() - restPositions;

  areasBelowMasses = widths.cwiseProduct(
      softplusMatrix(-massesInterpenetrations,
                     epsilonSmooth));  // 2 Comes from symmetric configuration
  smoothedIsOpened
      = (-(massesInterpenetrations / epsilonSmooth).array().tanh().matrix()
         + Eigen::Vector<ftype, 3>::Ones())
        / 2;
  massesInterpenetrations
      = softplusMatrix(massesInterpenetrations, epsilonSmooth);
}

template <typename ftype>
void Larynx<ftype>::computeEffectiveAreas() {
  Aratio = areasBelowMasses(1) / (areasBelowMasses(0) + 1e-14);
  effectiveSurfacesPsub.setZero();
  effectiveSurfacesPsup.setZero();
  if (Aratio > 1) {
    effectiveSurfacesPsup
        = widths.cwiseProduct(lengths).cwiseProduct(smoothedIsOpened);
  } else {
    effectiveSurfacesPsub(0) = widths(0) * lengths(0) * (1 - Aratio * Aratio)
                               * (smoothedIsOpened(0));
    effectiveSurfacesPsup(0)
        = widths(0) * lengths(0) * (Aratio * Aratio) * (smoothedIsOpened(0));
    effectiveSurfacesPsup(1) = widths(1) * lengths(1) * (smoothedIsOpened(1));
  }
}

template <typename ftype>
void Larynx<ftype>::computeRk() {
  Amin = areasBelowMasses(Eigen::seq(0, 1)).minCoeff();
  meanFlow = Amin * sqrt(2 / (kt * rho_0) * abs(Psub(idxNow) - Psup))
             * sgn(Psub(idxNow) - Psup);
  Rk = meanFlow
       / (Psub(idxNow) - Psup + std::copysign(1e-14, Psub(idxNow) - Psup));
}

template <typename ftype>
void Larynx<ftype>::computegSAV() {
  elongations = elongationMatrix * q(idxNext, Eigen::all).transpose();

  Enl = 0.25 * etaStiffness
        * (stiffnesses.diagonal().array() * elongations.array()
           * elongations.array() * elongations.array() * elongations.array())
              .sum();
  Enl += 0.5
             * (contactStiffness
                * (massesInterpenetrations(0) * massesInterpenetrations(0)
                   + massesInterpenetrations(1) * massesInterpenetrations(1)))
         + contactStiffness * etaContactStiffness
               * (pow(massesInterpenetrations(0), alphaContactStiffness + 1)
                  + pow(massesInterpenetrations(1), alphaContactStiffness + 1))
               / (alphaContactStiffness + 1);  // Contact

  Fnl = etaStiffness * elongationMatrix.transpose()
        * (stiffnesses.diagonal().array() * elongations.array()
           * elongations.array() * elongations.array())
              .matrix();
  Fnl(0) += contactStiffness
            * (etaContactStiffness
                   * pow(massesInterpenetrations(0), alphaContactStiffness)
               + massesInterpenetrations(0));
  Fnl(1) += contactStiffness
            * (etaContactStiffness
                   * pow(massesInterpenetrations(1), alphaContactStiffness)
               + massesInterpenetrations(1));

  gSav = Fnl / (sqrt(2 * Enl) + 1e-14);
}

template <typename ftype>
void Larynx<ftype>::process(float Pin) {
  // Optional computations needed for power balance variables
  subGlottalFlow
      = -Rk * (Psub(idxNext) - Psup)
        + 0.5 * effectiveSurfacesPsub.transpose() * massMatrixInv
              * (p(idxNow, Eigen::all) + p(idxNext, Eigen::all)).transpose();
  PdissFlow = Rk * pow(Psub(idxNext) - Psup, 2);
  auto pmid
      = 0.5 * (p(idxNow, Eigen::all) + p(idxNext, Eigen::all)).transpose();
  PdissFolds = pmid.transpose() * massMatrixInv * dissipationMatrix
               * massMatrixInv * pmid;
  Pdiss = PdissFlow + PdissFolds;

  PextSub = subGlottalFlow * Psub(idxNext);
  PextSup = supGlottalFlow * Psup;
  Pext = PextSub + PextSup;

  // Step 0: Advance state
  idxNow = idxNext;
  idxNext = (idxNow + 1) % 2;

  PsubCentered(idxNext) = Pin;  // P^{n+1}
  Psub(idxNext)
      = (PsubCentered(idxNext) + PsubCentered(idxNow)) * 0.5;  // P^{n+1/2}
  // Step 1: q update
  q(idxNext, Eigen::all)
      = q(idxNow, Eigen::all).transpose()
        + dt * massMatrixInv * p(idxNow, Eigen::all).transpose();

  // Step 2: Rk and gSav explicit computation
  fillMassesInterpenetrationsAndAreas();
  computeEffectiveAreas();
  computeRk();
  computegSAV();

  // Step 3: get resonator feedback coefficients
  std::tie(aResonator, bResonator)
      = resonator->getInputLinearDependencyCoefficients();
  C0 = 1 / (Rk + 1 / bResonator)
       * (aResonator / bResonator + Rk * Psub(idxNext)
          + 0.5 * effectiveSurfacesPsup.transpose() * massMatrixInv
                * p(idxNow, Eigen::all).transpose());
  C1 = 1 / (Rk + 1 / bResonator) * 0.5 * massMatrixInv * effectiveSurfacesPsup;

  // // Step 4: solve for pnext using Woodburry
  RHS = -stiffnessMatrix * q(idxNext, Eigen::all).transpose()
        + (1 / dt * Eigen::Matrix<ftype, 3, 3>::Identity()
           - dissipationMatrix * massMatrixInv)
              * p(idxNow, Eigen::all).transpose()
        - dt / 4 * gSav
              * (gSav.transpose() * massMatrixInv
                 * p(idxNow, Eigen::all).transpose())
        - gSav * r[idxNow] - effectiveSurfacesPsub * Psub(idxNext)
        - effectiveSurfacesPsup * C0;

  UWoodburry.col(0) = dt / 4 * gSav;
  UWoodburry.col(1) = effectiveSurfacesPsup;

  VWoodburry.row(0) = massMatrixInv * gSav;
  VWoodburry.row(1) = C1;
  woodburryInverse = (Eigen::Matrix<ftype, 2, 2>::Identity()
                      + VWoodburry * A0_inv * UWoodburry)
                         .inverse();

  p(idxNext, Eigen::all)
      = A0_inv * RHS
        - A0_inv * A0_inv * UWoodburry * woodburryInverse * (VWoodburry * RHS);

  // Step 5: r update
  r(idxNext)
      = r(idxNow)
        + 0.5 * dt * gSav.transpose() * massMatrixInv
              * (p(idxNow, Eigen::all) + p(idxNext, Eigen::all)).transpose();

  // Step 6: Resonator update
  Psup = C0 + C1.transpose() * p(idxNext, Eigen::all).transpose();
  supGlottalFlow = (Psup - aResonator) / bResonator;
  // supGlottalFlow
  //     = Rk * (Psub(idxNext) - Psup)
  //       + 0.5 * effectiveSurfacesPsup.transpose() * massMatrixInv
  //             * (p(idxNow, Eigen::all) + p(idxNext, Eigen::all)).transpose();
  resonator->process(supGlottalFlow);

  // Optional computations needed for power balance variables
  auto qmid
      = 0.5 * (q(idxNow, Eigen::all) + q(idxNext, Eigen::all)).transpose();
  kineticEnergy(idxNext) = 0.5 * p(idxNow, Eigen::all)
                           * (Eigen::Matrix<ftype, 3, 3>::Identity()
                              - dt * dt / 4 * massMatrixInv * stiffnessMatrix
                              - dt / 2 * massMatrixInv * dissipationMatrix)
                           * massMatrixInv * p(idxNow, Eigen::all).transpose();
  potentialEnergy(idxNext) = 0.5 * qmid.transpose() * stiffnessMatrix * qmid
                             + 0.5 * r(idxNow) * r(idxNow);

  PstoredKinetic = (kineticEnergy(idxNext) - kineticEnergy(idxNow)) / dt;
  PstoredPotential = (potentialEnergy(idxNext) - potentialEnergy(idxNow)) / dt;
  Pstored = PstoredKinetic + PstoredPotential;
};

template class Larynx<float>;
template class Larynx<double>;
