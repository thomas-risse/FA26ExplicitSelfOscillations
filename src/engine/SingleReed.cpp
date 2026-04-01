#include "SingleReed.h"

#include <iostream>

template <typename ftype>
SingleReed<ftype>::SingleReed(float samplerate) {
  sr = samplerate;
  dt = 1 / sr;

  resonator = std::make_shared<WebsterFDTD<ftype>>(sr);

  dissipationCoefficient = 2 * mass * damping;

  p.setZero();
  q.setZero();
  r.setZero();
  PsubCentered.setZero();
  Psub.setZero();

  kineticEnergy.setZero();
  potentialEnergy.setZero();
};

template <typename ftype>
void SingleReed<ftype>::fillOpeningAndInterpenetration() {
  interpenetration = q(idxNext) - layPosition;

  opening = width * softplus(-interpenetration, epsilonSmooth);
  interpenetration = softplus(interpenetration, epsilonSmooth);
  interpenetrationDerivative
      = softplusDerivative(interpenetration, epsilonSmooth);
}

template <typename ftype>
void SingleReed<ftype>::computeRk() {
  // Here the pressure drop is evaluated with a full timestep of lag: i.e. at
  // timestep n-1/2. Some kind of evaluation of Psup at timestep n, r better
  // timestep n+1/2 nees to be used for better results.
  meanFlow
      = opening * sqrt(2 / (kt * rho_0) * abs(Psub(idxNow) - Psup))
        * tanh(
            (Psub(idxNow) - Psup)
            / epsilonSmoothP);  // Some kind of smoothing of the sign function
  Rk = meanFlow
       / (Psub(idxNow) - Psup + std::copysign(1e-10, Psub(idxNow) - Psup));
}

template <typename ftype>
void SingleReed<ftype>::computegSAV() {
  Enl = contactStiffness * pow(interpenetration, alphaContactStiffness + 1)
        / (alphaContactStiffness + 1);

  Fnl = contactStiffness * pow(interpenetration, alphaContactStiffness)
        * interpenetrationDerivative;

  gSav = Fnl / (sqrt(2 * Enl) + 1e-14);
}

template <typename ftype>
void SingleReed<ftype>::process(float Pin) {
  // Optional computations needed for power balance variables
  mouthFlow = resonatorFlow;
  PdissFlow = Rk * pow(Psub(idxNext) - Psup, 2);
  ftype pmid = 0.5 * (p(idxNow) + p(idxNext));
  PdissReed = pow(pmid / mass, 2) * dissipationCoefficient;
  Pdiss = PdissFlow + PdissReed;

  PextSub = -mouthFlow * Psub(idxNext);
  PextSup = resonatorFlow * Psup;
  Pext = PextSub + PextSup;

  // Step 0: Advance state
  idxNow = idxNext;
  idxNext = (idxNow + 1) % 2;

  PsubCentered(idxNext) = Pin;  // P^{n+1}
  Psub(idxNext)
      = (PsubCentered(idxNext) + PsubCentered(idxNow)) * 0.5;  // P^{n+1/2}
  // Step 1: q update
  q(idxNext) = q(idxNow) + dt / mass * p(idxNow);

  // Step 2: Rk and gSav explicit computation
  fillOpeningAndInterpenetration();
  computeRk();
  computegSAV();

  // Step 3: get resonator feedback coefficients
  std::tie(aResonator, bResonator)
      = resonator->getInputLinearDependencyCoefficients();
  C0 = 1 / (Rk + 1 / bResonator)
       * (aResonator / bResonator + Rk * Psub(idxNext)
          + 0.5 * surface / mass * p(idxNow));
  C1 = 1 / (Rk + 1 / bResonator) * 0.5 / mass * surface;

  // Step 4: solve for pnext
  RHS = -stiffness * q(idxNext)
        + (1 / dt - 0.25 * dt * gSav * gSav / mass
           - dissipationCoefficient / (2 * mass))
              * p(idxNow)
        + surface * (Psub(idxNext) - C0) - gSav * r(idxNow);

  p(idxNext) = RHS
               / (1 / dt + 0.25 * dt * gSav * gSav / mass
                  + dissipationCoefficient / (2 * mass) + surface * C1);

  // Step 5: r update
  r(idxNext) = r(idxNow) + 0.5 * dt * gSav / mass * (p(idxNow) + p(idxNext));

  // Step 6: Resonator update
  Psup = C0 + C1 * p(idxNext);
  resonatorFlow = (Psup - aResonator) / bResonator;
  // resonatorFlow = Rk * (Psub(idxNext) - Psup)
  //                 + 0.5 * surface / mass * (p(idxNow) + p(idxNext));
  resonator->process(resonatorFlow);

  // Optional computations needed for power balance variables
  ftype qmid = 0.5 * (q(idxNow) + q(idxNext));
  kineticEnergy(idxNext) = 0.5 * p(idxNow)
                           * (1 - dt * dt / 4 * stiffness / mass) * p(idxNow)
                           / mass;
  potentialEnergy(idxNext)
      = 0.5 * qmid * stiffness * qmid + 0.5 * r(idxNow) * r(idxNow);

  PstoredKinetic = (kineticEnergy(idxNext) - kineticEnergy(idxNow)) / dt;
  PstoredPotential = (potentialEnergy(idxNext) - potentialEnergy(idxNow)) / dt;
  Pstored = PstoredKinetic + PstoredPotential;
};

template class SingleReed<float>;
template class SingleReed<double>;
