#ifndef LARYNX_H
#define LARYNX_H

#include <EigenUtility.h>
#include <WebsterFDTD.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

template <typename ftype>

class SingleReed {
 private:
  // Physical parameters
  ftype mass{3.37e-6}, stiffness{1821}, damping{1500}, width{1e-2},
      surface{1.46e-4}, layPosition{4e-4};                   // Reed
  ftype contactStiffness{1e13}, alphaContactStiffness{1.3};  // Contact
  ftype rho_0{1.2}, c_0{340}, kt{1.3};                       // Fluid

  // Intermediary quantities
  ftype dissipationCoefficient;  // R_0 matrix
  ftype opening, interpenetration;
  ftype interpenetrationDerivative;

  ftype gSav{0}, Enl{0}, Fnl{0};

  ftype meanFlow, Rk;

  ftype aResonator, bResonator;
  ftype C0, C1;

  ftype RHS;

  ftype epsilonSmooth{1e-7}, epsilonSmoothP{1};

  // State
  Eigen::Vector<ftype, 2> p, q, r;
  std::size_t idxNow{0}, idxNext{1};

  Eigen::Vector<ftype, 2> Psub, PsubCentered;
  ftype Psup;

  // Power variables
  Eigen::Vector<ftype, 2> kineticEnergy, potentialEnergy;
  ftype Pdiss, PdissFlow, PdissReed;
  ftype Pext, PextSub, PextSup;
  ftype Pstored, PstoredPotential, PstoredKinetic;

  // Resonator
  ftype mouthFlow{0}, resonatorFlow{0};
  std::shared_ptr<WebsterFDTD<ftype>> resonator;

  // Solver parameters
  ftype sr, dt;

  // Functions
  void fillOpeningAndInterpenetration();
  void computeRk();
  void computegSAV();

 public:
  SingleReed(float samplerate);

  void process(float Pin);

  inline ftype getCurrentDisplacement() {
    // Midpoint to evaluate on the same grid as inputs and momentums.
    return (q(idxNow) + q(idxNext)) * 0.5;
  };

  inline ftype getCurrentEffectiveOpening() {
    // Midpoint to evaluate on the same grid as inputs and momentums.
    return opening;
  };

  inline ftype getRadiatedPressure() {
    return resonator->getRadiatedPressure();
  }

  inline ftype getLayPosition() { return layPosition; };

  // Power variables
  std::tuple<ftype, ftype, ftype> getCurrentDissipatedPowers() {
    return {Pdiss, PdissFlow, PdissReed};
  };
  std::tuple<ftype, ftype, ftype> getCurrentExchangedPowers() {
    return {Pext, PextSub, PextSup};
  };
  std::tuple<ftype, ftype, ftype> getCurrentStoredPowers() {
    return {Pstored, PstoredKinetic, PstoredPotential};
  };

  // Flow variables
  inline ftype getCurrentMeanFlow() { return meanFlow; };
  inline ftype getCurrentResonatorFlow() { return resonatorFlow; };
  inline ftype getCurrentPressureDrop() { return Psub(idxNow) - Psup; };

  std::shared_ptr<WebsterFDTD<ftype>> getResonator() { return resonator; }
};

#endif
