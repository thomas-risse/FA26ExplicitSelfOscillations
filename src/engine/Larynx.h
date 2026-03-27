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

class Larynx {
 private:
  // Physical parameters
  // Vocal folds: order is lower-upper-body
  Eigen::DiagonalMatrix<ftype, 3> masses{1e-5, 1e-5, 5e-5};
  Eigen::Vector<ftype, 3> lengths{1.5e-3, 1.5e-3, 3e-3};
  Eigen::Vector<ftype, 3> widths{1e-2, 1e-2, 0};
  Eigen::Vector<ftype, 3> restPositions{1.8e-4, 1.79e-4, 3e-3};

  Eigen::Vector<ftype, 4> elongations;
  Eigen::DiagonalMatrix<ftype, 4> stiffnesses{
      5, 3.5, 20, 0.5};  // lower, upper, body, lower-upper
  ftype etaStiffness{1e6};

  ftype xi{0.4};
  Eigen::DiagonalMatrix<ftype, 4> dissipationCoefficients;

  ftype contactStiffness{15}, etaContactStiffness{5e6},
      alphaContactStiffness{3};

  ftype rho_0{1.2}, c_0{340}, kt{1.3};

  // Matrices and intermediary quantities
  Eigen::Matrix<ftype, 4, 3> elongationMatrix;
  Eigen::DiagonalMatrix<ftype, 3> massMatrixInv;
  Eigen::Matrix<ftype, 3, 3> stiffnessMatrix, dissipationMatrix;
  Eigen::Vector<ftype, 3> massesInterpenetrations, areasBelowMasses;
  Eigen::Vector<ftype, 3> effectiveSurfacesPsub, effectiveSurfacesPsup;

  Eigen::Vector<ftype, 3> gSav{0, 0, 0}, Fnl{0, 0, 0};
  ftype Enl;

  ftype Aratio, Amin;
  ftype meanFlow, Rk;

  ftype aResonator, bResonator;
  ftype C0;
  Eigen::Vector<ftype, 3> C1;

  ftype A0_inv;
  Eigen::Vector<ftype, 3> RHS;
  Eigen::Matrix<ftype, 2, 3> VWoodburry;
  Eigen::Matrix<ftype, 3, 2> UWoodburry;
  Eigen::Matrix<ftype, 2, 2> woodburryInverse;

  // State
  Eigen::Matrix<ftype, 2, 3> p, q;
  Eigen::Vector<ftype, 2> r;
  std::size_t idxNow{0}, idxNext{1};

  Eigen::Vector<ftype, 2> Psub;
  ftype Psup;

  // Power variables
  Eigen::Vector<ftype, 2> kineticEnergy, potentialEnergy;
  ftype Pdiss, PdissFlow, PdissFolds;
  ftype Pext, PextSub, PextSup;
  ftype Pstored, PstoredPotential, PstoredKinetic;

  // Resonator
  ftype subGlottalFlow{0}, supGlottalFlow{0};
  std::shared_ptr<WebsterFDTD<ftype>> resonator;

  // Solver parameters
  ftype sr, dt;

  // Functions
  void fillMassesInterpenetrationsAndAreas();
  void computeEffectiveAreas();
  void computeRk();
  void computegSAV();

 public:
  Larynx(float samplerate);

  void process(float Pin);

  inline Eigen::Vector<ftype, 3> getCurrentFoldDisplacement() {
    return q(idxNow, Eigen::all);
  };

  inline Eigen::Vector<ftype, 3> getRestPositions() { return restPositions; };

  // Power variables
  std::tuple<ftype, ftype, ftype> getCurrentDissipatedPowers() {
    return {Pdiss, PdissFlow, PdissFolds};
  };
  std::tuple<ftype, ftype, ftype> getCurrentExchangedPowers() {
    return {Pext, PextSub, PextSup};
  };
  std::tuple<ftype, ftype, ftype> getCurrentStoredPowers() {
    return {Pstored, PstoredKinetic, PstoredPotential};
  };

  // Flow variables
  inline ftype getCurrentSupGlottalFlow() { return supGlottalFlow; };
  inline ftype getCurrentMeanGlottalFlow() { return meanFlow; };
  inline ftype getCurrentPressureDrop() { return Psub(idxNow) - Psup; };

  std::shared_ptr<WebsterFDTD<ftype>> getResonator() { return resonator; }
};

#endif
