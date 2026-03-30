#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

#include "Larynx.h"
#include "ResultsStorage.h"

using namespace std::chrono;

int main(int argc, char const* argv[]) {
  // Get filename from command line argument or use default
  std::string filename
      = "/Users/risse/Projects/FA26ExplicitSelfOscillations/python/results/"
        "testVoice.hdf5";
  if (argc > 1) {
    filename = argv[1];
  }

  // HDF5 file
  ResultsStorage storage(filename);

  // Samplerate
  float sr;
  float simDuration;

  storage.readAttribute("sr", sr);
  storage.readAttribute("duration", simDuration);

  // Model instanciation
  Larynx<double> proc(sr);
  proc.getResonator()->setLength(17e-2);
  proc.getResonator()->setConstantSection(25e-4);

  // Run a simulation
  Eigen::VectorXd Pmouth
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  storage.readVector("Pmouth", Pmouth);

  // Storage
  Eigen::MatrixXd foldDisplacement
      = Eigen::MatrixXd::Zero(static_cast<int>(sr * simDuration), 3);
  Eigen::MatrixXd effectiveOpening
      = Eigen::MatrixXd::Zero(static_cast<int>(sr * simDuration), 3);
  Eigen::VectorXd supGlottalFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd meanGlottalFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd pressureDrop
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd PextSub
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd PextSup
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd Pext
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd PdissFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd PdissFolds
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd Pdiss
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd PstoredKinetic
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd PstoredPotential
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd Pstored
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd time
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  time.setLinSpaced(0, simDuration);

  // Simulation
  auto start = high_resolution_clock::now();
  for (int i = 0; i < sr * simDuration; i++) {
    if (i + 1 < sr * simDuration) {
      proc.process(Pmouth(i + 1));
    } else {
      proc.process(Pmouth(i));  // Duplicate last input for simplicity
    }
    foldDisplacement.row(i) = proc.getCurrentFoldDisplacement();
    supGlottalFlow(i) = proc.getCurrentSupGlottalFlow();
    meanGlottalFlow(i) = proc.getCurrentMeanGlottalFlow();
    pressureDrop(i) = proc.getCurrentPressureDrop();
    effectiveOpening.row(i) = proc.getCurrentEffectiveOpening();

    std::tie(Pext(i), PextSub(i), PextSup(i))
        = proc.getCurrentExchangedPowers();
    std::tie(Pdiss(i), PdissFlow(i), PdissFolds(i))
        = proc.getCurrentDissipatedPowers();
    std::tie(Pstored(i), PstoredKinetic(i), PstoredPotential(i))
        = proc.getCurrentStoredPowers();
  }
  auto stop = high_resolution_clock::now();
  float rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6
                  / simDuration;

  std::cout << "Real-time ratio: " << rtRatio * 100 << "%" << std::endl;

  // Write results

  storage.writeMatrix("foldDisplacement", foldDisplacement);
  storage.writeMatrix("effectiveOpening", effectiveOpening);
  Eigen::VectorXd restPositions = proc.getRestPositions();
  storage.writeVector("restPositions", restPositions);
  storage.writeVector("supGlottalFlow", supGlottalFlow);
  storage.writeVector("meanGlottalFlow", meanGlottalFlow);
  storage.writeVector("pressureDrop", pressureDrop);

  storage.writeVector("Pext", Pext);
  storage.writeVector("PextSub", PextSub);
  storage.writeVector("PextSup", PextSup);

  storage.writeVector("Pdiss", Pdiss);
  storage.writeVector("PdissFlow", PdissFlow);
  storage.writeVector("PdissFolds", PdissFolds);

  storage.writeVector("Pstored", Pstored);
  storage.writeVector("PstoredKinetic", PstoredKinetic);
  storage.writeVector("PstoredPotential", PstoredPotential);

  storage.writeVector("time", time);

  storage.close();
  return 0;
}
