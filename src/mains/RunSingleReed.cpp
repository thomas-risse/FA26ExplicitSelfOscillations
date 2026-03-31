#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

#include "ResultsStorage.h"
#include "SingleReed.h"

using namespace std::chrono;

int main(int argc, char const* argv[]) {
  // Get filename from command line argument or use default
  std::string filename
      = "/Users/risse/Projects/FA26ExplicitSelfOscillations/python/results/"
        "testSingleReed.hdf5";
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
  SingleReed<double> proc(sr);
  proc.getResonator()->setLength(0.66);                  // m
  proc.getResonator()->setConstantSection(M_PI * 1e-4);  // m^2

  // Run a simulation
  Eigen::VectorXd Pmouth
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  storage.readVector("Pmouth", Pmouth);

  // Storage
  Eigen::VectorXd reedDisplacement
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd effectiveOpening
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd meanFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd resonatorFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd PextSub
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd PextSup
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd Pext
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));

  Eigen::VectorXd PdissFlow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd PdissReed
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
    reedDisplacement(i) = proc.getCurrentDisplacement();
    meanFlow(i) = proc.getCurrentMeanFlow();
    resonatorFlow(i) = proc.getCurrentResonatorFlow();
    effectiveOpening(i) = proc.getCurrentEffectiveOpening();

    std::tie(Pext(i), PextSub(i), PextSup(i))
        = proc.getCurrentExchangedPowers();
    std::tie(Pdiss(i), PdissFlow(i), PdissReed(i))
        = proc.getCurrentDissipatedPowers();
    std::tie(Pstored(i), PstoredKinetic(i), PstoredPotential(i))
        = proc.getCurrentStoredPowers();
  }
  auto stop = high_resolution_clock::now();
  float rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6
                  / simDuration;

  std::cout << "Real-time ratio: " << rtRatio * 100 << "%" << std::endl;

  // Write results

  storage.writeVector("reedDisplacement", reedDisplacement);
  storage.writeVector("effectiveOpening", effectiveOpening);

  storage.writeAttribute("layPosition", proc.getLayPosition());
  storage.writeVector("meanFlow", meanFlow);
  storage.writeVector("resonatorFlow", resonatorFlow);

  storage.writeVector("Pext", Pext);
  storage.writeVector("PextSub", PextSub);
  storage.writeVector("PextSup", PextSup);

  storage.writeVector("Pdiss", Pdiss);
  storage.writeVector("PdissFlow", PdissFlow);
  storage.writeVector("PdissReed", PdissReed);

  storage.writeVector("Pstored", Pstored);
  storage.writeVector("PstoredKinetic", PstoredKinetic);
  storage.writeVector("PstoredPotential", PstoredPotential);

  storage.writeVector("time", time);

  storage.close();
  return 0;
}
