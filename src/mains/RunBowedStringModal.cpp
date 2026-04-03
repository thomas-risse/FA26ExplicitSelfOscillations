#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

#include "BowedStringModal.h"
#include "ResultsStorage.h"

using namespace std::chrono;

int main(int argc, char const* argv[]) {
  // Get filename from command line argument or use default
  std::string filename
      = "/Users/risse/Projects/NonlinearDissipations/python/testcpp.hdf5";
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

  // Bow instanciation
  BOWMODE bowMode = VIGUE;
  double mu_s = 0.4;
  double mu_d = 0.2;
  double n_bow = 100;
  double epsilon_bow = 1e-4;

  storage.readAttribute("mu_s", mu_s);
  storage.readAttribute("mu_d", mu_d);
  storage.readAttribute("epsilon", epsilon_bow);
  storage.readAttribute("n", n_bow);

  std::shared_ptr<Bow<double>> bow = std::make_shared<Bow<double>>();
  bow->setBowMode(bowMode);
  bow->setVigueParams(mu_s, mu_d, n_bow, epsilon_bow);

  // Model instanciation
  int Nmodes;
  storage.readAttribute("Nmodes", Nmodes);
  int dissipationMode;
  storage.readAttribute("DissipationMode", dissipationMode);
  BowedStringModal<double> proc(sr, bow, Nmodes, dissipationMode);

  Eigen::VectorXd M, K, R;
  storage.readVector("M", M);
  storage.readVector("K", K);
  storage.readVector("R", R);

  proc.setPhysicalParameters(M, K, R);

  double exPos;
  storage.readAttribute("ExPos", exPos);
  proc.setExPos(exPos);
  // Run a simulation
  Eigen::VectorXd vBow
      = Eigen::VectorXd::Zero(static_cast<int>(sr * simDuration));
  Eigen::VectorXd FBow{vBow};

  storage.readVector("vctrl", vBow);
  storage.readVector("Fctrl", FBow);

  Eigen::MatrixXd qout
      = Eigen::MatrixXd::Zero(static_cast<int>(sr * simDuration), Nmodes);
  Eigen::MatrixXd pout
      = Eigen::MatrixXd::Zero(static_cast<int>(sr * simDuration), Nmodes);

  auto start = high_resolution_clock::now();
  for (int i = 0; i < sr * simDuration; i++) {
    auto out = proc.process(vBow(i), FBow(i));
    qout.row(i) = std::get<0>(out);
    pout.row(i) = std::get<1>(out);
  }
  auto stop = high_resolution_clock::now();
  float rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6
                  / simDuration;

  std::cout << "Real-time ratio: " << rtRatio * 100 << "%" << std::endl;

  // Write results

  storage.writeMatrix("q", qout);
  storage.writeMatrix("p", pout);
  Eigen::MatrixXd v = (pout.array().rowwise()) / M.array().transpose();
  storage.writeMatrix("v", v);

  storage.close();
  return 0;
}
