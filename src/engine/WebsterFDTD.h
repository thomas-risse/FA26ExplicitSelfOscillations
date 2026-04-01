#ifndef WEBSTER_FDTD_h
#define WEBSTER_FDTD_h

#include <Articulation.h>
#include <Biquad.h>

#include <Eigen/Dense>

template <typename ftype>
class WebsterFDTD {
 private:
  // Physical parameters
  ftype m_c0{340}, m_rho0{1.2}, m_l0{17e-2};  // Acoustic
  ftype m_rout{0}, m_Lrad{0}, m_Rrad{0};      // Radiation
  bool m_yieldingWalls{false};
  ftype m_mwA{15}, m_bwA{16000};  // Yielding walls, per area values
  Eigen::Array<ftype, -1, 1> m_mw, m_bw, m_kw;  // Yielding walls

  // Articulation
  Eigen::Array<ftype, -1, 1> m_Sdirect, m_Sd, m_Sp, m_Gl, m_Gg;
  Eigen::Array<ftype, -1, 1> m_SdirectLast, m_SdLast, m_SpLast;
  Eigen::Array<ftype, -1, 1> m_SdirectTarget;
  Eigen::Array<ftype, -1, 1> m_dtSp;
  float m_lambdaS{10};

  // Discretization parameters
  ftype m_dt{0}, m_sr{0}, m_h{0};
  Eigen::Array<ftype, -1, 1> m_xp, m_xd, m_xdex;
  int m_Ndis;

  // State variables
  Eigen::Array<ftype, -1, 1> m_rhonow, m_rhonext, m_v;       // Acoustic
  Eigen::Array<ftype, -1, 1> m_da, m_dotdanow, m_dotdanext;  // Walls
  ftype m_qrad;                                              // Radiation

  // LPF
  std::vector<Biquad> m_lpfFilters;
  int m_numLPF{4};  // Default number of LPF instances
  void setNumLPF(int num);

  // Update coefficients
  Eigen::Array<ftype, -1, 1> m_inter;
  Eigen::Array<ftype, -1, 1> m_dplusv, m_A, m_B, m_Ctop, m_Clow, m_D, m_E, m_F,
      m_G, m_Aqrad, m_Bqrad;

  // Set the number of discretization elements to be at the stability condition
  void setNStability();
  // Computes the discrete geometry vectors
  void computeDisS();
  bool m_timeVaryingGeometry{false};
  // Recomputes discrete wall parameters from current geometry
  void updateWallParameters();
  // Recomputes radiation parameters from current geometry
  void updateRadiationParameters();
  // Recomputes intermediary variables from current geometry
  void updateCoefficients();

 public:
  WebsterFDTD(float sampleRate);

  // Recomputes everything
  void reinitDsp(float sampleRate);

  // Setters for target geometry
  void setSTargetFromArticulation(Articulation articulation);
  void setConstantSection(ftype section);

  // Geometry filtering functions
  void setLPFFreq(int index, float freq);
  void setLPFQ(int index, float freq);
  void setAllLPFFreq(float freq);
  void setAllLPFQ(float freq);

  void filterSdirectTarget();
  void initializeLPFStates();

  // Process a sample
  void process(ftype inputFlow);

  // Linear dependency of the next input pressure (conjugated output) on the
  // next input flow (input)
  std::tuple<ftype, ftype> getInputLinearDependencyCoefficients();

  // Getters
  inline ftype getSoundVelocity() { return m_c0; }
  inline ftype getRestDensity() { return m_rho0; }
  inline ftype getLength() { return m_l0; }
  inline ftype getWallSurfaceMass() { return m_mwA; }
  inline ftype getWallSurfaceDamping() { return m_bwA; }
  inline ftype getInputPressure() { return m_c0 * m_c0 * m_rhonow(0); }
  inline ftype getRadiatedPressure() {
    return m_c0 * m_c0 * m_rhonow(m_Ndis - 1);
  }

  // Setters
  inline void setYieldingWalls(bool isYielding) {
    m_yieldingWalls = isYielding;
  }
  inline void setTimeVaryingGeometry(bool isVarying) {
    m_timeVaryingGeometry = isVarying;
  }

  void setLength(ftype length) {
    m_l0 = length;
    reinitDsp(m_sr);
  }
};

#endif