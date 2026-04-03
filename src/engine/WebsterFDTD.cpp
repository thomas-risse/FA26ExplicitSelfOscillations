#include "WebsterFDTD.h"

#include <iostream>

template <typename ftype>
WebsterFDTD<ftype>::WebsterFDTD(float sampleRate) {
  reinitDsp(sampleRate);
}

template <typename ftype>
void WebsterFDTD<ftype>::reinitDsp(float sampleRate) {
  m_sr = sampleRate;
  m_dt = 1 / m_sr;
  setNStability();
  m_xdex = Eigen::Array<ftype, -1, 1>::LinSpaced(m_Ndis + 1, -m_h / 2,
                                                 m_l0 + m_h / 2);
  m_xd = m_xdex.segment(1, m_Ndis - 1);
  m_xp = Eigen::Array<ftype, -1, 1>::LinSpaced(m_Ndis, 0, m_l0);

  m_SdirectTarget.resize(m_Ndis + 1);
  m_SdirectTarget.setOnes();
  m_Sdirect.resize(m_Ndis + 1);
  m_Sd.resize(m_Ndis - 1);
  m_Sp.resize(m_Ndis);
  m_Sdirect = m_SdirectTarget;
  m_SdirectLast = m_Sdirect;
  computeDisS();
  m_SdLast = m_Sd;
  m_SpLast = m_Sp;
  m_dtSp.resize(m_Ndis);
  m_dtSp.setZero();

  m_Gl.resize(m_Ndis);
  m_Gl.setZero();
  m_Gl(m_Ndis - 1) = 1;
  m_Gg.resize(m_Ndis);
  m_Gg.setZero();
  m_Gg(0) = 1;

  m_mw.resize(m_Ndis);
  m_kw.resize(m_Ndis);
  m_bw.resize(m_Ndis);
  updateWallParameters();
  updateRadiationParameters();

  m_dplusv.resize(m_Ndis);
  m_dplusv.setZero();
  m_inter.resize(m_Ndis);
  m_inter.setZero();
  m_A.resize(m_Ndis);
  m_A.setZero();
  m_B.resize(m_Ndis);
  m_B.setZero();
  m_Ctop.resize(m_Ndis - 1);
  m_Ctop.setZero();
  m_Clow.resize(m_Ndis - 1);
  m_Clow.setZero();
  m_D.resize(m_Ndis);
  m_D.setZero();
  m_E.resize(m_Ndis);
  m_E.setZero();
  m_F.resize(m_Ndis);
  m_F.setZero();
  m_G.resize(m_Ndis);
  m_G.setZero();
  m_Aqrad.resize(m_Ndis);
  m_Aqrad.setZero();
  m_Bqrad.resize(m_Ndis);
  m_Bqrad.setZero();

  m_rhonow.resize(m_Ndis);
  m_rhonow.setZero();
  m_rhonext.resize(m_Ndis);
  m_rhonext.setZero();
  m_v.resize(m_Ndis - 1);
  m_v.setZero();
  m_da.resize(m_Ndis);
  m_da.setZero();
  m_dotdanow.resize(m_Ndis);
  m_dotdanow.setZero();
  m_dotdanext.resize(m_Ndis);
  m_dotdanext.setZero();
  m_qrad = 0;

  updateCoefficients();

  // Initialize LPF filters
  m_numLPF = m_Ndis + 1;
  m_lpfFilters.resize(m_numLPF);
  for (int i = 0; i < m_numLPF; ++i) {
    m_lpfFilters[i] = Biquad(m_sr, bq_type_lowpass, 10.0f, 0.0f, 0.5f);
    // Initialize filter state to corresponding SdirectTarget value
    if (i < m_SdirectTarget.size()) {
      m_lpfFilters[i].initializeState(static_cast<double>(m_SdirectTarget[i]));
    }
  }
  setAllLPFFreq(10.0);
  setAllLPFQ(0.5);
}

template <typename ftype>
void WebsterFDTD<ftype>::setNStability() {
  m_h = m_c0 / m_sr;
  m_Ndis = std::floor(m_l0 / m_h);
  m_h = m_l0 / m_Ndis;
}

template <typename ftype>
void WebsterFDTD<ftype>::setSTargetFromArticulation(Articulation articulation) {
  articulation.getAreas(m_xdex.data(), m_SdirectTarget.data(), m_Ndis + 1);
  if (!m_timeVaryingGeometry) {
    m_Sdirect = m_SdirectTarget;
    computeDisS();
    updateCoefficients();
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::setConstantSection(ftype section) {
  m_SdirectTarget = Eigen::Array<ftype, -1, 1>::Ones(m_Ndis + 1) * section;
  if (!m_timeVaryingGeometry) {
    m_Sdirect = m_SdirectTarget;
    computeDisS();
    updateRadiationParameters();
    updateCoefficients();
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::computeDisS() {
  // Sd = 0.25 * (Sdirect.segment(0, N-1) + 2*Sdirect.segment(1, N-1) +
  // Sdirect.segment(2, N-1));
  m_Sdirect(0) = m_Sdirect(1);
  // Sdirect(N-1) = 1e-10;
  m_Sdirect(m_Ndis) = m_Sdirect(m_Ndis - 1);
  m_Sd = m_Sdirect.segment(1, m_Ndis - 1);
  m_Sp = 0.5 * (m_Sdirect.segment(0, m_Ndis) + m_Sdirect.segment(1, m_Ndis));
  // Sp.segment(1, N-2) = 0.25 * (Sp.segment(0, N-2) + 2*Sp.segment(1, N-2) +
  // Sp.segment(2, N-2));
}

template <typename ftype>
void WebsterFDTD<ftype>::updateWallParameters() {
  m_mw = m_mwA * 2 * sqrt(m_Sp * M_PI);
  m_kw = m_mw * (2 * M_PI * 70) * (2 * M_PI * 70);
  m_bw = m_bwA * 2 * sqrt(m_Sp * M_PI);
}

template <typename ftype>
void WebsterFDTD<ftype>::updateRadiationParameters() {
  ftype Aout = m_Sp[m_Ndis - 1];
  m_rout = sqrt(Aout / M_PI);
  ftype Z0 = m_rho0 * m_c0 / Aout;
  m_Lrad = 8 * m_rout / (3 * M_PI * m_c0) * Z0;
  m_Rrad = 128 / (9 * M_PI * M_PI) * Z0;
}

template <typename ftype>
void WebsterFDTD<ftype>::updateCoefficients() {
  if (m_yieldingWalls) {
    ftype rhoc2 = m_rho0 * m_c0 * m_c0;
    m_inter = 2 / m_dt + m_bw / m_mw + m_kw / (2 * m_mw) * m_dt;

    m_A = (1 / m_dt + 4 * M_PI * rhoc2 / (2 * m_mw * m_inter)
           + rhoc2 * m_Gl / (m_Sp * 2 * m_h * m_Rrad)
           + rhoc2 * m_Gl * m_dt / (m_Sp * 4 * m_h * m_Lrad));
    m_B = (1 / m_dt - 4 * M_PI * rhoc2 / (2 * m_mw * m_inter)
           - rhoc2 * m_Gl / (m_Sp * 2 * m_h * m_Rrad)
           - rhoc2 * m_Gl * m_dt / (m_Sp * 4 * m_h * m_Lrad));
    m_Ctop = -1 / m_Sp.head(m_Ndis - 1) * m_rho0 / m_h * m_Sd;
    m_Clow = 1 / m_Sp.tail(m_Ndis - 1) * m_rho0 / m_h * m_Sd;

    m_D = -2 * m_rho0 / (m_dt * m_inter * m_Sp);
    m_E = m_rho0 / (m_inter * m_Sp) * m_kw / (m_mw);
    m_F = -m_rho0 * m_Gl / (m_h * m_Sp);
    m_G = m_rho0 * m_Gg / (m_h * m_Sp);

    m_Aqrad = 1 / m_dt + m_bw / (2 * m_mw) + m_kw / (4 * m_mw) * m_dt;
    m_Bqrad = 1 / m_dt - m_bw / (2 * m_mw) - m_kw / (4 * m_mw) * m_dt;
  } else {
    ftype rhoc2 = m_rho0 * m_c0 * m_c0;
    m_inter = 2 / m_dt;

    m_A = (1 / m_dt + rhoc2 * m_Gl / (m_Sp * 2 * m_h * m_Rrad)
           + rhoc2 * m_Gl * m_dt / (m_Sp * 4 * m_h * m_Lrad));
    m_B = (1 / m_dt - rhoc2 * m_Gl / (m_Sp * 2 * m_h * m_Rrad)
           - rhoc2 * m_Gl * m_dt / (m_Sp * 4 * m_h * m_Lrad));
    m_Ctop = -1 / m_Sp.head(m_Ndis - 1) * m_rho0 / m_h * m_Sd;
    m_Clow = 1 / m_Sp.tail(m_Ndis - 1) * m_rho0 / m_h * m_Sd;

    m_D = -2 * m_rho0 / (m_dt * m_inter * m_Sp);
    m_E.setZero();
    m_F = -m_rho0 * m_Gl / (m_h * m_Sp);
    m_G = m_rho0 * m_Gg / (m_h * m_Sp);

    m_Aqrad = 1 / m_dt;
    m_Bqrad = 1 / m_dt;
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::process(ftype inputFlow) {
  if (m_yieldingWalls) {
    m_dplusv.setZero();
    m_dplusv.head(m_Ndis - 1) = m_Ctop * m_v;
    m_dplusv.tail(m_Ndis - 1) += m_Clow * m_v;
    m_rhonext
        = (1 / m_A)
          * (m_B * m_rhonow + m_dplusv + m_D * m_dotdanow + m_E * m_da
             + m_F * m_qrad + m_G * inputFlow - m_rho0 * (1 / m_Sp * m_dtSp));
    m_v = (1 / (1 / m_dt))
          * ((1 / m_dt) * m_v
             - m_c0 * m_c0 / (m_rho0) / m_h
                   * (m_rhonext.tail(m_Ndis - 1) - m_rhonext.head(m_Ndis - 1)));
    m_dotdanext
        = (1 / m_Aqrad)
          * (m_Bqrad * m_dotdanow - (m_kw / m_mw) * (m_da)
             + (m_Sp * m_c0 * m_c0 / m_mw * 0.5) * (m_rhonow + m_rhonext));
    m_da = m_da + m_dt * 0.5 * (m_dotdanow + m_dotdanext);
    m_qrad = m_qrad
             + m_dt * m_c0 * m_c0 / m_Lrad * 0.5
                   * (m_rhonext(m_Ndis - 1) + m_rhonow(m_Ndis - 1));
  } else {
    m_dplusv.setZero();
    m_dplusv.head(m_Ndis - 1) = m_Ctop * m_v;
    m_dplusv.tail(m_Ndis - 1) += m_Clow * m_v;
    m_rhonext = (1 / m_A)
                * (m_B * m_rhonow + m_dplusv + m_F * m_qrad + m_G * inputFlow
                   - m_rho0 * (1 / m_Sp * m_dtSp));
    m_v = (1 / (1 / m_dt))
          * ((1 / m_dt) * m_v
             - m_c0 * m_c0 / (m_rho0) / m_h
                   * (m_rhonext.tail(m_Ndis - 1) - m_rhonext.head(m_Ndis - 1)));
    m_qrad = m_qrad
             + m_dt * m_c0 * m_c0 / m_Lrad * 0.5
                   * (m_rhonext(m_Ndis - 1) + m_rhonow(m_Ndis - 1));
  }
  m_rhonow = m_rhonext;
  m_dotdanow = m_dotdanext;

  m_SdirectLast = m_Sdirect;
  m_SdLast = m_Sd;
  m_SpLast = m_Sp;

  if (m_timeVaryingGeometry) {
    // Apply filtering to each element of SdirectTarget
    for (int i = 0; i < m_SdirectTarget.size() && i < m_numLPF; i++) {
      m_Sdirect[i] = static_cast<ftype>(
          m_lpfFilters[i].process(static_cast<double>(m_SdirectTarget[i])));
      // Sdirect[i] = SdirectTarget[i];
    }

    computeDisS();
    updateWallParameters();
    updateRadiationParameters();
    updateCoefficients();

    m_dtSp = (m_Sp - m_SpLast) / m_dt;
  }
}

template <typename ftype>
std::tuple<ftype, ftype>
WebsterFDTD<ftype>::getInputLinearDependencyCoefficients() {
  return {m_c0 * m_c0 * 0.5
              * (m_rhonow(0)
                 + 1 / m_A(0)
                       * (m_B(0) * m_rhonow(0)
                          - m_rho0 / (m_h)*m_Sd(0) / m_Sp(0) * m_v(0)
                          - m_rho0 * (1 / m_Sp(0) * m_dtSp(0)))),
          0.5 * m_c0 * m_c0 * (1 / m_A(0)) * m_G(0)};
}

// LPF filter methods implementation
template <typename ftype>
void WebsterFDTD<ftype>::setNumLPF(int num) {
  m_numLPF = std::max(1, num);
  m_lpfFilters.resize(m_numLPF);
  for (int i = 0; i < m_numLPF; ++i) {
    m_lpfFilters[i] = Biquad(m_sr, bq_type_lowpass, 10.0f, 0.0f, 0.7f);
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::setLPFFreq(int index, float freq) {
  if (index >= 0 && index < m_numLPF) {
    m_lpfFilters[index].setFreq(freq);
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::setLPFQ(int index, float Q) {
  if (index >= 0 && index < m_numLPF) {
    m_lpfFilters[index].setQ(Q);
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::setAllLPFFreq(float freq) {
  for (int i = 0; i < m_numLPF; ++i) {
    m_lpfFilters[i].setFreq(freq);
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::setAllLPFQ(float Q) {
  for (int i = 0; i < m_numLPF; ++i) {
    m_lpfFilters[i].setQ(Q);
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::filterSdirectTarget() {
  for (int i = 0; i < m_SdirectTarget.size() && i < m_numLPF; ++i) {
    m_SdirectTarget[i] = static_cast<ftype>(
        m_lpfFilters[i].process(static_cast<double>(m_SdirectTarget[i])));
  }
}

template <typename ftype>
void WebsterFDTD<ftype>::initializeLPFStates() {
  for (int i = 0; i < m_numLPF && i < m_SdirectTarget.size(); ++i) {
    m_lpfFilters[i].initializeState(static_cast<double>(m_SdirectTarget[i]));
  }
}

template class WebsterFDTD<float>;
template class WebsterFDTD<double>;
