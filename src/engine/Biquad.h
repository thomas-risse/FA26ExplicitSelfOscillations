#ifndef BIQUAD_H
#define BIQUAD_H

#include <algorithm>
#include <cmath>

enum {
  bq_type_lowpass = 0,
  bq_type_highpass,
  bq_type_bandpass,
  bq_type_notch,
  bq_type_peak,
  bq_type_lowshelf,
  bq_type_highshelf
};

class Biquad {
 private:
  double x0{0}, x1{0}, x2{0}, y0{0}, y1{0}, y2{0};
  float sr{44100};

  float freq{100}, dBgain{0}, Q{0.7};

  double A, omega, sinomega, cosomega, alpha, beta;

  double b0{1}, b1{0}, b2{0}, a0{0}, a1{0}, a2{0};

  int mode{0};

  void computeInter() {
    if (mode < 4) {
      A = sqrt(pow(10, dBgain / 20));
    } else {
      A = pow(10, dBgain / 40);
    }
    omega = 2 * M_PI * freq / sr;
    sinomega = sin(omega);
    cosomega = cos(omega);
    alpha = sinomega / (2 * Q);
    beta = sqrt(A) / Q;
  }

  void computeCoeffs() {
    switch (mode) {
      case 0:  // LPF
        b0 = (1 - cosomega) / 2;
        b1 = 1 - cosomega;
        b2 = b0;
        a0 = 1 + alpha;
        a1 = -2 * cosomega;
        a2 = 1 - alpha;
        break;
      case 1:  // HPF
        b0 = (1 + cosomega) / 2;
        b1 = -1 - cosomega;
        b2 = b0;
        a0 = 1 + alpha;
        a1 = -2 * cosomega;
        a2 = 1 - alpha;
        break;
      case 2:  // BFP
        b0 = Q * alpha;
        b1 = 0;
        b2 = -Q * alpha;
        a0 = 1 + alpha;
        a1 = -2 * cosomega;
        a2 = 1 - alpha;
        break;
      case 3:  // Notch
        b0 = 1;
        b1 = -2 * cosomega;
        b2 = 1;
        a0 = 1 + alpha;
        a1 = -2 * cosomega;
        a2 = 1 - alpha;
        break;
      case 4:  // peak
        b0 = 1 + alpha * A;
        b1 = -2 * cosomega;
        b2 = A - alpha * A;
        a0 = 1 + alpha / A;
        a1 = -2 * cosomega;
        a2 = 1 - alpha / A;
        break;
      case 5:  // Lowshelf
        b0 = A * ((A + 1) - (A - 1) * cosomega + beta * sinomega);
        b1 = 2 * A * ((A - 1) - (A + 1) * cosomega);
        b2 = A * ((A + 1) - (A - 1) * cosomega - beta * sinomega);
        a0 = (A + 1) + (A - 1) * cosomega + beta * sinomega;
        a1 = -2 * ((A - 1) + (A + 1) * cosomega);
        a2 = (A + 1) + (A - 1) * cosomega - beta * sinomega;
        break;
      case 6:  // Highshelf
        b0 = A * ((A + 1) - (A - 1) * cosomega + beta * sinomega);
        b1 = -2 * A * ((A - 1) + (A + 1) * cosomega);
        b2 = A * ((A + 1) - (A - 1) * cosomega - beta * sinomega);
        a0 = (A + 1) - (A - 1) * cosomega + beta * sinomega;
        a1 = 2 * ((A - 1) - (A + 1) * cosomega);
        a2 = (A + 1) - (A - 1) * cosomega - beta * sinomega;
        break;
    }
  }

 public:
  Biquad(float samplerate = 44100, int mode = 0, float freq = 100,
         float dBgain = 0, float Q = 0.7) {
    sr = samplerate;
    setMode(mode);
    setFreq(freq);
    setGain(dBgain);
    setQ(Q);
    computeInter();
    computeCoeffs();
  };

  void setFreq(float freq) {
    this->freq = std::clamp(freq, float(0), sr);
    computeInter();
    computeCoeffs();
  }

  void setGain(float dBgain) {
    this->dBgain = std::clamp(dBgain, float(-60.0), float(20.0));
    computeInter();
    computeCoeffs();
  }

  void setQ(float Q) {
    this->Q = std::clamp(Q, float(0.001), float(100.0));
    computeInter();
    computeCoeffs();
  }

  void setMode(int mode) {
    this->mode = mode;
    computeInter();
    computeCoeffs();
  }

  void initializeState(double initValue) {
    // Initialize filter states to steady-state value
    // For a step input, steady-state output equals DC gain * input
    double dcGain = (b0 + b1 + b2) / (a0 + a1 + a2);
    double steadyStateOutput = dcGain * initValue;

    x0 = x1 = x2 = initValue;
    y0 = y1 = y2 = steadyStateOutput;
  }

  double process(double input) {
    x0 = input;
    y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1
         - (a2 / a0) * y2;

    y2 = y1;
    y1 = y0;

    x2 = x1;
    x1 = x0;
    return y0;
  }
};

#endif