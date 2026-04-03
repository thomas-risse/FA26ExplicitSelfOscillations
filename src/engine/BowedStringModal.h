#ifndef BOWED_STRING_MODAL_h
#define BOWED_STRING_MODAL_h

#include <cmath>
#include <vector>
#include <tuple>

#include <Eigen/Dense>
#include <string_view>

#include "EigenUtility.h"
#include "Bow.h"

template <typename ftype>
class BowedStringModal
{
private:
    using Vec = Eigen::ArrayX<ftype>;
    using VecRef = Eigen::Ref<const Vec>;
    // Numerical epsilon value
    constexpr static ftype NUM_EPS{1e-14};

    // Number of modes
    int Nmodes{1};

    // Linear part: system parameters (diagonal)
    Vec M, K, R0;
    // Higer level modal parameters
    Vec Amps, Omega, Decays;

    // Nonlinear parameters
    ftype vrel, phival, Fb_v;

    // Bow model
    std::shared_ptr<Bow<ftype>> bow;

    // Bow position
    float exPos{0.5};
    Vec exProj;
    void computeInputVector();

    // Time-scheme parameters
    float sr;
    ftype dt;
    bool controlTerm{true};
    ftype lambda0{0};
    int dissipationMode{0};

    // System state
    Vec qlast, qnow, qnext;
    ftype vBowLast{NAN}, FBowLast{NAN};

    // Intermediate vectors
    Vec RHS, LHS;
    Vec A0_inv, A0_inv_phi, M_qnow, M_qlast;
    ftype phi_A0_inv_phi;

    // Functions to go from high levels to low levels modal parameters
    void computePhysicalParameters();
    void computeModalParameters() {};

    // Update fixed intermediate quantities
    void updateIntermediateQuantities();

public:
    BowedStringModal(float sampleRate, std::shared_ptr<Bow<ftype>> bow, int Nmmodes, int dissipationMode = 0);

    void ReinitDsp(float sampleRate);

    std::tuple<Vec, Vec> process(ftype vBow, ftype FBow);

    // Physical parameters
    void setPhysicalParameters(VecRef M, VecRef K, VecRef R);

    // Higher level modal parameters
    void setLinearParameters(VecRef Amps, VecRef Freqs, VecRef Decay);
    void setAmps(VecRef Amp);
    void setFreqs(VecRef Freq);
    void setDecays(VecRef Decay);

    void setLambda0(ftype lambda0) { this->lambda0 = std::clamp(lambda0, ftype(0), ftype(10000)); };

    void setExPos(ftype pos);
};

#endif
