#include "time_Derivative_Activation_stance.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h" 
#include "Utils/Vector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int emg_Excitation_sparsity[3] = {m.nbMuscleTotal(), 1, 1};
static casadi_int emg_Activation_sparsity[3] = {m.nbMuscleTotal(), 1, 1};

static casadi_int ActivationDot_sparsity[3] = {m.nbMuscleTotal(), 1, 1};


const char* libtime_Derivative_Activation_stance_name(void){
    return "libtime_Derivative_Activation_stance";
}

int libtime_Derivative_Activation_stance(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::utils::Vector  emg_Excitation(m.nbMuscleTotal()), 
                            emg_Activation(m.nbMuscleTotal()), 
                            ActivationDot(m.nbMuscleTotal());
 
     

    // Dispatch data
    for (unsigned int i = 0; i < m.nbMuscleTotal(); ++i){
         emg_Excitation[i] = arg[0][i];
         emg_Activation[i] = arg[1][i];
    }

    // Set the activation for each muscle
    int n_muscle = 0; 
    for (unsigned int nGrp = 0; nGrp < m.nbMuscleGroups(); ++nGrp){
         for (unsigned int nMus = 0; nMus < m.muscleGroup(nGrp).nbMuscles(); ++nMus){
         biorbd::muscles::StateDynamics state;
         double excitation = emg_Excitation[n_muscle];
         double activation = emg_Activation[n_muscle];
         double activationdot = state.timeDerivativeActivation(excitation, 
                                                        activation, 
                                                        m.muscleGroup(nGrp).muscle(nMus).characteristics(), 
                                                        true); 
         ActivationDot[n_muscle] = activationdot;
         ++ n_muscle; 
         }
     }

    // Return the answers
    for (unsigned int i = 0; i < m.nbMuscleTotal(); ++i){
    res[0][i] = ActivationDot[i];
    }
    return 0;
}

// IN
casadi_int libtime_Derivative_Activation_stance_n_in(void){
    return 2;
}
const char* libtime_Derivative_Activation_stance_name_in(casadi_int i){
    switch (i) {
    case 0: return "emg_Excitation";
    case 1: return "emg_Activation";
    default: return nullptr;
    }
}
const casadi_int* libtime_Derivative_Activation_stance_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return emg_Excitation_sparsity;
    case 1: return emg_Activation_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libtime_Derivative_Activation_stance_n_out(void){
    return 1;
}
const char* libtime_Derivative_Activation_stance_name_out(casadi_int i){
    switch (i) {
    case 0: return "ActivationDot";
    default: return nullptr;
    }
}
const casadi_int* libtime_Derivative_Activation_stance_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return ActivationDot_sparsity;
    default: return nullptr;
    }
}

int libtime_Derivative_Activation_stance_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbMuscleTotal() + m.nbMuscleTotal();
    if (sz_res) *sz_res = m.nbMuscleTotal();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libtime_Derivative_Activation_stance_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
