#include <math.h>
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "Muscles/Characteristics.h"
#include "Muscles/Muscle.h"
#include "Muscles/MuscleGroup.h"
#include "Muscles/StateDynamics.h"

static biorbd::Model m("/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod");

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

bool  libtime_Derivative_Activation_stance_has_derivative(void);
const char* libtime_Derivative_Activation_stance_name(void);

int libtime_Derivative_Activation_stance(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libtime_Derivative_Activation_stance_n_in(void);
const char* libtime_Derivative_Activation_stance_name_in(casadi_int i);
const casadi_int* libtime_Derivative_Activation_stance_sparsity_in(casadi_int i);

// OUT
casadi_int libtime_Derivative_Activation_stance_n_out(void);
const char* libtime_Derivative_Activation_stance_name_out(casadi_int i);
const casadi_int* libtime_Derivative_Activation_stance_sparsity_out(casadi_int i);

int libtime_Derivative_Activation_stance_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);
// BIORBD_API int jaco_libtime_Derivative_Activation_stance(const casadi_real** arg,
//                                                   double** res);

#ifdef __cplusplus
} /* extern "C" */
#endif

