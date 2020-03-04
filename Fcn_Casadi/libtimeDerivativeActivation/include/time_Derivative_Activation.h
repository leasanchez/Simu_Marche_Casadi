#include <math.h>
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"
#include "Muscles/Characteristics.h"
#include "Muscles/Muscle.h"
#include "Muscles/MuscleGroup.h"
#include "Muscles/StateDynamics.h"

static biorbd::Model m("/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod");

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

bool  libtime_Derivative_Activation_has_derivative(void);
const char* libtime_Derivative_Activation_name(void);

int libtime_Derivative_Activation(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libtime_Derivative_Activation_n_in(void);
const char* libtime_Derivative_Activation_name_in(casadi_int i);
const casadi_int* libtime_Derivative_Activation_sparsity_in(casadi_int i);

// OUT
casadi_int libtime_Derivative_Activation_n_out(void);
const char* libtime_Derivative_Activation_name_out(casadi_int i);
const casadi_int* libtime_Derivative_Activation_sparsity_out(casadi_int i);

int libtime_Derivative_Activation_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);
// BIORBD_API int jaco_libtime_Derivative_Activation(const casadi_real** arg,
//                                                   double** res);

#ifdef __cplusplus
} /* extern "C" */
#endif

