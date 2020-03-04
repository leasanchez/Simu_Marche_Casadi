#include <math.h>
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"

static biorbd::Model m("/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod");

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

bool  libmuscular_joint_torque_has_derivative(void);
const char* libmuscular_joint_torque_name(void);

int libmuscular_joint_torque(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libmuscular_joint_torque_n_in(void);
const char* libmuscular_joint_torque_name_in(casadi_int i);
const casadi_int* libmuscular_joint_torque_sparsity_in(casadi_int i);

// OUT
casadi_int libmuscular_joint_torque_n_out(void);
const char* libmuscular_joint_torque_name_out(casadi_int i);
const casadi_int* libmuscular_joint_torque_sparsity_out(casadi_int i);

int libmuscular_joint_torque_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);
// BIORBD_API int jaco_libmuscular_joint_torque(const casadi_real** arg,
//                                                   double** res);

#ifdef __cplusplus
} /* extern "C" */
#endif

