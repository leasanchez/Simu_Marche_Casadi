#include <math.h>
#include <casadi/casadi.hpp>
#include "BiorbdModel.h"

static biorbd::Model m("/home/leasanchez/programmation/Five_Link/DoubleSupport/Five_Link_Biped_BIORBD_doubleSupport.bioMod");

#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

bool  libcompute_constraint_5link_biped_has_derivative(void);
const char* libcompute_constraint_5link_biped_name(void);

int libcompute_constraint_5link_biped(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libcompute_constraint_5link_biped_n_in(void);
const char* libcompute_constraint_5link_biped_name_in(casadi_int i);
const casadi_int* libcompute_constraint_5link_biped_sparsity_in(casadi_int i);

// OUT
casadi_int libcompute_constraint_5link_biped_n_out(void);
const char* libcompute_constraint_5link_biped_name_out(casadi_int i);
const casadi_int* libcompute_constraint_5link_biped_sparsity_out(casadi_int i);

int libcompute_constraint_5link_biped_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);
// BIORBD_API int jaco_libcompute_constraint_5link_biped(const casadi_real** arg,
//                                                   double** res);

#ifdef __cplusplus
} /* extern "C" */
#endif

