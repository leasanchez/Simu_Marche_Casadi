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

bool  libGJCS_casadi_has_derivative(void);
const char* libGJCS_casadi_name(void);

int libGJCS_casadi(const casadi_real** arg,
                                                   double** res,
                                                   casadi_int* iw,
                                                   casadi_real* w,
                                                   void* mem);

// IN
casadi_int libGJCS_casadi_n_in(void);
const char* libGJCS_casadi_name_in(casadi_int i);
const casadi_int* libGJCS_casadi_sparsity_in(casadi_int i);

// OUT
casadi_int libGJCS_casadi_n_out(void);
const char* libGJCS_casadi_name_out(casadi_int i);
const casadi_int* libGJCS_casadi_sparsity_out(casadi_int i);

int libGJCS_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w);


#ifdef __cplusplus
} /* extern "C" */
#endif

