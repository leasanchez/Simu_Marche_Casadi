#include "CoM.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "RigidBody/NodeSegment.h"
#include "Utils/Vector.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int CoM_sparsity[3] = {3, 1, 1};


const char* libCoM_name(void){
    return "libCoM";
}

int libCoM(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::rigidbody::GeneralizedCoordinates  Q(m);
     biorbd::utils::Vector3d CoM;
    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){

       Q[i] = arg[0][i];
    }

    // calculate CoM coordinates
    CoM = m.CoM(Q,true);

    // Return the answers
   for (unsigned int i = 0; i < 3; ++i){
	res[0][i]   = CoM[i];
        }   
    return 0;
}

// IN
casadi_int libCoM_n_in(void){
    return 1;
}
const char* libCoM_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    default: return nullptr;
    }
}
const casadi_int* libCoM_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return Q_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libCoM_n_out(void){
    return 1;
}
const char* libCoM_name_out(casadi_int i){
    switch (i) {
    case 0: return "CoM";
    default: return nullptr;
    }
}
const casadi_int* libCoM_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return CoM_sparsity;
    default: return nullptr;
    }
}

int libCoM_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ();
    if (sz_res) *sz_res = 3;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libCoM_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
