#include "marche_Florent_markers_swing.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "RigidBody/NodeSegment.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int bool1_sparsity[3] = {1, 1, 1};
static casadi_int bool2_sparsity[3] = {1, 1, 1};
static casadi_int markers_sparsity[3] = {m.nbMarkers()*3, 1, 1};


const char* libmarche_Florent_markers_swing_name(void){
    return "libmarche_Florent_markers_swing";
}

int libmarche_Florent_markers_swing(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::rigidbody::GeneralizedCoordinates  Q(m);
     bool bool1,bool2;
    std::vector<biorbd::rigidbody::NodeSegment> markers;
    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){

       Q[i] = arg[0][i];
    }
    bool1 = arg[1][0];
    bool2 = arg[2][0];
    // Perform the forward dynamics
    markers = m.markers(Q,bool1,bool2);

    // Return the answers
   for (unsigned int i = 0; i < m.nbMarkers(); ++i){
       for (unsigned int j = 0; j < 3; ++j){
	res[0][3*i+j]   = markers[i][j];
        }
   }   
    return 0;
}

// IN
casadi_int libmarche_Florent_markers_swing_n_in(void){
    return 3;
}
const char* libmarche_Florent_markers_swing_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    case 1: return "bool1";
    case 2: return "bool2";
    default: return nullptr;
    }
}
const casadi_int* libmarche_Florent_markers_swing_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return bool1_sparsity;
    case 2: return bool2_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libmarche_Florent_markers_swing_n_out(void){
    return 1;
}
const char* libmarche_Florent_markers_swing_name_out(casadi_int i){
    switch (i) {
    case 0: return "markers";
    default: return nullptr;
    }
}
const casadi_int* libmarche_Florent_markers_swing_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return markers_sparsity;
    default: return nullptr;
    }
}

int libmarche_Florent_markers_swing_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + 2;
    if (sz_res) *sz_res = m.nbMarkers() * 3;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libmarche_Florent_markers_swing_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
