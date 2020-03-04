#include "compute_constraint_5link_biped.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "RigidBody/Contacts.h"
#include "Utils/Vector.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int QdotMinus_sparsity[3] = {m.nbQdot(), 1, 1};
static casadi_int QdotPlus_sparsity[3] = {2*m.nbQdot() + m.nbContacts(), 1, 1};


const char* libcompute_constraint_5link_biped_name(void){
    return "libcompute_constraint_5link_biped";
}

int libcompute_constraint_5link_biped(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::rigidbody::GeneralizedCoordinates  Q(m), QdotMinus(m), QdotPlus(m);
     biorbd::rigidbody::Contacts  C(m.getConstraints()); 
     biorbd::utils::Vector        f;
    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){

       Q[i] = arg[0][i];
       QdotMinus[i] = arg[1][i];
    }

    // Perform the forward dynamics
    RigidBodyDynamics::ComputeConstraintImpulsesDirect(m, Q, QdotMinus, C, QdotPlus);
    f = C.getForce();
    // Return the answers
    for (unsigned int i = 0; i< m.nbQ(); ++i){
        res[0][i]   = Q[i];
        res[0][i+m.nbQ()] = QdotPlus[i];
    }   
    for (unsigned int i = 0; i< m.nbContacts(); ++i){
        res[0][2*m.nbQ()+i]   = f[i];
    }   
    return 0;
}

// IN
casadi_int libcompute_constraint_5link_biped_n_in(void){
    return 2;
}
const char* libcompute_constraint_5link_biped_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    case 1: return "QdotMinus";
    default: return nullptr;
    }
}
const casadi_int* libcompute_constraint_5link_biped_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return QdotMinus_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libcompute_constraint_5link_biped_n_out(void){
    return 1;
}
const char* libcompute_constraint_5link_biped_name_out(casadi_int i){
    switch (i) {
    case 0: return "QdotPlus";
    default: return nullptr;
    }
}
const casadi_int* libcompute_constraint_5link_biped_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return QdotPlus_sparsity;
    default: return nullptr;
    }
}

int libcompute_constraint_5link_biped_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + m.nbQdot() + m.nbGeneralizedTorque() ;
    if (sz_res) *sz_res = m.nbQ() + m.nbQdot() + m.nbContacts();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libcompute_constraint_5link_biped_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
