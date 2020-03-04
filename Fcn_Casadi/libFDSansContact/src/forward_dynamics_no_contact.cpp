#include "forward_dynamics_no_contact.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Qdot_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Qddot_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Tau_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Xp_sparsity[3] = {2*m.nbQ(), 1, 1};


const char* libforward_dynamics_no_contact_name(void){
    return "libforward_dynamics_no_contact";
}

int libforward_dynamics_no_contact(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::rigidbody::GeneralizedCoordinates  Q(m), Qdot(m), Qddot(m);
     biorbd::rigidbody::GeneralizedCoordinates  Tau(m);

    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){

       Q[i] = arg[0][i];
       Qdot[i] = arg[1][i];
       Tau[i] = arg[2][i];
    }

    // Perform the forward dynamics
    RigidBodyDynamics::ForwardDynamics(m, Q, Qdot, Tau, Qddot);

    // Return the answers
    for (unsigned int i = 0; i< m.nbQ(); ++i){
        res[0][i]   = Qdot[i];
        res[0][i+m.nbQ()] = Qddot[i];
    }   
    return 0;
}

// IN
casadi_int libforward_dynamics_no_contact_n_in(void){
    return 3;
}
const char* libforward_dynamics_no_contact_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    case 1: return "Qdot";
    case 2: return "Tau";
    default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_no_contact_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return Qdot_sparsity;
    case 2: return Tau_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libforward_dynamics_no_contact_n_out(void){
    return 1;
}
const char* libforward_dynamics_no_contact_name_out(casadi_int i){
    switch (i) {
    case 0: return "Xp";
    // case 1: return "Qddot";
    default: return nullptr;
    }
}
const casadi_int* libforward_dynamics_no_contact_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return Xp_sparsity;
    // case 1: return Qddot_sparsity;
    default: return nullptr;
    }
}

int libforward_dynamics_no_contact_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + m.nbQdot() + m.nbGeneralizedTorque();
    if (sz_res) *sz_res = m.nbQ() + m.nbQdot();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libforward_dynamics_no_contact_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
