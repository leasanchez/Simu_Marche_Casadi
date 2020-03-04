#include "muscular_joint_torque.h"
#include "rbdl/Dynamics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include "RigidBody/GeneralizedVelocity.h"
#include "Muscles/Muscles.h"
#include "Muscles/State.h" 
#include "Muscles/StateDynamics.h"
#include "Utils/Vector.h"
#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int activations_sparsity[3] = {m.nbMuscleTotal(), 1, 1};
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Qdot_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int torque_sparsity[3] = {m.nbGeneralizedTorque(), 1, 1};


const char* libmuscular_joint_torque_name(void){
    return "libmuscular_joint_torque";
}

int libmuscular_joint_torque(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
     biorbd::utils::Vector                      activations(m.nbMuscleTotal());
     biorbd::rigidbody::GeneralizedCoordinates  Q(m); 
     biorbd::rigidbody::GeneralizedVelocity     Qdot(m);
     biorbd::rigidbody::GeneralizedTorque       torque; 

    // Dispatch data
    for (unsigned int i = 0; i < m.nbMuscleTotal(); ++i){
        activations[i] = arg[0][i];
    }
    for (unsigned int i = 0; i < m.nbQ(); ++i){
       Q[i] = arg[1][i];
       Qdot[i] = arg[2][i];
    }
  
    // Set muscle activation  
    std::vector<std::shared_ptr<biorbd::muscles::StateDynamics>> states(m.nbMuscleTotal());
    int nMus = 0;
    for (auto& state : states){
        state = std::make_shared<biorbd::muscles::StateDynamics>();
        state->setActivation(activations[nMus]); 
        nMus ++; 
    }

    // Perform the muscular joint torque 
    torque = m.muscularJointTorque(states, true, &Q, &Qdot);

    // Return the answers
    for (unsigned int i = 0; i< m.nbQ(); ++i){
        res[0][i]   = torque[i];
    }   
    return 0;
}

// IN
casadi_int libmuscular_joint_torque_n_in(void){
    return 3;
}
const char* libmuscular_joint_torque_name_in(casadi_int i){
    switch (i) {
    case 0: return "activations";
    case 1: return "Q";
    case 2: return "Qdot";
    default: return nullptr;
    }
}
const casadi_int* libmuscular_joint_torque_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return activations_sparsity;
    case 1: return Q_sparsity;
    case 2: return Qdot_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libmuscular_joint_torque_n_out(void){
    return 1;
}
const char* libmuscular_joint_torque_name_out(casadi_int i){
    switch (i) {
    case 0: return "torque";
    default: return nullptr;
    }
}
const casadi_int* libmuscular_joint_torque_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return torque_sparsity;
    default: return nullptr;
    }
}

int libmuscular_joint_torque_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbMuscleTotal() + m.nbQ() + m.nbQdot();
    if (sz_res) *sz_res = m.nbGeneralizedTorque();
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libmuscular_joint_torque_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
