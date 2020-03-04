#include "GJCS_casadi.h"
#include "rbdl/Dynamics.h"
#include "rbdl/Kinematics.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare sparsity dimensions
static bool isSparsityFilled(false);
static casadi_int Q_sparsity[3] = {m.nbQ(), 1, 1};
static casadi_int Id_sparsity[3] = {1, 1, 1};
static casadi_int BodyPoint_sparsity[3] = {3, 1, 1};
static casadi_int res_sparsity[3] = {12, 1, 1};

const char* libGJCS_casadi_name(void){
    return "libGJCS_casadi";
}

int libGJCS_casadi(const casadi_real** arg, double** res, casadi_int*, casadi_real*, void*){
    biorbd::rigidbody::GeneralizedCoordinates Q(m);
	int body_id;
	RigidBodyDynamics::Math::Vector3d body_point_position;

    // Dispatch data
    for (unsigned int i = 0; i < m.nbQ(); ++i){
       Q[i] = arg[0][i];
    }
    body_id = arg[1][0];
    
    for (unsigned int i = 0; i < 3; ++i){
       body_point_position[i] = arg[2][i];
    }
	//~ std::cout << "Q \n" << Q << std::endl;
	//~ std::cout << "body_id \n" << body_id << std::endl;
	//~ std::cout << "body_point_position \n" << body_point_position << std::endl;

	
    // Compute the coordinates of the body_id th body in the global JCS
    RigidBodyDynamics::Math::Vector3d body_t;
    RigidBodyDynamics::Math::Matrix3d body_r;
    body_r = RigidBodyDynamics::CalcBodyWorldOrientation(m,Q, body_id,true);
    body_t = RigidBodyDynamics::CalcBodyToBaseCoordinates(m,Q,body_id,body_point_position,true); 		
	//~ std::cout << "body_r \n" << body_r << std::endl;
	//~ std::cout << "body_t \n" << body_t << std::endl;
    // Return the answers
    // bodyPoint translation
    int cnt = 0;
    for (unsigned int i = 0; i< 3; ++i){
        res[0][i]   = body_t[i];
    }   
    for (unsigned int i = 0; i< 3; ++i){
		for (unsigned int j = 0; j < 3; ++j){
            res[0][cnt+3]   = body_r(i,j);
            cnt += 1;
         }   
    }
    return 0;
}

// IN
casadi_int libGJCS_casadi_n_in(void){
    return 3;
}
const char* libGJCS_casadi_name_in(casadi_int i){
    switch (i) {
    case 0: return "Q";
    case 1: return "Id";
    case 2: return "BodyPoint";
    default: return nullptr;
    }
}
const casadi_int* libGJCS_casadi_sparsity_in(casadi_int i) {
    switch (i) {
    case 0: return Q_sparsity;
    case 1: return Id_sparsity;
    case 2: return BodyPoint_sparsity;
    default: return nullptr;
    }
}

// OUT
casadi_int libGJCS_casadi_n_out(void){
    return 1;
}
const char* libGJCS_casadi_name_out(casadi_int i){
    switch (i) {
    case 0: return "bodyST";
    default: return nullptr;
    }
}
const casadi_int* libGJCS_casadi_sparsity_out(casadi_int i) {
    switch (i) {
    case 0: return res_sparsity;
    default: return nullptr;
    }
}

int libGJCS_casadi_work(casadi_int *sz_arg,
                                               casadi_int* sz_res,
                                               casadi_int *sz_iw,
                                               casadi_int *sz_w) {
    if (sz_arg) *sz_arg = m.nbQ() + 1;
    if (sz_res) *sz_res = 12;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

bool libGJCS_casadi_has_derivative(void)
{
    return true;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
