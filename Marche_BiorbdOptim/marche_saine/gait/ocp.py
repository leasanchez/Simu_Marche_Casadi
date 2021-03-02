import numpy as np
from casadi import vertcat, MX
from .dynamics_function import dynamics
from .objective_function import objective
from.constraint_function import constraint
from.parameters_functions import parameter

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ObjectiveList,
    InterpolationType,
    ConstraintList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    Solver,
    ParameterList,
)


class gait_muscle_driven:
    def __init__(self, models, nb_shooting, phase_time, q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref, save_path=None, n_threads=1):
        self.models=models

        # Element for the optimization
        self.n_phases = len(models)
        self.nb_shooting = nb_shooting
        self.phase_time = phase_time
        self.n_threads=n_threads

        # Element for the tracking
        self.q_ref=q_ref
        self.qdot_ref=qdot_ref
        self.markers_ref = markers_ref
        self.grf_ref=grf_ref
        self.moments_ref=moments_ref
        self.cop_ref=cop_ref

        # Element from the model
        self.nb_q=models[0].nbQ()
        self.nb_qdot=models[0].nbQdot()
        self.nb_tau=models[0].nbGeneralizedTorque()
        self.nb_mus=models[0].nbMuscleTotal()
        self.torque_min, self.torque_max, self.torque_init = -1000, 1000, 0
        self.activation_min, self.activation_max, self.activation_init = 1e-3, 1.0, 0.1

        # objective functions
        self.objective_functions = ObjectiveList()
        self.set_objective_function()

        # dynamics
        self.dynamics = DynamicsList()
        self.set_dynamics()

        # constraints
        self.constraints = ConstraintList()
        self.set_constraint()

        # parameters
        self.parameters = ParameterList()
        self.set_parameters()

        # Phase transitions
        self.phase_transition = PhaseTransitionList()
        self.set_phase_transition()

        # Path constraint
        self.x_bounds = BoundsList()
        self.u_bounds = BoundsList()
        self.set_bounds()

        # Initial guess
        self.x_init = InitialGuessList()
        self.u_init = InitialGuessList()
        if save_path is not None:
            self.save_path = save_path
            self.set_initial_guess_from_solution()
        else:
            self.set_initial_guess()

        # Ocp
        self.ocp = OptimalControlProgram(
            self.models,
            self.dynamics,
            self.nb_shooting,
            self.phase_time,
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            u_init=self.u_init,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            phase_transitions=self.phase_transition,
            n_threads=self.n_threads,
            # parameters=self.parameters,
        )


    def set_objective_function(self):
        objective.set_objective_function_heel_strike(self.objective_functions, self.markers_ref[0], self.grf_ref[0], self.moments_ref[0], self.cop_ref[0])
        objective.set_objective_function_flatfoot(self.objective_functions, self.markers_ref[1], self.grf_ref[1], self.moments_ref[1], self.cop_ref[1])
        objective.set_objective_function_forefoot(self.objective_functions, self.markers_ref[2], self.grf_ref[2], self.moments_ref[2], self.cop_ref[2])
        objective.set_objective_function_swing(self.objective_functions, self.markers_ref[3], self.grf_ref[3], self.moments_ref[3], self.cop_ref[3])


    def set_dynamics(self):
        dynamics.set_muscle_driven_dynamics(self.dynamics)

    def set_constraint(self):
        constraint.set_constraint_heel_strike(self.constraints)
        constraint.set_constraint_flatfoot(self.constraints)
        constraint.set_constraint_forefoot(self.constraints)

    def set_parameters(self):
        parameter.set_parameters_f_iso(self.parameters, self.models[0])


    def set_phase_transition(self):
        self.phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
        self.phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    def set_bounds(self):
        for p in range(self.n_phases):
            self.x_bounds.add(bounds=QAndQDotBounds(self.models[p]))
            self.u_bounds.add([self.torque_min] * self.nb_tau + [self.activation_min] * self.nb_mus,
                              [self.torque_max] * self.nb_tau + [self.activation_max] * self.nb_mus)

    def set_initial_guess(self):
        n_shoot=0
        for p in range(self.n_phases):
            init_x = np.zeros((self.nb_q + self.nb_qdot, self.nb_shooting[p] + 1))
            init_x[:self.nb_q, :] = self.q_ref[p]
            init_x[self.nb_q:self.nb_q + self.nb_qdot, :] = self.qdot_ref[p]
            self.x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

            init_u = [self.torque_init] * self.nb_tau + [self.activation_init] * self.nb_mus
            self.u_init.add(init_u)
            n_shoot += self.nb_shooting[p]

    def set_initial_guess_from_solution(self):
        n_shoot = 0
        for p in range(self.n_phases):
            init_x = np.zeros((self.nb_q + self.nb_qdot, self.nb_shooting[p] + 1))
            init_x[:self.nb_q, :] = np.load(self.save_path + "q.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            init_x[self.nb_q:self.nb_q + self.nb_qdot, :] = np.load(self.save_path + "qdot.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            self.x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

            init_u = np.zeros((self.nb_tau + self.nb_mus, self.nb_shooting[p]))
            init_u[:self.nb_tau, :] = np.load(self.save_path + "tau.npy")[:, n_shoot:n_shoot + self.nb_shooting[p]]
            init_u[self.nb_tau:, :] = np.load(self.save_path + "muscle.npy")[:, n_shoot:n_shoot + self.nb_shooting[p]]
            self.u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
            n_shoot += self.nb_shooting[p]

    def solve(self):
        sol = self.ocp.solve(
            solver=Solver.IPOPT,
            solver_options={
                "ipopt.tol": 1e-3,
                "ipopt.max_iter": 1000,
                "ipopt.hessian_approximation": "exact",
                "ipopt.limited_memory_max_history": 50,
                "ipopt.linear_solver": "ma57",
            },
            show_online_optim=False,
        )
        return sol