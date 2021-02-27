import numpy as np
from casadi import vertcat, MX
from .dynamics_function import dynamics
from .objective_function import objective

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    PhaseTransitionList,
    PhaseTransitionFcn,
    PenaltyNodes,
    Solver,
)


# --- force nul at last point ---
def get_last_contact_force_null(pn: PenaltyNodes, contact_name: str) -> MX:
    """
    Adds the constraint that the force at the specific contact point should be null
    at the last phase point.
    All contact forces can be set at 0 at the last node by using 'all' at contact_name.

    Parameters
    ----------
    pn: PenaltyNodes
        The penalty node elements
    contact_name: str
        Name of the contacts that sould be null at the last node

    Returns
    -------
    The value that should be constrained in the MX format

    """

    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    if contact_name == "all":
        val = force
    else:
        cn = pn.nlp.model.contactNames()
        val = []
        for i, c in enumerate(cn):
            if isinstance(contact_name, tuple):
                for name in contact_name:
                    if name in c.to_string():
                        val = vertcat(val, force[i])
            else:
                if contact_name in c.to_string():
                    val = vertcat(val, force[i])
    return val



class gait_torque_driven:
    def __init__(self, models, nb_shooting, phase_time, q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref, n_threads=1, four_contact=False):
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
        self.four_contact=four_contact
        self.torque_min, self.torque_max, self.torque_init = -1000, 1000, 0

        # objective functions
        self.objective_functions = ObjectiveList()
        self.set_objective_function()

        # dynamics
        self.dynamics = DynamicsList()
        self.set_dynamics()

        # constraints
        self.constraints = ConstraintList()
        if self.four_contact:
            self.set_constraint_four_contact()
        else:
            self.set_constraint_three_contact()

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
        self.set_initial_guess()
        # self.set_initial_guess_from_solution(save_path)

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
        )



    def set_objective_function(self):
        for p in range(self.n_phases):
            objective.set_objective_function_markers(self.objective_functions, self.markers_ref[p], p)

        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[0], 0)
        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[1], 1)
        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[2], 2)

        objective.set_objective_function_moments(self.objective_functions, self.moments_ref[1], self.cop_ref[1], 1)
        objective.set_objective_function_moments(self.objective_functions, self.moments_ref[2], self.cop_ref[2], 2)


    def set_dynamics(self):
        dynamics.set_torque_driven_dynamics(self.dynamics)

    def set_constraint_four_contact(self):
        self.constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=0,
        )
        # --- phase flatfoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=1,
        )
        self.constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=1,
        )

        self.constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='Heel_r',
            phase=1,
        )

        # --- phase forefoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(2, 4, 5),
            phase=2,
        )
        self.constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 4, 5),
            tangential_component_idx=1,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=2,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='all',
            phase=2,
        )

    def set_constraint_three_contact(self):
        self.constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=0,
        )
        # --- phase flatfoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=1,
        )
        self.constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=1,
        )

        self.constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='Heel_r',
            phase=1,
        )

        # --- phase forefoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 4),
            phase=2,
        )
        self.constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 4),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x toes
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=4,
            tangential_component_idx=2,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='all',
            phase=2,
        )

    def set_phase_transition(self):
        self.phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
        self.phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    def set_bounds(self):
        for p in range(self.n_phases):
            self.x_bounds.add(bounds=QAndQDotBounds(self.models[p]))
            self.u_bounds.add([self.torque_min] * self.nb_tau,[self.torque_max] * self.nb_tau)

    def set_initial_guess(self):
        n_shoot=0
        for p in range(self.n_phases):
            init_x = np.zeros((self.nb_q + self.nb_qdot, self.nb_shooting[p] + 1))
            init_x[:self.nb_q, :] = self.q_ref[p]
            init_x[self.nb_q:self.nb_q + self.nb_qdot, :] = self.qdot_ref[p]
            self.x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

            init_u = np.zeros((self.nb_tau, self.nb_shooting[p]))
            self.u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
            n_shoot += self.nb_shooting[p]

    def set_initial_guess_from_solution(self, save_path):
        n_shoot = 0
        for p in range(self.n_phases):
            init_x = np.zeros((self.nb_q + self.nb_qdot, self.nb_shooting[p] + 1))
            init_x[:self.nb_q, :] = np.load(save_path + "q.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            init_x[self.nb_q:self.nb_q + self.nb_qdot, :] = np.load(save_path + "q_dot.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            self.x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

            init_u = np.load(save_path + "tau.npy")[:, n_shoot:n_shoot + self.nb_shooting[p]]
            self.u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
            n_shoot += self.nb_shooting[p]

    def solve(self):
        sol = self.ocp.solve(
            solver=Solver.IPOPT,
            solver_options={
                "ipopt.tol": 1e-3,
                "ipopt.max_iter": 5000,
                "ipopt.hessian_approximation": "exact",
                "ipopt.limited_memory_max_history": 50,
                "ipopt.linear_solver": "ma57",
            },
            show_online_optim=False,
        )
        return sol


class gait_muscle_driven:
    def __init__(self, models, nb_shooting, phase_time, q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref, n_threads=1):
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
        if self.four_contact:
            self.set_constraint_four_contact()
        else:
            self.set_constraint_three_contact()

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
        self.set_initial_guess()
        # self.set_initial_guess_from_solution(save_path)

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
        )


    def set_objective_function(self):
        for p in range(self.n_phases):
            objective.set_objective_function_markers(self.objective_functions, self.markers_ref[p], p)
            objective.set_objective_function_controls(self.objective_functions, p)

        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[0], 0)
        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[1], 1)
        objective.set_objective_function_forces(self.objective_functions, self.grf_ref[2], 2)

        objective.set_objective_function_moments(self.objective_functions, self.moments_ref[1], self.cop_ref[1], 1)
        objective.set_objective_function_moments(self.objective_functions, self.moments_ref[2], self.cop_ref[2], 2)


    def set_dynamics(self):
        dynamics.set_muscle_driven_dynamics(self.dynamics)

    def set_constraint_four_contact(self):
        self.constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=0,
        )
        # --- phase flatfoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=1,
        )
        self.constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=1,
        )

        self.constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='Heel_r',
            phase=1,
        )

        # --- phase forefoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(2, 4, 5),
            phase=2,
        )
        self.constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 4, 5),
            tangential_component_idx=1,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=2,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='all',
            phase=2,
        )

    def set_constraint_three_contact(self):
        self.constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=0,
        )
        # --- phase flatfoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=1,
        )
        self.constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=1,
        )
        self.constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=1,
        )

        self.constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='Heel_r',
            phase=1,
        )

        # --- phase forefoot ---
        self.constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 4),
            phase=2,
        )
        self.constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 4),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(  # non slipping x toes
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=4,
            tangential_component_idx=2,
            static_friction_coefficient=0.2,
            phase=2,
        )
        self.constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='all',
            phase=2,
        )

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

            init_u = np.zeros((self.nb_tau + self.nb_mus, self.nb_shooting[p]))
            self.u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
            n_shoot += self.nb_shooting[p]

    def set_initial_guess_from_solution(self, save_path):
        n_shoot = 0
        for p in range(self.n_phases):
            init_x = np.zeros((self.nb_q + self.nb_qdot, self.nb_shooting[p] + 1))
            init_x[:self.nb_q, :] = np.load(save_path + "q.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            init_x[self.nb_q:self.nb_q + self.nb_qdot, :] = np.load(save_path + "q_dot.npy")[:, n_shoot:n_shoot + self.nb_shooting[p] + 1]
            self.x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

            init_u = np.zeros((self.nb_tau + self.nb_mus, self.nb_shooting[p]))
            init_u[:self.nb_tau, :] = np.load(save_path + "tau.npy")[:, n_shoot:n_shoot + self.nb_shooting[p]]
            init_u[self.nb_tau:, :] = np.load(save_path + "activation.npy")[:, n_shoot:n_shoot + self.nb_shooting[p]]
            self.u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
            n_shoot += self.nb_shooting[p]

    def solve(self):
        sol = self.ocp.solve(
            solver=Solver.IPOPT,
            solver_options={
                "ipopt.tol": 1e-3,
                "ipopt.max_iter": 5000,
                "ipopt.hessian_approximation": "exact",
                "ipopt.limited_memory_max_history": 50,
                "ipopt.linear_solver": "ma57",
            },
            show_online_optim=False,
        )
        return sol