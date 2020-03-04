% =========================================================================
% ANsWER | F. Moissenet | 2018
% =========================================================================
% Name : ComputeActivationDynamics.m
% Aim  : Muscle activation dynamics
% =========================================================================

function [T,A] = ComputeActivationDynamics(model,a0,node,control)

Ti = [];
Ai = [];
for i = 1:size(a0,2)-1    
    u = control.u(:,i);
    % Rectify the control (between 0 and 1)
    u(u < 0) = 0;
    u(u > 1) = 1;    
    % Differential equation
    fun = @(ti,ai)FwDyn_MuscleAct(ti,ai,model,u);
    [ti,ai] = ode45(fun,[node.tps(i) node.tps(i+1)],a0(:,i));
    % Export T and X
    Ti = [Ti; ti];
    Ai = [Ai; ai];    
end
[Ti,idx] = unique(Ti);
A = interp1(Ti,Ai(idx,:),node.tps);
T = node.tps;

% -------------------------------------------------------------------------
% Subfunctions
% -------------------------------------------------------------------------
function adot = FwDyn_MuscleAct(ti,ai,model,u)

adot = S2M_rbdl('muscleActivationDot',model,u,ai,true);