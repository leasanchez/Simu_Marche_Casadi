% =========================================================================
% Marche_Florent_Casadi | L. Sanchez | 2020
% =========================================================================
% Name : Marche_Florent_Casadi_Main.m
% Aim  : Main script to generate data files from experimental measurements
% =========================================================================

clearvars;
close all;
clc;

% -------------------------------------------------------------------------
% Which gait phase?
% -------------------------------------------------------------------------
Gaitphase = 'swing'; %'swing' ; 'stance'

% -------------------------------------------------------------------------
% Set paths
% -------------------------------------------------------------------------
paths.User       = '/home/leasanchez/programmation/Marche_Florent/'; 
paths.Btk        = '/home/leasanchez/programmation/miniconda3/envs/marche/share/btk-0.4dev/Wrapping/Matlab'; 
paths.Data       = [paths.User 'DonneesMouvement/'];
paths.FichierMat = [paths.User 'Marche_Florent_Casadi/Fcn_Matlab/Data/'];
paths.Model      = [paths.User 'ModelesS2M/'];     
addpath([paths.User 'Marche_Florent_Casadi/Fcn_Matlab']);      
addpath(genpath(char(paths.Btk)));                   

% -------------------------------------------------------------------------
% Load the model
% -------------------------------------------------------------------------
FileModel     = 'ANsWER_Rleg_6dof_17muscle_0contact.bioMod';  

model         = biorbd('new',[paths.Model,FileModel]);
alias.nDof    = biorbd('nQ', model);                                     % DoFs
alias.nTag    = biorbd('nMarkers', model);                               % Markers
alias.nRoot   = biorbd('nRoot', model);                                  % Roots
alias.nMus    = biorbd('nmuscles', model);                               % Muscles
alias.nBody   = biorbd('nBody', model);                                  % Segments

% -------------------------------------------------------------------------
% Set the number of interval (nodes+1)
% -------------------------------------------------------------------------
alias.nInterp = 26;                                                      % Stance or swing

% -------------------------------------------------------------------------
% Load the data equinus a lancer
% -------------------------------------------------------------------------
% btk    = btkReadAcquisition([paths.Data,'equinus15_out.c3d']);
btk    = btkReadAcquisition([paths.Data,'normal04_out.c3d']);

kalman = load([paths.Data,'equinus15_out_MOD5000_leftHanded_GenderF_Florent_.Q2'],'-mat'); 
[init,time,node,data,nCycle] = LoadData(btk,kalman,alias);

%
% -------------------------------------------------------------------------
% Save Data
% -------------------------------------------------------------------------
param.liss  = 0.9995;

% Temps 
filename = [Gaitphase, '/Time_equinus15_out.c3d.mat'];
save([paths.FichierMat, filename], data.t); 

% OBJECTIVE FUNCTION
%
% EMG : (nbMus - 7) x nbNoeuds
[~,idxT] = unique(data.t);
EMG      = interp1(data.t(idxT)', data.EMG(:,idxT)', node.tps,'spline')';
filename = [Gaitphase, '/EMG_equinus15_out.c3d.mat']; 
save([paths.FichierMat, filename], EMG); 

% Marker : nbMarker x 3 x nbNoeuds
[~,idxT] = unique(data.t);
Marker   = permute(interp1(data.t(idxT)', permute(data.M(:,idxT,:),[2,1,3]),node.tps,'spline'),[2,1,3]);
filename = [Gaitphase, '/Markers_equinus15_out.c3d.mat']; 
save([paths.FichierMat, filename], Marker); 

% GRF : nbContact x 2 x nbNoeuds
[~,idxT] = unique(data.t);
GRF      = interp1(data.t(idxT)',data.R(:,idxT)',node.tps,'spline')';
filename = [Gaitphase, '/GRF_equinus15_out.c3d.mat']; 
save([paths.FichierMat, filename], GRF); 


% INITIAL GUESS
%
% U0 : nMus x nbNoeuds
[~,idxT]    = unique(data.t);
U0 = [ ...
    data.EMG(10,:); ...                                                    % glut_max1_r
    data.EMG(10,:); ...                                                    % glut_max2_r
    data.EMG(10,:); ...                                                    % glut_max3_r
    data.EMG(9,:); ...                                                     % glut_med1_r
    data.EMG(9,:); ...                                                     % glut_med2_r
    data.EMG(9,:); ...                                                     % glut_med3_r
    data.EMG(7,:); ...                                                     % semimem_r
    data.EMG(7,:); ...                                                     % semiten_r
    data.EMG(8,:); ...                                                     % bi_fem_lh_r
    data.EMG(6,:); ...                                                     % rectus_fem_r
    data.EMG(5,:); ...                                                     % vas_med_r
    data.EMG(5,:); ...                                                     % vas_int_r
    data.EMG(5,:); ...                                                     % vas_lat_r
    data.EMG(3,:); ...                                                     % gas_med_r
    data.EMG(4,:); ...                                                     % gas_lat_r
    data.EMG(2,:); ...                                                     % soleus_r
    data.EMG(1,:); ...                                                     % tib_ant_r
    ];
U0 = interp1(data.t(idxT)',U0(:,idxT)', node.tps,'spline')';
filename = [Gaitphase, '/U0_equinus15_out.c3d.mat'];
save([paths.FichierMat, filename], data.Q);

% Q : nDofs x nbNoeuds
[~,idxT] = unique(data.t);
Q        = interp1(data.t(idxT)',data.Q(:,idxT)',node.tps,'spline')';
filename = [Gaitphase, '/Q_equinus15_out.c3d.mat'];
save([paths.FichierMat, filename], data.Q);

% Qdot : nDofs x nbNoeuds  
pp1         = csaps(data.t,data.Q,param.liss);
ppderivee11 = fnder(pp1,1);
data.Qdot   = fnval(ppderivee11,data.t);
filename    = [Gaitphase, '/Qdot_equinus15_out.c3d.mat'];
save([paths.FichierMat, filename], data.Qdot);

% Activation : nMus x nbNoeuds
a0       = U0(:,1).*ones(alias.nMus,alias.nInterp);
[~,A]    = ComputeActivationDynamics(model,a0,node,control);
A(A < 0) = 0;                                                            % Activations adjusted between 0 and 1
A(A > 1) = 1;
filename = [Gaitphase, '/A_equinus15_out.c3d.mat'];
save([paths.FichierMat, filename], A);

