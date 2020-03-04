% =========================================================================
% ANsWER | F. Moissenet | 2018
% =========================================================================
% Name : LoadData.m
% Aim  : Load data from .c3d file and store in the structure data
% =========================================================================

function [init,time,node,data,nCycle] = LoadData(btk,kalman,alias)

% Set the temporal timing of the data                         // STANCE
% events = btkGetEvents(btk);
% if events.General_RHS(1) > events.General_RTO(1)
%     start = round(events.General_RHS(1)*btkGetPointFrequency(btk))+1;
%     stop = round(events.General_RTO(2)*btkGetPointFrequency(btk))+1;
% elseif events.General_RHS(1) < events.General_RTO(1)
%     start = round(events.General_RHS(1)*btkGetPointFrequency(btk))+1;
%     stop = round(events.General_RTO(1)*btkGetPointFrequency(btk))+1;
% end
% nCycle  = stop-start+1;
% Set the temporal timing of the data                        // SWING
events = btkGetEvents(btk);
if events.General_RHS(1) < events.General_RTO(1)
    start = round(events.General_RTO(1)*btkGetPointFrequency(btk))+1;
    stop = round(events.General_RHS(2)*btkGetPointFrequency(btk))+1;
elseif events.General_RHS(1) > events.General_RTO(1)
    start = round(events.General_RTO(2)*btkGetPointFrequency(btk))+1;
    stop = round(events.General_RHS(2)*btkGetPointFrequency(btk))+1;
end
nCycle  = stop-start+1;
% % Set the temporal timing of the data // GAIT CYCLE
% events  = btkGetEvents(btk);
% start   = round(events.General_RHS(1)*btkGetPointFrequency(btk))+1;
% stop    = round(events.General_RHS(2)*btkGetPointFrequency(btk))+1;
% nCycle  = stop-start+1;
	
% Set init
init.freq_kin = btkGetPointFrequency(btk);
init.freq_emg = btkGetAnalogFrequency(btk);
init.t_act    = 0.01;                                                      % constant time for muscle activation (s)
init.t_dea    = 0.04;                                                      % constant time for muscle desactivation (s)

% Set time
time.dt       = 1/init.freq_kin;
time.instants = 1:stop-start+1;
time.duration = time.dt*time.instants(end);
data.t = (linspace(0,(time.instants(end)-1)*time.dt,...
    time.instants(end)));

% Set node
node.deb = 1;
node.fin = alias.nInterp;                                                  % Time intervals (nodes)
node.pas = 1;
node.tps = linspace(data.t(1),data.t(end),alias.nInterp)';

% Set data (markers)
temp = btkGetMarkers(btk);
ntemp = fieldnames(temp);
for j = 1:length(ntemp)
    markers.(ntemp{j}) = temp.(ntemp{j})(start:stop,:)*1e-3;               % Keep only right cycle data
    markers.(ntemp{j})(:,3) = markers.(ntemp{j})(:,3);              
end
data.M = [permute(markers.L_IAS,[3 1 2]); ...                              % Pelvis markers
          permute(markers.L_IPS,[3 1 2]); ...
          permute(markers.R_IPS,[3 1 2]); ...
          permute(markers.R_IAS,[3 1 2]); ...
          permute(markers.R_FTC,[3 1 2]); ...                              % Femur markers
          permute(markers.R_Thigh_Top,[3 1 2]); ...
          permute(markers.R_Thigh_Down,[3 1 2]); ...
          permute(markers.R_Thigh_Front,[3 1 2]); ...
          permute(markers.R_Thigh_Back,[3 1 2]); ...
          permute(markers.R_FLE,[3 1 2]); ...
          permute(markers.R_FME,[3 1 2]); ...
          permute(markers.R_FAX,[3 1 2]); ...                              % Tibia markers
          permute(markers.R_TTC,[3 1 2]); ...
          permute(markers.R_Shank_Top,[3 1 2]); ...
          permute(markers.R_Shank_Down,[3 1 2]); ...
          permute(markers.R_Shank_Front,[3 1 2]); ...
          permute(markers.R_Shank_Tibia,[3 1 2]); ...
          permute(markers.R_FAL,[3 1 2]); ...
          permute(markers.R_TAM,[3 1 2]); ... 
          permute(markers.R_FCC,[3 1 2]); ...                              % Foot_marker markers
          permute(markers.R_FM1,[3 1 2]); ... 
          permute(markers.R_FMP1,[3 1 2]); ... 
          permute(markers.R_FM2,[3 1 2]); ... 
          permute(markers.R_FMP2,[3 1 2]); ... 
          permute(markers.R_FM5,[3 1 2]); ... 
          permute(markers.R_FMP5,[3 1 2])];
clear temp ntemp;

% Set data (angles)
% temp = btkGetAngles(btk);
% ntemp = fieldnames(temp);
% for j = 1:length(ntemp)
%     angles.(ntemp{j}) = temp.(ntemp{j})(start:stop,:);                     % Keep only right cycle data
% end
% clear temp ntemp;
% data.Q = [...
%     angles.Pelvis_translations(:,1)'; ...
%     angles.Pelvis_translations(:,2)';...
%     deg2rad(angles.Pelvis_rotations(:,3))'; ...
%     deg2rad(angles.R_Hip(:,1))'; ...
%     deg2rad(angles.R_Knee(:,1))'; ...
%     deg2rad(angles.R_Ankle(:,1))' ...
%     ];
data.Q = [kalman.Q2(1,start:stop); ...
          kalman.Q2(2,start:stop); ...
          kalman.Q2(3,start:stop); ...
          kalman.Q2(4,start:stop); ...
          kalman.Q2(5,start:stop); ...
          kalman.Q2(6,start:stop)];

% Set data (GRF)
temp = btkGetGroundReactionWrenches(btk);
ntemp = fieldnames(temp);
for j = 1:length(ntemp)
    grf.(ntemp{j}) = temp(2).(ntemp{j})(start:stop,:);                     % Keep only right cycle data
end
clear temp ntemp;
data.R = [...
    grf.F(:,1:3)'; ...
    grf.M(:,1:3)'.*1e-3 ...
    ];

FPs = btkGetForcePlatforms(btk)
GRW = btkGetGroundReactionWrenches(btk)

M = zeros(308, 3)
M(:, 1) = FPs(1).channels.Mx1
M(:, 2) = FPs(1).channels.My1
M(:, 3) = FPs(1).channels.Mz1

figure()
plot(FPs(2).corners(1, :), FPs(2).corners(2, :), '+')
hold on 
plot(GRW(1).P(:, 1), GRW(1).P(:, 2))

figure()
plot(GRW(1).M)
hold on
plot(FPs(1).channels.Mx1,'--')
plot(FPs(1).channels.My1,'--')
plot(FPs(1).channels.Mz1,'--')

M_new = M + cross(GRW(1).P, GRW(1).F)

% Set data (EMGs)
temp = btkGetScalars(btk);
ntemp = fieldnames(temp);
for j = 1:length(ntemp)
    if max(temp.(ntemp{j})(start:stop,1)) > 1
        emg.(ntemp{j}) = temp.(ntemp{j})(start:stop,1)/...
            max(temp.(ntemp{j})(start:stop,1));                            % If MVC normalisation is not enough
    else
        emg.(ntemp{j}) = temp.(ntemp{j})(start:stop,1);                    % Keep only right cycle data
    end
end
clear temp ntemp;
data.EMG = [...
    emg.R_Tibialis_Anterior'; ...
    emg.R_Soleus'; ...
    emg.R_Gastrocnemius_Medialis'; ...
    emg.R_Gastrocnemius_Lateralis'; ...
    emg.R_Vastus_Medialis'; ...
    emg.R_Rectus_Femoris'; ...
    emg.R_Semitendinosus'; ...
    emg.R_Biceps_Femoris'; ...
    emg.R_Gluteus_Medius'; ...
    emg.R_Gluteus_Maximus' ...
    ];