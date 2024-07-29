clc;
clear;
close all;

% Define physical parameters
L = 0.225;          % distance between motor and CoM of quadrotor (m)
h = 0.042;          % Vertical distance between CoG and center of the rotor (m)
Ixx = 0.116;        % kgm^2
Iyy = 0.0408;       % kgm^2
Izz = 0.105;        % kgm^2
m = 1.192;          % kg
g = 9.81;           % m/s^2
CT = 2.98e-6;       % lift constant - thrust factor (Ns^2/rad^2)

% Continuous-time state-space matrices
A_cont = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      
B_cont = [0, 0, 0, 0;
          0, 0, 0, 0;
          0, 0, 0, 0;
          1/m, 0, 0, 0;
          0, 1/Ixx, 0, 0;
          0, 0, 1/Iyy, 0;
          0, 0, 0, 1/Izz;
          0, 0, 0, 0;
          0, 0, 0, 0;
          0, 0, 0, 0;
          0, 0, 0, 0;
          0, 0, 0, 0];
      
C_cont = eye(12);  % Output all states
D_cont = zeros(12, 4);  % No direct feedthrough

% Initial state and reference trajectory
dt = 0.1;
T_final = 100;  % Final time
N = T_final/dt;  % Number of time steps

% Create continuous-time state-space system
sys_cont = ss(A_cont, B_cont, C_cont, D_cont);

% Discretize the system using zero-order hold (ZOH)
sys_disc = c2d(sys_cont, dt, 'zoh');

% Extract the discrete-time state-space matrices
A_disc = sys_disc.A;
B_disc = sys_disc.B;
C_disc = sys_disc.C;
D_disc = sys_disc.D;

% Define the prediction horizon and weights
Hp = 20;  % Prediction horizon
Q_pos = diag([10 10 10]);  % Weight for position
Q_ori = diag([1 1 1]);  % Weight for orientation
R = eye(4);  % Control weighting matrix

% Normalization factors
x_norm = diag([1 1 1 1 1 1 1 1 1 1 1 1]);  % Example normalization factors for states
u_norm = diag([1 1 1 1]);  % Example normalization factors for controls

% Define YALMIP variables for state trajectory and control inputs
x = sdpvar(12, Hp+1);  % State trajectory
u = sdpvar(4, Hp);     % Control inputs

% Define YALMIP variable for reference trajectory
xref_param = sdpvar(12, Hp+1);  % Reference trajectory parameter

% Normalized cost function
cost = 0;
constraints = [];

for k = 1:Hp
    % Track position and orientation (modify weights as needed)
    cost = cost + (x_norm(1:3,1:3)*(x(1:3,k) - xref_param(1:3,k)))'*Q_pos*(x_norm(1:3,1:3)*(x(1:3,k) - xref_param(1:3,k))) + ...
                  (x_norm(4:6,1:3)*(x(4:6,k) - xref_param(4:6,k)))'*Q_ori*(x_norm(4:6,1:3)*(x(4:6,k) - xref_param(4:6,k)));
    % Minimize control effort
    cost = cost + (u_norm*u(:,k))'*R*(u_norm*u(:,k));
    constraints = [constraints, x(:,k+1) == A_disc*x(:,k) + B_disc*u(:,k)];

end
% Terminal cost
cost = cost + (x_norm(1:3,1:3)*(x(1:3,Hp+1) - xref_param(1:3,Hp+1)))'*Q_pos*(x_norm(1:3,1:3)*(x(1:3,Hp+1) - xref_param(1:3,Hp+1))) + ...
                (x_norm(4:6,1:3)*(x(4:6,Hp+1) - xref_param(4:6,Hp+1)))'*Q_ori*(x_norm(4:6,1:3)*(x(4:6,Hp+1) - xref_param(4:6,Hp+1)));

% Input constraints (if any)
u_min = -10*ones(4, 1);  % Lower bounds on inputs
u_max = 10*ones(4, 1);   % Upper bounds on inputs

for k = 1:Hp
    constraints = [constraints, u_min <= u(:,k) <= u_max];
end


% Set options for solver
options = sdpsettings('solver', 'quadprog', 'verbose', 0);

% Define the optimization problem
controller = optimizer(constraints, cost, options, {x(:,1), xref_param}, u(:,1));

% Initialize state and input trajectories
x_traj = zeros(12, N+1);
u_traj = zeros(4, N);
comp_time_lmpc = zeros(N, 1); % To store computation times

% Set initial state (assume the system starts from the origin)
x0 = zeros(12,1); % Initial state vector
x_traj(:,1) = x0;

% Define reference trajectory (example: moving in a spiral)
xref = zeros(12, N+1);
R_spiral = 0.1;  % Radius of the spiral
z_spiral = 0.1;  % Height change per revolution
for i = 1:N+1
    theta_spiral = 2 * pi * i / (N / 10);  % Angle for each point
    xref(1, i) = R_spiral * cos(theta_spiral);
    xref(2, i) = R_spiral * sin(theta_spiral);
    xref(3, i) = z_spiral * theta_spiral / (2 * pi);
end

% Simulate closed-loop system
for k = 1:N
    % Start timer
    tic;
    
    % Define the current reference trajectory for the prediction horizon
    current_ref = xref(:,k:min(k+Hp,N));
    if size(current_ref, 2) < Hp+1
        current_ref = [current_ref, repmat(xref(:,end), 1, Hp+1-size(current_ref,2))];
    end
    
    % Solve the LMPC problem
    u_opt = controller{x_traj(:,k), current_ref};
    
    % Record computation time
    comp_time_lmpc(k) = toc;

    % Apply the first control input
    u_traj(:,k) = u_opt;
    
    % Update the state
    x_traj(:,k+1) = A_disc*x_traj(:,k) + B_disc*u_traj(:,k);
end

% Verify trajectory data
disp('Trajectory data:');
disp(x_traj(1:3,:));

% KPI Calculation
% Control Input Error Calculation (RMSE)
u_ref = zeros(size(u_traj));  % Assuming reference control inputs are zeros or modify based on your reference
control_input_error_lmpc = sqrt(mean(sum((u_traj - u_ref).^2, 2)));

% Computation Time Calculation
avg_comp_time_lmpc = mean(comp_time_lmpc);
total_comp_time_lmpc = sum(comp_time_lmpc);

% Display KPI results
disp('KPI for LMPC');
fprintf('Control Input Error (RMSE): %.6f\n', control_input_error_lmpc);
fprintf('Avg. Computation Time: %.6f s\n', avg_comp_time_lmpc);
fprintf('Total Computation Time: %.6f s\n', total_comp_time_lmpc);

% Plot the bi-copter's trajectory vs the reference trajectory
figure;
plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:), 'b', 'LineWidth', 1.5);
hold on;
plot3(xref(1,:), xref(2,:), xref(3,:), 'r--', 'LineWidth', 1.5);
grid on;
title('Bi-Copter Trajectory Tracking');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
legend('Actual Trajectory', 'Reference Trajectory');

% Plot the control inputs over time
figure;
t = 0:dt:T_final-dt; % Time vector for control inputs plot
subplot(4,1,1);
plot(t, u_traj(1,:), 'LineWidth', 1.5);
title('Control Inputs');
xlabel('Time (s)');
ylabel('u_1');

subplot(4,1,2);
plot(t, u_traj(2,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_2');

subplot(4,1,3);
plot(t, u_traj(3,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_3');

subplot(4,1,4);
plot(t, u_traj(4,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_4');

% Visualize the bi-copter with wings at certain intervals
figure;
hold on;
plot3(xref(1,:), xref(2,:), xref(3,:), 'g', 'LineWidth', 2);

% Plot the bi-copter's trajectory and its orientation at intervals
for i = 1:round(N/10):N
    phi = x_traj(4, i);
    theta = x_traj(5, i);
    psi = x_traj(6, i);
    
    xm2 = x_traj(1, i) + L * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta));
    ym2 = x_traj(2, i) - L * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta));
    zm2 = x_traj(3, i) - L * cos(theta) * sin(phi);
    
    xm4 = x_traj(1, i) - L * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta));
    ym4 = x_traj(2, i) + L * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta));
    zm4 = x_traj(3, i) + L * cos(theta) * sin(phi);
    
    line([x_traj(1, i); xm2], [x_traj(2, i); ym2], [x_traj(3, i); zm2], 'color', [0.5, 0.5, 0.5]);
    line([x_traj(1, i); xm4], [x_traj(2, i); ym4], [x_traj(3, i); zm4], 'color', [0.5, 0.5, 0.5]);

    % Plot the wings
    [xx, yy, zz] = cylinder;
    R = 0.239 / 2;  % radius of propeller (m)
    thickness = 0.03;  % thickness of rotating propeller (m)
    
    w2 = surf(R * xx + xm2, R * yy + ym2, -thickness * zz + zm2);
    rotate(w2, [1, 0, 0], phi * 180/pi, [xm2, ym2, zm2]);
    rotate(w2, [0, 1, 0], theta * 180/pi, [xm2, ym2, zm2]);
    rotate(w2, [0, 0, 1], psi * 180/pi, [xm2, ym2, zm2]);
    set(w2, 'FaceColor', [0, 0.5, 1], 'EdgeColor', [0, 0.5, 1]);

    w4 = surf(R * xx + xm4, R * yy + ym4, -thickness * zz + zm4);
    rotate(w4, [1, 0, 0], phi * 180/pi, [xm4, ym4, zm4]);
    rotate(w4, [0, 1, 0], theta * 180/pi, [xm4, ym4, zm4]);
    rotate(w4, [0, 0, 1], psi * 180/pi, [xm4, ym4, zm4]);
    set(w4, 'FaceColor', [0, 0, 1], 'EdgeColor', [0, 0, 1]);
end

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal;
grid on;
box on;
view([1, 1, 1]);