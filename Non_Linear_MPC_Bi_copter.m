clc;
clear;
close all;
addpath('C:\Users\Aspire 7\Documents\Thesis\Simulation\mpc\Casadii');
import casadi.*

% Define physical parameters
L = 0.225;          % distance between motor and CoM of quadrotor (m)
h = 0.042;          % Vertical distance between CoG and center of the rotor (m)
Ixx = 0.116;        % kgm^2
Iyy = 0.0408;       % kgm^2
Izz = 0.105;        % kgm^2
m = 1.192;          % kg
g = 9.81;           % m/s^2
CT = 2.98e-6;       % lift constant - thrust factor (Ns^2/rad^2)

% Define the non-linear dynamics using CasADi
x = MX.sym('x', 12);
u = MX.sym('u', 4);
w = MX.sym('w', 3); % Wind disturbance

phi = x(4);
theta = x(5);
psi = x(6);
x_dot = x(7);
y_dot = x(8);
z_dot = x(9);
phi_dot = x(10);
theta_dot = x(11);
psi_dot = x(12);

T1 = u(1);  % Total thrust
tau_phi = u(2);  % Roll torque
tau_theta = u(3);  % Pitch torque
tau_psi = u(4);  % Yaw torque

R = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta);
     sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*cos(theta);
     cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*cos(theta)];

F_gravity = [0; 0; m*g];
F_thrust = [0; 0; -T1];
acc = (1/m) * (R * F_thrust + F_gravity) + w;

omega = [phi_dot; theta_dot; psi_dot];
tau = [tau_phi; tau_theta; tau_psi];
I = diag([Ixx, Iyy, Izz]);
omega_dot = I \ (tau - cross(omega, I*omega));

dxdt = [x_dot; y_dot; z_dot; phi_dot; theta_dot; psi_dot; acc; omega_dot];

dynamics = Function('dynamics', {x, u, w}, {dxdt});

% Initial state and reference trajectory
dt = 0.1;
T_final = 100;  % Final time
N = T_final / dt;  % Number of time steps

% Define reference trajectory (linear uprising path)
xref = zeros(12, N+1);
for i = 1:N+1
    xref(1, i) = i * dt * 0.01;  % Linear increment in X direction
    xref(2, i) = i * dt * 0.005; % Linear increment in Y direction
    xref(3, i) = i * dt * 0.01;  % Linear increment in Z direction (upward)
end

% Define CasADi variables for the NMPC problem
Hp = 20;  % Prediction horizon
X = MX.sym('X', 12, Hp+1);
U = MX.sym('U', 4, Hp);
X_ref = MX.sym('X_ref', 12, Hp+1);

% Define weights
Q_pos = diag([100, 100, 100]);
Q_ori = diag([10, 10, 10]);
R = eye(4);

% Normalize the weights
max_pos_error = 1;  % Assume maximum position error of 1 meter
max_ori_error = pi;  % Assume maximum orientation error of pi radians
max_control = 10;  % Maximum control input value

Q_pos_norm = Q_pos / (max_pos_error^2);
Q_ori_norm = Q_ori / (max_ori_error^2);
R_norm = R / (max_control^2);

% Initialize the cost function and constraints
cost = 0;
constraints = [];

% Define the cost function and constraints over the prediction horizon
for k = 1:Hp
    state_error = X(1:3, k) - X_ref(1:3, k);
    orientation_error = X(4:6, k) - X_ref(4:6, k);
    
    cost = cost + state_error' * Q_pos_norm * state_error + orientation_error' * Q_ori_norm * orientation_error;
    cost = cost + U(:, k)' * R_norm * U(:, k);
    
    x_next = X(:, k) + dt * dynamics(X(:, k), U(:, k), MX.zeros(3,1));
    
    % Ensure both are dense before concatenation
    constraints = [constraints; full(X(:, k+1)) - full(x_next)];
end

% Terminal cost
state_error_terminal = X(1:3, Hp+1) - X_ref(1:3, Hp+1);
orientation_error_terminal = X(4:6, Hp+1) - X_ref(4:6, Hp+1);
cost = cost + state_error_terminal' * Q_pos_norm * state_error_terminal + orientation_error_terminal' * Q_ori_norm * orientation_error_terminal;

% Define optimization variables and parameters
opt_vars = [reshape(X, 12*(Hp+1), 1); reshape(U, 4*Hp, 1)];
p = reshape(X_ref, 12*(Hp+1), 1);

% Define bounds for the control inputs
u_min = -10 * ones(4, 1);
u_max = 10 * ones(4, 1);

% Create NLP problem
nlp_prob = struct('f', cost, 'x', opt_vars, 'g', vertcat(constraints), 'p', p);

% Define solver options
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter = 1000;
opts.ipopt.tol = 1e-6;

% Create solver
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% Define argument structure
args = struct;
args.lbg = zeros(length(constraints), 1);  % Lower bound on g
args.ubg = zeros(length(constraints), 1);  % Upper bound on g
args.lbx = [-inf*ones(12*(Hp+1), 1); repmat(u_min, Hp, 1)];
args.ubx = [inf*ones(12*(Hp+1), 1); repmat(u_max, Hp, 1)];

% Initialize state and input trajectories
x_traj = zeros(12, N+1);
u_traj = zeros(4, N);
comp_time_nmpc = zeros(N, 1); % To store computation times

% Set initial state (assume the system starts from the origin)
x0 = zeros(12,1); % Initial state vector
x_traj(:,1) = x0;

% Simulate closed-loop system with disturbance
wind_disturbance = 0.1 * randn(3, N); % Wind disturbance (Gaussian noise)
for k = 1:N
    % Define the current reference trajectory for the prediction horizon
    current_ref = xref(:,k:min(k+Hp,N));
    if size(current_ref, 2) < Hp+1
        current_ref = [current_ref, repmat(xref(:,end), 1, Hp+1-size(current_ref,2))];
    end
    
    % Set initial guesses for optimization variables
    args.p = reshape(current_ref, 12*(Hp+1), 1);
    args.x0 = [reshape(repmat(x_traj(:,k), 1, Hp+1), 12*(Hp+1), 1); zeros(4*Hp, 1)];
    
    % Start timer
    tic;
    
    % Solve the NMPC problem
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    
    % Record computation time
    comp_time_nmpc(k) = toc;

    % Extract the optimal control input
    u_opt = reshape(full(sol.x(12*(Hp+1)+1:end)), 4, Hp);
    
    % Apply the first control input
    u_traj(:,k) = u_opt(:,1);
    
    % Update the state using RK4 integration with disturbance
    w_k = wind_disturbance(:,k);
    k1 = full(dynamics(x_traj(:,k), u_traj(:,k), w_k));
    k2 = full(dynamics(x_traj(:,k) + 0.5*dt*k1, u_traj(:,k), w_k));
    k3 = full(dynamics(x_traj(:,k) + 0.5*dt*k2, u_traj(:,k), w_k));
    k4 = full(dynamics(x_traj(:,k) + dt*k3, u_traj(:,k), w_k));
    x_traj(:,k+1) = x_traj(:,k) + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% Verify trajectory data
disp('Trajectory data:');
disp(x_traj(1:3,:));

% KPI Calculation
% Control Input Error Calculation (RMSE)
u_ref = zeros(size(u_traj));  % Assuming reference control inputs are zeros or modify based on your reference
control_input_error_nmpc = sqrt(mean(sum((u_traj - u_ref).^2, 2)));

% Computation Time Calculation
avg_comp_time_nmpc = mean(comp_time_nmpc);
total_comp_time_nmpc = sum(comp_time_nmpc);

% Display KPI results
disp('KPI for NMPC');
fprintf('Control Input Error (RMSE): %.6f\n', control_input_error_nmpc);
fprintf('Avg. Computation Time: %.6f s\n', avg_comp_time_nmpc);
fprintf('Total Computation Time: %.6f s\n', total_comp_time_nmpc);

% Plot the non-linear trajectory vs the reference trajectory
figure;
plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:), 'b', 'LineWidth', 1.5);
hold on;
plot3(xref(1,:), xref(2,:), xref(3,:), 'r--', 'LineWidth', 1.5);
grid on;
title('Bi-Copter Non-linear Trajectory Tracking with Wind Disturbance');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
legend('Non-linear Trajectory', 'Reference Trajectory');

% Plot position error over time
position_error = sqrt(sum((x_traj(1:3,:) - xref(1:3,:)).^2, 1));
figure;
plot(0:dt:T_final, position_error, 'LineWidth', 1.5);
title('Position Error Over Time');
xlabel('Time (s)');
ylabel('Position Error (m)');

% Adjust time vector for plotting control inputs
t_u = 0:dt:(T_final-dt); % Time vector for control inputs plot

% Plot the control inputs over time
figure;
subplot(4,1,1);
plot(t_u, u_traj(1,:), 'LineWidth', 1.5);
title('Control Inputs');
xlabel('Time (s)');
ylabel('u_1');

subplot(4,1,2);
plot(t_u, u_traj(2,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_2');

subplot(4,1,3);
plot(t_u, u_traj(3,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_3');

subplot(4,1,4);
plot(t_u, u_traj(4,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u_4');