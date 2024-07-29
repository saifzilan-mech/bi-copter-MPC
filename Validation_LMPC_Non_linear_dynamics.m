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


% Initial state and simulation parameters
dt = 0.1;
T_final = 100;  % Final time
N = T_final / dt;  % Number of time steps
x0 = zeros(12,1); % Initial state vector

% Control inputs for the nonlinear simulation (example inputs)
u_nl = [ones(1, N); zeros(3, N)]; % Maintain a constant thrust

% Simulate the non-linear system to generate the reference trajectory
params.m = m;
params.Ixx = Ixx;
params.Iyy = Iyy;
params.Izz = Izz;
params.g = g;
params.L = L;
params.CT = CT;

t = 0:dt:T_final;

x_nl_traj = zeros(12, length(t));
x_nl_traj(:,1) = x0;

for k = 1:length(t)-1
    u_k = u_nl(:,k);
    [~, x_next] = ode45(@(t, x) nonlinear_dynamics(t, x, u_k, params), [t(k), t(k+1)], x_nl_traj(:,k));
    x_nl_traj(:,k+1) = x_next(end,:)';
end

% Use this trajectory as the reference for the MPC
xref = x_nl_traj;
% Define continuous-time state-space matrices
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
Q_pos = diag([100 100 100]);  % Increased weight for position
Q_ori = diag([10 10 10]);     % Increased weight for orientation
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

% Set initial state (assume the system starts from the origin)
x0 = zeros(12,1); % Initial state vector
x_traj(:,1) = x0;

% Simulate closed-loop system
for k = 1:N
    % Define the current reference trajectory for the prediction horizon
    current_ref = xref(:,k:min(k+Hp,N));
    if size(current_ref, 2) < Hp+1
        current_ref = [current_ref, repmat(xref(:,end), 1, Hp+1-size(current_ref,2))];
    end
    
    % Solve the LMPC problem
    u_opt = controller{x_traj(:,k), current_ref};
    
    % Apply the first control input
    u_traj(:,k) = u_opt;
    
    % Update the state
    x_traj(:,k+1) = A_disc*x_traj(:,k) + B_disc*u_traj(:,k);
end

% Validate with non-linear dynamics
% Parameters
params.m = m;
params.Ixx = Ixx;
params.Iyy = Iyy;
params.Izz = Izz;
params.g = g;
params.L = L;
params.CT = CT;

% Time vector
t = 0:dt:T_final;

% Pre-allocate state trajectory
x_nl_traj_mpc = zeros(12, length(t));
x_nl_traj_mpc(:,1) = x0;

% Simulate the non-linear system using the control inputs from the linear MPC
for k = 1:length(t)-1
    u_k = u_traj(:,k);
    [~, x_next] = ode45(@(t, x) nonlinear_dynamics(t, x, u_k, params), [t(k), t(k+1)], x_nl_traj_mpc(:,k));
    x_nl_traj_mpc(:,k+1) = x_next(end,:)';
end

% Plot the non-linear trajectory vs the reference trajectory
figure;
plot3(x_nl_traj_mpc(1,:), x_nl_traj_mpc(2,:), x_nl_traj_mpc(3,:), 'b', 'LineWidth', 1.5);
hold on;
plot3(xref(1,:), xref(2,:), xref(3,:), 'r--', 'LineWidth', 1.5);
grid on;
title('Bi-Copter Non-linear Trajectory Tracking');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
legend('Non-linear Trajectory', 'Reference Trajectory');

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

% Visualize the bi-copter with wings at certain intervals
figure;
hold on;
plot3(xref(1,:), xref(2,:), xref(3,:), 'g', 'LineWidth', 2);

% Plot the bi-copter's trajectory and its orientation at intervals
for i = 1:round(N/10):N
    phi = x_nl_traj_mpc(4, i);
    theta = x_nl_traj_mpc(5, i);
    psi = x_nl_traj_mpc(6, i);
    
    xm2 = x_nl_traj_mpc(1, i) + L * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta));
    ym2 = x_nl_traj_mpc(2, i) - L * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta));
    zm2 = x_nl_traj_mpc(3, i) - L * cos(theta) * sin(phi);
    
    xm4 = x_nl_traj_mpc(1, i) - L * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta));
    ym4 = x_nl_traj_mpc(2, i) + L * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta));
    zm4 = x_nl_traj_mpc(3, i) + L * cos(theta) * sin(phi);
    
    line([x_nl_traj_mpc(1, i); xm2], [x_nl_traj_mpc(2, i); ym2], [x_nl_traj_mpc(3, i); zm2], 'color', [0.5, 0.5, 0.5]);
    line([x_nl_traj_mpc(1, i); xm4], [x_nl_traj_mpc(2, i); ym4], [x_nl_traj_mpc(3, i); zm4], 'color', [0.5, 0.5, 0.5]);

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

% Define the nonlinear dynamics function
function dxdt = nonlinear_dynamics(t, x, u, params)
    % Extract parameters
    m = params.m;
    Ixx = params.Ixx;
    Iyy = params.Iyy;
    Izz = params.Izz;
    g = params.g;
    L = params.L;
    CT = params.CT;

    % State variables
    phi = x(4);
    theta = x(5);
    psi = x(6);
    x_dot = x(7);
    y_dot = x(8);
    z_dot = x(9);
    phi_dot = x(10);
    theta_dot = x(11);
    psi_dot = x(12);

    % Control inputs
    T1 = u(1);  % Total thrust
    tau_phi = u(2);  % Roll torque
    tau_theta = u(3);  % Pitch torque
    tau_psi = u(4);  % Yaw torque

    % Rotation matrix
    R = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta);
         sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*cos(theta);
         cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*cos(theta)];

    % Translational dynamics
    F_gravity = [0; 0; m*g];
    F_thrust = [0; 0; -T1];
    acc = (1/m) * (R * F_thrust + F_gravity);

    % Rotational dynamics
    omega = [phi_dot; theta_dot; psi_dot];
    tau = [tau_phi; tau_theta; tau_psi];
    I = diag([Ixx, Iyy, Izz]);
    omega_dot = I \ (tau - cross(omega, I*omega));

    % State derivatives
    dxdt = zeros(12,1);
    dxdt(1:3) = x(7:9);
    dxdt(4:6) = omega;
    dxdt(7:9) = acc;
    dxdt(10:12) = omega_dot;
end

