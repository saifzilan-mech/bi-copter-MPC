# Bi-Copter Model Predictive Control (MPC) Project

## Overview

This project explores the dynamics and control of a bi-copter using Model Predictive Control (MPC). The bi-copter, an unmanned aerial vehicle (UAV), benefits from reduced costs, vibrations, and power demands due to its simplified design with fewer motors and propellers. However, the complex and nonlinear dynamics of bi-copters pose significant control challenges.

## Objectives

- Develop and validate both Linear MPC (LMPC) and Nonlinear MPC (NMPC) frameworks for trajectory tracking.
- Compare the performance of LMPC and NMPC in terms of stability, robustness, and computational efficiency.
- Ensure the control system's adaptability to environmental changes such as wind disturbances and payload variations.

## Methodology

1. **Dynamic Modeling**: Developed a detailed nonlinear dynamic model with six degrees of freedom.
2. **Control Strategy Design**: Implemented MPC to follow various reference trajectories, from simple circular paths to complex helical patterns.
3. **Linear and Nonlinear MPC**:
   - **LMPC**: Linearized the model around hover and applied LMPC using the YALMIP toolbox in MATLAB.
   - **NMPC**: Developed NMPC to handle the true nonlinear dynamics of the bi-copter more accurately.
4. **Simulation and Validation**: Conducted multiple simulation scenarios in MATLAB to test and assess the performance of both control methods under various conditions.

## Results

- **Control Performance**:
  - LMPC showed lower control input error (RMSE: 0.532833) and higher computational efficiency (average computation time: 0.011802 seconds) compared to NMPC.
  - NMPC, although potentially more powerful in handling nonlinear dynamics, required further optimization to reduce control input error and computation time.
- **Trajectory Tracking**:
  - Both LMPC and NMPC effectively maintained the desired trajectory despite disturbances, with LMPC showing superior initial control and stability over longer periods.

## Future Work

- Enhance the robustness of both control strategies against persistent disturbances by integrating adaptive control techniques or robust MPC frameworks.
- Implement real-time optimization methods and explore machine learning approaches for predictive modeling and control.
- Extensive testing under various environmental conditions to ensure reliability and effectiveness in practical applications.


## License

This project is licensed under the MIT License.
