# Nanothermodynamics & Computational Heat Transfer

I work on modeling and simulation of heat and energy transport at the nanoscale,  
with a focus on thermal wave propagation, entropy dynamics, and non-equilibrium systems.  
My goal is to develop predictive computational models that support experimental  
research in advanced materials and nanotechnology.

---

## Research Interests
- Nanoscale heat transfer
- Thermal wave propagation & non-Fourier effects
- Entropy dynamics and non-equilibrium thermodynamics
- Computational physics and scientific computing
- Simulation-driven experimental design

---

## Tools & Methods
- **Programming Languages:** Python (NumPy, SciPy, Matplotlib)
- **Numerical Modeling:** Finite Difference, Finite Element Methods
- **Simulation & Analysis:** Data visualization, parameter studies, model validation
- **Version Control:** Git & GitHub
- **Other Tools:** Jupyter Notebook, GitHub CLI

---

## Goals
- Bridge theoretical insights with numerical simulations
- Provide actionable insights for experimental design
- Explore new physical phenomena in advanced materials
- Accelerate innovation in thermal management and nanoscale devices

---

## Simulation Code Example

```python
# =========================
# thermal_wave_simulation.py
# Baseline 1D thermal transport model (Fourier framework)
# Tools: Python, NumPy, SciPy, Matplotlib
# =========================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# Simulation Parameters
# -------------------------
L       = 1e-6       # Length of the material (1 micron)
Nx      = 200        # Number of spatial points
x       = np.linspace(0, L, Nx)
alpha   = 1e-7       # Thermal diffusivity (m^2/s)
t_max   = 1e-9       # Maximum simulation time (s)
t_eval  = np.linspace(0, t_max, 500)

# -------------------------
# Initial Condition: Gaussian Thermal Pulse
# -------------------------
def initial_condition(x):
    return np.exp(-((x - L/2)**2) / (2 * (L/20)**2))

u0 = initial_condition(x)

# -------------------------
# 1D Heat Diffusion Equation (Fourier-type)
# -------------------------
def thermal_transport(t, u):
    dudt = np.zeros_like(u)
    dudt[1:-1] = alpha * (u[2:] - 2*u[1:-1] + u[:-2]) / (x[1] - x[0])**2
    return dudt

# -------------------------
# Solve PDE using Method of Lines
# -------------------------
solution = solve_ivp(
    thermal_transport,
    [0, t_max],
    u0,
    t_eval=t_eval,
    method='RK45'
)

# -------------------------
# Visualization
# -------------------------
plt.figure(figsize=(8, 4))
for i in range(0, len(t_eval), 50):
    plt.plot(x * 1e6, solution.y[:, i], label=f't = {t_eval[i]*1e9:.2f} ns')

plt.xlabel('Position (µm)')
plt.ylabel('Temperature (arbitrary units)')
plt.title('Thermal Transport at the Nanoscale (Baseline Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()