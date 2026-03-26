import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Definicja układu równań różniczkowych (model systemu)
def system_dynamics(t, z):
    y, dy, d2y = z
    # Nasze wymuszenie u(t) = e^(-6t)
    u_t = np.exp(-6 * t)
    
    # Równanie: y''' = -7y'' - 11y' - 5y + u(t)
    d3y = -7*d2y - 11*dy - 5*y + u_t
    
    return [dy, d2y, d3y]

# 2. Parametry symulacji
t_span = (0, 10)  # Czas od 0 do 10 sekund
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Punkty, w których chcemy wynik
initial_conditions = [0, 0, 0]  # y(0)=0, y'(0)=0, y''(0)=0

# 3. Rozwiązanie numeryczne (Solver ODE)
sol = solve_ivp(system_dynamics, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# 4. Rozwiązanie analityczne (wyliczone przez nas wcześniej)
# y(t) = -1/25 * e^(-6t) + 1/16 * e^(-5t) - 9/400 * e^(-t) + 1/20 * t * e^(-t)
y_analytical = (-(1/25) * np.exp(-6 * t_eval) + 
                (1/16) * np.exp(-5 * t_eval) - 
                (9/400) * np.exp(-t_eval) + 
                (1/20) * t_eval * np.exp(-t_eval))

# 5. Wizualizacja wyników
plt.style.use('seaborn-v0_8-muted') # Nowoczesny wygląd wykresu
plt.figure(figsize=(10, 6))

plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Symulacja numeryczna (solve_ivp)')
plt.plot(t_eval, y_analytical, 'r--', linewidth=2, label='Rozwiązanie analityczne (Laplace)')

plt.title('Odpowiedź układu y(t) na wymuszenie $u(t) = e^{-6t}$', fontsize=12)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda $y(t)$')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()