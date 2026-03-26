import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. DEFINICJE UKŁADÓW (DYNAMIKA) ---

def system_step(t, z):
    """Układ z wymuszeniem skokowym u(t) = 1"""
    y, dy, d2y = z
    u_t = 1.0  # Skok jednostkowy
    d3y = -7*d2y - 11*dy - 5*y + u_t
    return [dy, d2y, d3y]

def system_impulse(t, z):
    """Układ z wymuszeniem impulsowym (realizowane przez warunki początkowe)"""
    y, dy, d2y = z
    u_t = 0.0  # Brak ciągłego wymuszenia
    d3y = -7*d2y - 11*dy - 5*y + u_t
    return [dy, d2y, d3y]

# --- 2. PARAMETRY SYMULACJI ---
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# --- 3. SYMULACJE NUMERYCZNE ---
# Dla skoku: zerowe warunki początkowe
sol_step = solve_ivp(system_step, t_span, [0, 0, 0], t_eval=t_eval)

# Dla impulsu: y''(0) = 1 (odpowiednik impulsu Diraca w chwili t=0)
sol_impulse = solve_ivp(system_impulse, t_span, [0, 0, 1], t_eval=t_eval)

# --- 4. ROZWIĄZANIA ANALITYCZNE (Z NASZYCH OBLICZEŃ) ---
# Odpowiedź skokowa
y_step_an = 0.2 - 0.0125*np.exp(-5*t_eval) - 0.1875*np.exp(-t_eval) - 0.25*t_eval*np.exp(-t_eval)

# Odpowiedź impulsowa
y_impulse_an = (1/16)*np.exp(-5*t_eval) - (1/16)*np.exp(-t_eval) + 0.25*t_eval*np.exp(-t_eval)

# --- 5. WIZUALIZACJA ---
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Wykres odpowiedzi skokowej
ax1.plot(sol_step.t, sol_step.y[0], 'b-', linewidth=2, label='Numerycznie (solve_ivp)')
ax1.plot(t_eval, y_step_an, 'r--', linewidth=2, label='Analitycznie (wzór)')
ax1.set_title('Odpowiedź skokowa układu (wymuszenie $u(t)=1$)', fontsize=14)
ax1.set_ylabel('Amplituda $y(t)$')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Wykres odpowiedzi impulsowej
ax2.plot(sol_impulse.t, sol_impulse.y[0], 'b-', linewidth=2, label='Numerycznie (warunek początkowy)')
ax2.plot(t_eval, y_impulse_an, 'r--', linewidth=2, label='Analitycznie (wzór)')
ax2.set_title('Odpowiedź impulsowa układu (wymuszenie $u(t)=\delta(t)$)', fontsize=14)
ax2.set_xlabel('Czas [s]')
ax2.set_ylabel('Amplituda $y(t)$')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()