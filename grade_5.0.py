import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# ==========================================
# CZĘŚĆ 1: ANALITYCZNA (SYMPY)
# ==========================================
s, t = sp.symbols('s t')

# Mianownik układu
mianownik = s**3 + 7*s**2 + 11*s + 5

# Wzory Y(s) dla poszczególnych przypadków:
# 1. Baza: y(0)=0, y'(0)=0, y''(0)=0
Y_base = 1 / (s * mianownik)

# 2. Tylko y(0) = 1
Y_y0 = Y_base + (s**2 + 7*s + 11) / mianownik

# 3. Tylko y'(0) = 1
Y_dy0 = Y_base + (s + 7) / mianownik

# 4. Tylko y''(0) = 1
Y_d2y0 = Y_base + 1 / mianownik

print("--- ANALITYCZNE WZORY (Rozkład na ułamki proste) ---")
print("1. Wpływ y(0)=1:")
print(sp.apart(Y_y0))
print("\n2. Wpływ y'(0)=1:")
print(sp.apart(Y_dy0))
print("\n3. Wpływ y''(0)=1:")
print(sp.apart(Y_d2y0))
print("----------------------------------------------------\n")

# ==========================================
# CZĘŚĆ 2: SYMULACJA NUMERYCZNA I WYKRES
# ==========================================
def system(t, z):
    y, dy, d2y = z
    u_t = 1.0  # Ciągle wymuszamy skokiem jednostkowym
    d3y = -7*d2y - 11*dy - 5*y + u_t
    return [dy, d2y, d3y]

t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Rozwiązujemy równanie dla 4 różnych zestawów warunków początkowych [y(0), y'(0), y''(0)]
sol_base = solve_ivp(system, t_span, [0, 0, 0], t_eval=t_eval) # Zwykła odpowiedź skokowa
sol_y0   = solve_ivp(system, t_span, [1, 0, 0], t_eval=t_eval) # Zmienione y(0)
sol_dy0  = solve_ivp(system, t_span, [0, 1, 0], t_eval=t_eval) # Zmienione y'(0)
sol_d2y0 = solve_ivp(system, t_span, [0, 0, 1], t_eval=t_eval) # Zmienione y''(0)

# Rysowanie wykresów
plt.style.use('seaborn-v0_8-muted')
plt.figure(figsize=(12, 7))

plt.plot(sol_base.t, sol_base.y[0], 'k--', linewidth=2, label='Baza (warunki zerowe)')
plt.plot(sol_y0.t, sol_y0.y[0], 'r-', linewidth=2, label="$y(0)=1$ (start z wyższej pozycji)")
plt.plot(sol_dy0.t, sol_dy0.y[0], 'g-', linewidth=2, label="$y'(0)=1$ (dodatkowa prędkość pocz.)")
plt.plot(sol_d2y0.t, sol_d2y0.y[0], 'b-', linewidth=2, label="$y''(0)=1$ (dodatkowe przyspieszenie pocz.)")

plt.title('Wpływ niezerowych warunków początkowych na odpowiedź skokową układu', fontsize=14)
plt.xlabel('Czas [s]', fontsize=12)
plt.ylabel('Amplituda $y(t)$', fontsize=12)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()