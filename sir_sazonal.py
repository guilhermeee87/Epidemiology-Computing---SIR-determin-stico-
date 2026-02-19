import numpy as np
import matplotlib.pyplot as plt

# Parâmetros epidemiológicos
beta0 = 0.4          # taxa média de transmissão
gamma = 1/7          # taxa de recuperação (7 dias infeccioso)
alpha = 0.2          # intensidade da sazonalidade
N = 1_000_000        # população total

# Condições iniciais
I0 = 10
R0 = 0
S0 = N - I0 - R0

# Tempo
t_max = 365 * 3      # 3 anos
dt = 0.1
steps = int(t_max / dt)

# Vetores
S = np.zeros(steps)
I = np.zeros(steps)
R = np.zeros(steps)
t = np.linspace(0, t_max, steps)

S[0] = S0
I[0] = I0
R[0] = R0

def beta(t):
    return beta0 * (1 + alpha * np.cos(2 * np.pi * t / 365))

def deriv(s, i, r, time):
    b = beta(time)
    dS = -b * s * i / N
    dI = b * s * i / N - gamma * i
    dR = gamma * i
    return dS, dI, dR

# Integração RK4
for k in range(steps - 1):
    s, i, r = S[k], I[k], R[k]

    k1 = deriv(s, i, r, t[k])
    k2 = deriv(s + dt*k1[0]/2, i + dt*k1[1]/2, r + dt*k1[2]/2, t[k] + dt/2)
    k3 = deriv(s + dt*k2[0]/2, i + dt*k2[1]/2, r + dt*k2[2]/2, t[k] + dt/2)
    k4 = deriv(s + dt*k3[0], i + dt*k3[1], r + dt*k3[2], t[k] + dt)

    S[k+1] = s + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    I[k+1] = i + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    R[k+1] = r + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

# Gráfico
plt.figure(figsize=(10,6))
plt.plot(t, S/N, label="Suscetíveis")
plt.plot(t, I/N, label="Infectados")
plt.plot(t, R/N, label="Recuperados")
plt.xlabel("Tempo (dias)")
plt.ylabel("Proporção da população")
plt.legend()
plt.title("Modelo SIR com Transmissão Sazonal")
plt.show()
