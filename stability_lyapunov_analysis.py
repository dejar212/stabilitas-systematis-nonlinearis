import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from scipy.signal import find_peaks
import sympy as sp

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.figsize': (10, 6), 'lines.linewidth': 2.0, 'grid.alpha': 0.3, 'axes.grid': True,
})

def derive_model_functions():
    x, th1, th2 = sp.symbols('x theta1 theta2')
    dx, dth1, dth2 = sp.symbols('dx dtheta1 dtheta2')
    u = sp.symbols('u')
    
    m0, m1, m2 = sp.symbols('m0 m1 m2')
    L1, L2 = sp.symbols('L1 L2')
    g = sp.symbols('g')
    
    l1, l2 = L1 / 2, L2 / 2
    I1, I2 = m1 * L1**2 / 12, m2 * L2**2 / 12

    dx1 = dx + l1 * sp.cos(th1) * dth1
    dy1 = -l1 * sp.sin(th1) * dth1
    
    dxj1 = dx + L1 * sp.cos(th1) * dth1
    dyj1 = -L1 * sp.sin(th1) * dth1
    
    dx2 = dxj1 + l2 * sp.cos(th2) * dth2
    dy2 = dyj1 - l2 * sp.sin(th2) * dth2
    
    v1_sq = dx1**2 + dy1**2
    v2_sq = dx2**2 + dy2**2
    
    T = 0.5 * m0 * dx**2 + 0.5 * m1 * v1_sq + 0.5 * I1 * dth1**2 + 0.5 * m2 * v2_sq + 0.5 * I2 * dth2**2
    y1 = l1 * sp.cos(th1)
    y2 = L1 * sp.cos(th1) + l2 * sp.cos(th2)
    V = m1 * g * y1 + m2 * g * y2
    
    L_lag = T - V
    q = sp.Matrix([x, th1, th2])
    dq = sp.Matrix([dx, dth1, dth2])
    
    M = sp.Matrix([[sp.diff(sp.diff(T, dqi), dqj) for dqj in dq] for dqi in dq])
    
    G = sp.Matrix([sp.diff(V, qi) for qi in q])
    
    C = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += 0.5 * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) - sp.diff(M[k, j], q[i])) * dq[k]
    
    B_gen = sp.Matrix([1, 0, 0])
    
    Eq_subs = {x: 0, th1: 0, th2: 0, dx: 0, dth1: 0, dth2: 0}
    M_eq = M.subs(Eq_subs)
    M_eq_inv = M_eq.inv()
    
    dG_dq = G.jacobian(q).subs(Eq_subs)
    dCdq_ddq = (C * dq).jacobian(dq).subs(Eq_subs)
    
    As_mat = M_eq_inv * (-dG_dq)
    Cs_mat = M_eq_inv * (-dCdq_ddq)
    Bs_mat = M_eq_inv * B_gen
    
    params_sym = (m0, m1, m2, L1, L2, g)
    state_sym = (x, th1, th2, dx, dth1, dth2)
    
    M_func = sp.lambdify((state_sym, params_sym), M, 'numpy')
    G_func = sp.lambdify((state_sym, params_sym), G, 'numpy')
    C_func = sp.lambdify((state_sym, params_sym), C, 'numpy')
    
    As_f = sp.lambdify((params_sym,), As_mat, 'numpy')
    Cs_f = sp.lambdify((params_sym,), Cs_mat, 'numpy')
    Bs_f = sp.lambdify((params_sym,), Bs_mat, 'numpy')
    
    V_up = (m1*g*l1 + m2*g*(L1+l2))
    Lyap_func = sp.lambdify((state_sym, params_sym), T + V_up - V, 'numpy')
    
    return M_func, C_func, G_func, As_f, Cs_f, Bs_f, Lyap_func

M_f, C_f, G_f, As_f, Cs_f, Bs_f, Lyap_f = derive_model_functions()

params_nom = {'m0': 1.5, 'm1': 0.5, 'm2': 0.75, 'L1': 0.5, 'L2': 0.75, 'g': 9.81}

def get_lqr_gain(p_dict, Q, R):
    p_t = (p_dict['m0'], p_dict['m1'], p_dict['m2'], p_dict['L1'], p_dict['L2'], p_dict['g'])
    As = np.array(As_f(p_t))
    Cs = np.array(Cs_f(p_t))
    Bs = np.array(Bs_f(p_t))
    
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = As
    A[3:, 3:] = Cs
    B = np.zeros((6, 1))
    B[3:, :] = Bs
    
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P

Q_nom = np.diag([20.0, 200.0, 200.0, 5.0, 20.0, 20.0])
R_nom = np.eye(1) * 0.05
K_lqr = get_lqr_gain(params_nom, Q_nom, R_nom)

def simulate(x0, p_dict, K, T_end=5.0):
    t_eval = np.linspace(0, T_end, 500)
    p_t = (p_dict['m0'], p_dict['m1'], p_dict['m2'], p_dict['L1'], p_dict['L2'], p_dict['g'])
    
    def dyn(y, t):
        u = np.clip((-K @ y)[0], -50, 50)
        M_val = M_f(y, p_t)
        C_val = C_f(y, p_t)
        G_val = G_f(y, p_t)
        
        dq = y[3:]
        B_vec = np.array([1, 0, 0])
        acc = np.linalg.solve(M_val, B_vec * u - C_val @ dq - G_val.flatten())
        return np.concatenate((dq, acc))

    return t_eval, odeint(dyn, x0, t_eval)

angles_deg = [5, 15, 25, 35, 45, 55]
trajs = []
for ang in angles_deg:
    th1_r = np.radians(ang)
    th2_r = th1_r * 0.4
    x0 = np.array([0.0, th1_r, th2_r, 0.0, 0.0, 0.0])
    t, sol = simulate(x0, params_nom, K_lqr, T_end=5.0)
    trajs.append((ang, sol))

plt.figure(figsize=(8, 6))
for ang, sol in trajs:
    is_stable = np.max(np.abs(sol[-50:, 1])) < 0.1
    color = 'g' if is_stable else 'r'
    plt.plot(sol[:, 1], sol[:, 4], color=color, alpha=0.6)
    plt.plot(sol[0, 1], sol[0, 4], 'o', color=color)
plt.title('Phase Portrait: 1st Link'); plt.xlabel('th1'); plt.ylabel('dth1')
plt.savefig('group2_phase_portrait.pdf')

plt.figure(figsize=(8, 6))
for ang, sol in trajs:
    is_stable = np.max(np.abs(sol[-50:, 1])) < 0.1
    color = 'g' if is_stable else 'r'
    plt.plot(sol[:, 2], sol[:, 5], color=color, alpha=0.6)
    plt.plot(sol[0, 2], sol[0, 5], 'o', color=color)
plt.title('Phase Portrait: 2nd Link'); plt.xlabel('th2'); plt.ylabel('dth2')
plt.savefig('group2_phase_th2.pdf')

sol_s = trajs[2][1]
t_s = np.linspace(0, 5.0, 500)
p_t_nom = (params_nom['m0'], params_nom['m1'], params_nom['m2'], params_nom['L1'], params_nom['L2'], params_nom['g'])
V_vals = np.array([Lyap_f(s, p_t_nom) for s in sol_s])
V_dot = np.gradient(V_vals, t_s)
peaks, _ = find_peaks(V_vals)
t_peaks, env = (t_s[peaks], V_vals[peaks]) if len(peaks)>0 else (t_s, V_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(t_s, V_vals, 'b-'); ax1.plot(t_peaks, env, 'r--'); ax1.set_title('Lyapunov V(t)')
ax2.plot(t_s, V_dot, 'g-'); ax2.fill_between(t_s, V_dot, 0, where=V_dot<0, alpha=0.3, color='green')
ax2.set_title('V_dot(t)'); plt.savefig('group2_lyapunov.pdf')

def find_crit_angle(p_dict, Q, R):
    try:
        K = get_lqr_gain(p_dict, Q, R)
    except: return 0.0
    for ang in np.linspace(5, 60, 12):
        th = np.radians(ang)
        x0 = np.array([0, th, th*0.5, 0, 0, 0])
        _, sol = simulate(x0, p_dict, K, T_end=3.0)
        if not (np.abs(sol[-1, 1]) < 0.1 and np.abs(sol[-1, 2]) < 0.1):
            return ang - 5
    return 60.0

vars = np.linspace(0.8, 1.2, 7)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = [r'm_1', r'L_1', r'Q_{weights}']

data_m1 = []
for v in vars:
    p = params_nom.copy(); p['m1'] *= v
    data_m1.append(find_crit_angle(p, Q_nom, R_nom))
axes[0].plot(vars, data_m1, 'o-b')

data_L1 = []
for v in vars:
    p = params_nom.copy(); p['L1'] *= v
    data_L1.append(find_crit_angle(p, Q_nom, R_nom))
axes[1].plot(vars, data_L1, 's-r')

data_Q = []
for v in vars:
    Q_v = Q_nom.copy(); Q_v[1,1] *= v; Q_v[2,2] *= v
    data_Q.append(find_crit_angle(params_nom, Q_v, R_nom))
axes[2].plot(vars, data_Q, '^-g')
for i, ax in enumerate(axes): ax.set_title(f'Crit Angle vs {titles[i]}'); ax.grid(True)
plt.savefig('group2_robustness.pdf')
