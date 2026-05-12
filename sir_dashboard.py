import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="Modelo SIR — COVID-19 Colombia",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f4f6f9; }
.block-container { padding: 1.2rem 2rem 1rem 2rem; }

/* Header */
.header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A5C6E 100%);
    padding: 1.4rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
}
.header h1 { color: #ffffff; font-size: 1.7rem; margin: 0; font-weight: 700; }
.header p  { color: #94D2BD; margin: 0.25rem 0 0 0; font-size: 0.92rem; }

/* Metric cards */
.metric-big {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    height: 100%;
}
.metric-big .value { font-size: 2.2rem; font-weight: 700; margin: 0; line-height: 1.1; }
.metric-big .label { font-size: 0.78rem; color: #6c757d; margin: 0.3rem 0 0 0; }

/* R0 semáforo */
.r0-box {
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.r0-box .r0-val { font-size: 3rem; font-weight: 700; margin: 0; line-height: 1; }
.r0-box .r0-label { font-size: 0.82rem; margin: 0.3rem 0 0 0; font-weight: 600; }

/* Escenario pills */
.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.pill-red   { background:#ffd6d6; color:#7a0000; }
.pill-amber { background:#fff3cd; color:#7a4000; }
.pill-green { background:#d4edda; color:#155724; }

/* Slider labels */
.slider-label { font-size: 0.85rem; font-weight: 600; color: #0D1B2A; margin-bottom: 2px; }
.slider-hint  { font-size: 0.75rem; color: #6c757d; margin-bottom: 6px; }

/* Section headers */
.sec-header {
    font-size: 0.92rem; font-weight: 700; color: #0D1B2A;
    border-left: 4px solid #0A9396;
    padding-left: 8px; margin: 0.8rem 0 0.6rem 0;
}

/* Scenario selector buttons */
div[data-testid="stHorizontalBlock"] .stButton button {
    width: 100%; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── DATOS REALES COLOMBIA 2020 ────────────────────────────────────────────────
raw = """t,I_real,R_real,S_real
0,1,0,51874023
14,478,0,51873545
28,1944,133,51871923
42,4810,277,51869213
56,12846,848,51861306
70,28641,2108,51843251
84,43978,3470,51826552
98,68640,6853,51805882
112,102824,8777,51769900
126,114113,13829,51748058
140,109353,17025,51750471
154,65500,23443,51806207
168,28641,25103,51843306
182,206300,25969,51641731
196,206300,26936,51641631
203,206300,28294,51639406
210,185000,29685,51659315
224,150000,30698,51693302
238,120000,31962,51722038
252,90000,33090,51750910
266,65000,34306,51774694
280,45000,35750,51793250
294,30000,37925,51806075"""
df_real = pd.read_csv(StringIO(raw))

# ── RK4 ──────────────────────────────────────────────────────────────────────
def sir_rk4(S0, I0, R0, beta, gamma, N, dias,
            cuarentena=False, dia_cuarentena=25, beta_post=None, h=1.0):
    pasos = int(dias / h)
    t = np.zeros(pasos + 1)
    S = np.zeros(pasos + 1)
    I = np.zeros(pasos + 1)
    R = np.zeros(pasos + 1)
    t[0], S[0], I[0], R[0] = 0, S0, I0, R0

    def derivs(S_, I_, R_, b):
        dS = -b * S_ * I_ / N
        dI =  b * S_ * I_ / N - gamma * I_
        dR =  gamma * I_
        return dS, dI, dR

    for n in range(pasos):
        b = beta
        if cuarentena and t[n] >= dia_cuarentena:
            b = beta_post if beta_post else beta * 0.30
        Sn, In, Rn = S[n], I[n], R[n]
        k1s,k1i,k1r = derivs(Sn,In,Rn,b)
        k2s,k2i,k2r = derivs(Sn+h*k1s/2,In+h*k1i/2,Rn+h*k1r/2,b)
        k3s,k3i,k3r = derivs(Sn+h*k2s/2,In+h*k2i/2,Rn+h*k2r/2,b)
        k4s,k4i,k4r = derivs(Sn+h*k3s,In+h*k3i,Rn+h*k3r,b)
        S[n+1] = Sn + (h/6)*(k1s+2*k2s+2*k3s+k4s)
        I[n+1] = In + (h/6)*(k1i+2*k2i+2*k3i+k4i)
        R[n+1] = Rn + (h/6)*(k1r+2*k2r+2*k3r+k4r)
        t[n+1] = t[n] + h
    return t, S, I, R

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header">
    <h1>🦠 Modelo SIR — COVID-19 Colombia 2020</h1>
    <p>Solución numérica Runge-Kutta 4 · Kermack & McKendrick (1927) · UTADEO — Ecuaciones Diferenciales</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT: controles izquierda | gráfica derecha
# ══════════════════════════════════════════════════════════════════════════════
col_ctrl, col_plot = st.columns([1, 2.6])

with col_ctrl:

    # ── Semáforo R0 (placeholder, se actualiza abajo) ─────────────────────────
    r0_placeholder = st.empty()

    st.markdown('<div class="sec-header">⚙️ Parámetros</div>', unsafe_allow_html=True)

    N = 51_874_024

    # Escenarios rápidos
    st.markdown("**Escenarios predefinidos**")
    esc1, esc2, esc3 = st.columns(3)
    escenario = None
    with esc1:
        if st.button("🔴 Sin NPIs", use_container_width=True):
            escenario = "sin"
    with esc2:
        if st.button("🟡 Parcial", use_container_width=True):
            escenario = "parcial"
    with esc3:
        if st.button("🟢 Cuarentena", use_container_width=True):
            escenario = "cuarentena"

    # Defaults por escenario
    defaults = {
        "sin":       dict(beta=0.30, gamma=1/14, dias=160, cuarentena=False),
        "parcial":   dict(beta=0.18, gamma=1/14, dias=250, cuarentena=False),
        "cuarentena":dict(beta=0.30, gamma=1/14, dias=294, cuarentena=True),
        None:        dict(beta=0.30, gamma=1/14, dias=294, cuarentena=False),
    }
    def_vals = defaults[escenario]

    st.markdown('<p class="slider-label">β — Tasa de transmisión</p>'
                '<p class="slider-hint">Sin intervención ≈ 0.30 · Con cuarentena ≈ 0.09</p>',
                unsafe_allow_html=True)
    beta = st.slider("β", 0.05, 0.80, def_vals["beta"], 0.01,
                     label_visibility="collapsed")

    st.markdown('<p class="slider-label">γ — Tasa de recuperación</p>'
                '<p class="slider-hint">COVID-19: 1/14 ≈ 0.071 día⁻¹ (OMS)</p>',
                unsafe_allow_html=True)
    gamma = st.slider("γ", 0.02, 0.20, def_vals["gamma"], 0.001,
                      format="%.3f", label_visibility="collapsed")

    dias = st.slider("📅 Días a simular", 60, 365, def_vals["dias"], 5)

    # Cuarentena toggle
    st.markdown('<div class="sec-header">🔒 Simulación de cuarentena</div>',
                unsafe_allow_html=True)
    cuarentena_on = st.toggle("Activar cuarentena (día 25)", value=def_vals["cuarentena"])
    if cuarentena_on:
        reduccion = st.slider("Reducción de β tras cuarentena (%)", 30, 90, 70, 5)
        beta_post = beta * (1 - reduccion/100)
        st.caption(f"β efectivo post-cuarentena = {beta_post:.3f}")
    else:
        beta_post = None

    # ── Mostrar datos reales toggle ───────────────────────────────────────────
    st.markdown('<div class="sec-header">📊 Datos reales</div>',
                unsafe_allow_html=True)
    mostrar_real = st.toggle("Superponer datos reales Colombia 2020", value=True)

# ── CALCULAR ──────────────────────────────────────────────────────────────────
I0 = 1
S0 = N - I0
R0_init = 0

t_arr, S_arr, I_arr, R_arr = sir_rk4(
    S0, I0, R0_init, beta, gamma, N, dias,
    cuarentena=cuarentena_on, beta_post=beta_post
)

R0_val = beta / gamma
idx_pico = np.argmax(I_arr)
dia_pico = t_arr[idx_pico]
pico_I   = I_arr[idx_pico]
pct_prot = S_arr[-1] / N * 100

# ── Semáforo R0 ───────────────────────────────────────────────────────────────
if R0_val >= 2.5:
    r0_color = "#dc3545"; r0_bg = "#fff0f0"; r0_txt = "⚠️ Epidemia explosiva"
elif R0_val >= 1:
    r0_color = "#fd7e14"; r0_bg = "#fff8f0"; r0_txt = "⚠️ Epidemia crece"
else:
    r0_color = "#28a745"; r0_bg = "#f0fff4"; r0_txt = "✅ Epidemia se extingue"

with col_ctrl:
    r0_placeholder.markdown(f"""
    <div class="r0-box" style="background:{r0_bg}; border: 2px solid {r0_color}; margin-bottom:12px">
        <p style="font-size:0.75rem;color:#6c757d;margin:0">Número reproductivo básico</p>
        <p class="r0-val" style="color:{r0_color}">R₀ = {R0_val:.2f}</p>
        <p class="r0-label" style="color:{r0_color}">{r0_txt}</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL DERECHO
# ══════════════════════════════════════════════════════════════════════════════
with col_plot:

    # ── Métricas grandes ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-big">
            <p class="value" style="color:#0A9396">Día {int(dia_pico)}</p>
            <p class="label">📅 Día del pico de infectados</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        label_pico = f"{pico_I/1e6:.2f}M" if pico_I >= 1e6 else f"{pico_I/1e3:.0f}k"
        st.markdown(f"""<div class="metric-big">
            <p class="value" style="color:#dc3545">{label_pico}</p>
            <p class="label">🦠 Infectados simultáneos en el pico</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-big">
            <p class="value" style="color:#28a745">{pct_prot:.1f}%</p>
            <p class="label">🛡️ Población nunca infectada</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Gráfica ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor='white',
                             gridspec_kw={'wspace': 0.32})

    COLOR_S = '#0D1B2A'
    COLOR_I = '#dc3545'
    COLOR_R = '#1a7a4a'
    COLOR_REAL_I = '#f4a261'
    COLOR_REAL_R = '#52b788'

    # ── Panel izquierdo: S(t) ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1.plot(t_arr, S_arr/1e6, color=COLOR_S, lw=2.5, label='S(t) modelo')
    ax1.fill_between(t_arr, S_arr/1e6, alpha=0.08, color=COLOR_S)
    if mostrar_real and not cuarentena_on:
        mask = df_real['t'] <= dias
        ax1.scatter(df_real[mask]['t'], df_real[mask]['S_real']/1e6,
                    s=22, color=COLOR_S, alpha=0.5, zorder=5,
                    label='S real Colombia')
    if cuarentena_on:
        ax1.axvline(25, color='orange', lw=1.5, ls='--', alpha=0.7, label='Día cuarentena')

    ax1.set_title('Susceptibles S(t)', fontsize=11, fontweight='bold', color='#0D1B2A')
    ax1.set_xlabel('Días', fontsize=10)
    ax1.set_ylabel('Personas (millones)', fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}M'))
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25, ls='--')
    ax1.spines[['top','right']].set_visible(False)

    # ── Panel derecho: I(t) y R(t) ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.plot(t_arr, I_arr/1e3, color=COLOR_I, lw=2.5, label='I(t) modelo')
    ax2.plot(t_arr, R_arr/1e3, color=COLOR_R, lw=2.5, label='R(t) modelo')
    ax2.fill_between(t_arr, I_arr/1e3, alpha=0.12, color=COLOR_I)
    ax2.fill_between(t_arr, R_arr/1e3, alpha=0.10, color=COLOR_R)

    if mostrar_real and not cuarentena_on:
        mask = df_real['t'] <= dias
        ax2.scatter(df_real[mask]['t'], df_real[mask]['I_real']/1e3,
                    s=22, color=COLOR_REAL_I, zorder=5, label='I real Colombia')
        ax2.scatter(df_real[mask]['t'], df_real[mask]['R_real']/1e3,
                    s=22, color=COLOR_REAL_R, zorder=5, label='R real Colombia')

    # Línea pico
    ax2.axvline(dia_pico, color=COLOR_I, lw=1.2, ls='--', alpha=0.5)
    label_pico2 = f"{pico_I/1e6:.1f}M" if pico_I >= 1e6 else f"{pico_I/1e3:.0f}k"
    ax2.annotate(f'Pico día {int(dia_pico)}\n{label_pico2}',
                 xy=(dia_pico, pico_I/1e3),
                 xytext=(min(dia_pico + 15, dias - 30), pico_I/1e3 * 0.80),
                 fontsize=8, color=COLOR_I, style='italic',
                 arrowprops=dict(arrowstyle='->', color=COLOR_I, lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLOR_I, alpha=0.9))

    if cuarentena_on:
        ax2.axvline(25, color='orange', lw=1.5, ls='--', alpha=0.7, label='Día cuarentena')

    ax2.set_title('Infectados I(t) y Recuperados R(t)', fontsize=11,
                  fontweight='bold', color='#0D1B2A')
    ax2.set_xlabel('Días', fontsize=10)
    ax2.set_ylabel('Personas (miles)', fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}k'))
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.25, ls='--')
    ax2.spines[['top','right']].set_visible(False)

    # Supertítulo
    escenario_label = ""
    if cuarentena_on:
        escenario_label = f" | 🔒 Cuarentena día 25 (β→{beta_post:.3f})"
    fig.suptitle(
        f"β = {beta:.2f}  ·  γ = {gamma:.3f}  ·  R₀ = {R0_val:.2f}{escenario_label}",
        fontsize=10, color='#555', y=1.01
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Nota metodológica ─────────────────────────────────────────────────
    st.caption(
        "★ Datos reales: INS Colombia / JHU CSSE · "
        "R estimado como C(t−14d) − D(t) · "
        "Método numérico: Runge-Kutta orden 4, h = 1 día · "
        "N = 51.874.024 (DANE 2021)"
    )

    # ── Gráfica datos reales Colombia 2020 ────────────────────────────────
    st.markdown('<div class="sec-header">📊 Comportamiento real — Colombia 2020</div>',
                unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.4), facecolor='white',
                               gridspec_kw={'wspace': 0.32})

    # Panel izquierdo real: S_real
    ax_r1 = axes2[0]
    ax_r1.set_facecolor('#f8f9fa')
    ax_r1.plot(df_real['t'], df_real['S_real']/1e6, color=COLOR_S, lw=2.5,
               marker='o', markersize=5, label='S real Colombia')
    ax_r1.fill_between(df_real['t'], df_real['S_real']/1e6, alpha=0.08, color=COLOR_S)
    ax_r1.set_title('Susceptibles reales S(t)', fontsize=11, fontweight='bold', color='#0D1B2A')
    ax_r1.set_xlabel('Días desde inicio pandemia', fontsize=10)
    ax_r1.set_ylabel('Personas (millones)', fontsize=10)
    ax_r1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}M'))
    ax_r1.legend(fontsize=8, framealpha=0.9)
    ax_r1.grid(True, alpha=0.25, ls='--')
    ax_r1.spines[['top', 'right']].set_visible(False)

    # Panel derecho real: I_real y R_real
    ax_r2 = axes2[1]
    ax_r2.set_facecolor('#f8f9fa')
    ax_r2.plot(df_real['t'], df_real['I_real']/1e3, color=COLOR_REAL_I, lw=2.5,
               marker='o', markersize=5, label='I real Colombia')
    ax_r2.plot(df_real['t'], df_real['R_real']/1e3, color=COLOR_REAL_R, lw=2.5,
               marker='s', markersize=5, label='R real Colombia')
    ax_r2.fill_between(df_real['t'], df_real['I_real']/1e3, alpha=0.15, color=COLOR_REAL_I)
    ax_r2.fill_between(df_real['t'], df_real['R_real']/1e3, alpha=0.12, color=COLOR_REAL_R)

    idx_pico_real = df_real['I_real'].idxmax()
    dia_pico_real = int(df_real.loc[idx_pico_real, 't'])
    val_pico_real = df_real.loc[idx_pico_real, 'I_real']
    ax_r2.axvline(dia_pico_real, color=COLOR_REAL_I, lw=1.2, ls='--', alpha=0.6)
    ax_r2.annotate(f'Pico día {dia_pico_real}\n{val_pico_real/1e3:.0f}k',
                   xy=(dia_pico_real, val_pico_real/1e3),
                   xytext=(dia_pico_real + 12, val_pico_real/1e3 * 0.78),
                   fontsize=8, color=COLOR_REAL_I, style='italic',
                   arrowprops=dict(arrowstyle='->', color=COLOR_REAL_I, lw=1),
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLOR_REAL_I, alpha=0.9))

    ax_r2.set_title('Infectados y Recuperados reales I(t), R(t)', fontsize=11,
                    fontweight='bold', color='#0D1B2A')
    ax_r2.set_xlabel('Días desde inicio pandemia', fontsize=10)
    ax_r2.set_ylabel('Personas (miles)', fontsize=10)
    ax_r2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}k'))
    ax_r2.legend(fontsize=8, framealpha=0.9)
    ax_r2.grid(True, alpha=0.25, ls='--')
    ax_r2.spines[['top', 'right']].set_visible(False)

    fig2.suptitle('Datos reales INS Colombia / JHU CSSE · 2020',
                  fontsize=10, color='#555', y=1.01)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PIE — ecuaciones compactas
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
e1, e2, e3, e4 = st.columns(4)
with e1:
    st.markdown("**dS/dt** = −β · S · I / N")
with e2:
    st.markdown("**dI/dt** = β · S · I / N − γ · I")
with e3:
    st.markdown("**dR/dt** = γ · I")
with e4:
    st.markdown(f"**N** = S + I + R = {N:,} (constante)")