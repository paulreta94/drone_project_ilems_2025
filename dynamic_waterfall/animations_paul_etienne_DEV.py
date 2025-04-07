from dash import Dash, dcc, html, Input, Output
import plotly.express as px # type:ignore
import plotly.graph_objects as go# type:ignore
import pandas as pd# type:ignore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import rectangular_pulse
# Constants
SoB0 = 10**(5/10)
Pe = 10**((11-30)/10)
# print(f"Puissance d'émission = {Pe*1e3:.3f} mW")

# FMCW
fmin = 24.015e9
fmax = 24.235e9
fDoppler = 24.125e9  # fréquence CW pour la mesure Doppler

fc = (fmin + fmax) / 2
# print(f"Fréquence centrale = {fc*1e-9:.3f} GHz")

c = 3e8
lambda_val = c / fc
# print(f"Longueur d'onde = {lambda_val*1e3:.3f} mm")

L = 26 * 1e-3  # m
S = L * L
Ge = 4 * np.pi * S / lambda_val**2
Gr = Ge
# print(f"Gain antenne émission = {10*np.log10(Ge):.3f} dBi")

FFe = 1
RCS = 10**(-37/10)
Le = 10**(-1/10)
Lr = 1
Lrad = 1
Ltrt = 1
Llobe = 1
Attng = 1
FFr = FFe
k = 1.380649e-23  # m**2.kg.s^(-2).K^(-1)
T = 290
B = fmax - fmin
# print(f"Bande instantanée = {B*1e-6:.3f} MHz")
F = 10**(2.5/10)

D = ((Pe*Ge*Gr*lambda_val**2*FFe**2*RCS) / ((4*np.pi)**3*Le*Lr*Lrad*Ltrt*Llobe*Attng*SoB0*FFr*k*T*B*F))**0.25
# print(f"Distance de détection = {D:.3f} m")

# Résolution distance
deltaD = c / (2 * B)
# print(f"Résolution distance = {deltaD:.3f} m")

df=pd.read_json('putty.json',lines=True)
I = np.array(df['I'].dropna().to_list())
Q = np.array(df['Q'].dropna().to_list())
buffer_size=np.size(I[0])
# Résolution vitesse
Fs = 320e3
Ti = buffer_size / Fs
# print(f"Durée rampe = {Ti*1e3:.3f} ms")
deltaV = lambda_val / (2 * Ti)
# print(f"Résolution vitesse = {deltaV:.3f} m/s")

# Vitesse maximale (m/s)
vmax = buffer_size * deltaV / 2
# print(f"Vitesse maximale = {vmax:.3f} m/s")

# Normalisation
nbit = 2**12
I = I - nbit / 2#type:ignore
Q = Q - nbit / 2#type:ignore
I = I / (nbit / 2)#type:ignore
Q = Q / (nbit / 2)#type:ignore

# Calcul de l'amplitude
amplitude = np.linspace(-1, 1, num=2**12)

# Calcul de S
S = np.conj(I + 1j*Q)# type:ignore

# Fenêtre de Hanning
# stacking len(I[0])-long hanning vector, len(I) times; then multiplying it term-by-term with S
# hanning vector is a weighted cosine, probably used to improve FFT quality
S = np.multiply(np.tile(np.hanning(len(I[0])), (len(I), 1)), S) # type:ignore

# Computing inverse FT of S to get a time spectrum, then shifting the spectrum at the center (c*t/2 is distance so
# having the spectrum in t/2 is smart)
s = np.fft.fftshift(np.fft.ifft(S, axis=1), axes=1)
f = np.linspace(fmin, fmax, num=buffer_size)
deltaf = f[1] - f[0]
deltat = 1 / deltaf
d = deltaD * (-buffer_size/2 + np.arange(buffer_size))
min_data_to_process = 25
dist = np.zeros((1,min_data_to_process))
# new_s=np.array([s.T[:,:2],s.T[:,2:4],s.T[:,4:6],s.T[:,6:8],s.T[:,8:]]) # slicing the s array (to display it part by 
# part afterwards )
r=rectangular_pulse(d,-3,3)
signal_couplage =(np.abs(s.T[:, 0]) * r).reshape(-1, 1) 

s = s-np.tile(signal_couplage,10).T
# new_s = s.flatten().reshape(5,512,2)
fig=px.imshow(np.abs(signal_couplage),
                          aspect="auto",
                            # y=d,
                          zmin=0.01,
                          zmax=0.15,
                        #   height=400,
                        #   width=1000,
                        #   labels=dict(x="Sample number", y="Distance", color="Amplitude"),
        # animation_frame=0 # new_s has 5 dimensions along axis 0 so we get an animation of 5 frames
        )
# fig.update_yaxes(range=[-10,30],autorange=False)
fig.show()

# Conversion en temps
# t = 2 * d / c
# app = Dash(__name__)


# app.layout = html.Div([ #type:ignore
#     html.H4('Animated GDP and population over decades'),
    # html.P("Select an animation:"),
    # dcc.RadioItems(
    #     id='animations-x-selection',
    #     options=["GDP - Scatter", "Population - Bar"],
    #     value='GDP - Scatter',
    # ),
    # dcc.Loading(dcc.Graph(id="animations-x-graph"), type="cube")
# ])
# @app.callback(
#     Output("animations-x-graph", "figure"),
#     Input(None))
# def display():
#         fig=px.imshow(np.abs(s.reshape(4,512,2)),
#                                 #   aspect="auto",
#                                   y=-d,
#                                 #   zmin=1e-6,
#                                 #   zmax=0.2,
#                                 #   height=400,
#                                 #   width=1000,
#                                 #   labels=dict(x="Sample number", y="Distance", color="Amplitude"),
#                 # x='index',
#                 animation_frame=0
#                 # facet_col=1,
#                 )
#         fig.update_layout(yaxis=dict(range=[-10,30]))
#         return fig

# @app.callback(
#     Output("animations-x-graph", "figure"),
#     # Input("animations-x-selection", "value"))
# def display_animated_graph(selection):
#     animations = {
#         'GDP - Scatter':px.imshow(np.abs(s.reshape(4,512,2)),
#                                 #   aspect="auto",
#                                   y=-d,
#                                 #   zmin=1e-6,
#                                 #   zmax=0.2,
#                                 #   height=400,
#                                 #   width=1000,
#                                 #   labels=dict(x="Sample number", y="Distance", color="Amplitude"),
#                 # x='index',
#                 animation_frame=0
#                 # facet_col=1,
#                 ).update_layout(yaxis=dict(range=[-10,30])),
#                 # ).update_yaxes(range=[-30, 30], autorange=False),
#         # 'Population - Bar': px.bar(
#         'Population - Bar' : px.scatter(
#             df_plot,
#             x='index',
#             # x=df['Q'].dropna().index,
#             # y=df['Q'].tolist,
#             y='Q',
#             # y = np.array(df["Q"].dropna().to_list).flatten(),
#             # x=df['Q'].index,
#             # y=df['Q'].dropna().loc[3],
#             # x="continent", 
#             # y="pop", color="continent",
#             # animation_frame="year",
#             # animation_frame=np.repeat(np.array(df["Q"].dropna().index),512), 
#             animation_frame='index_anim',
#             # animation_group="country",
#             # range_y=[0,4000000000]
#             ),
#     }
#     return animations[selection]


# if __name__ == "__main__":
#     app.run(debug=True)
