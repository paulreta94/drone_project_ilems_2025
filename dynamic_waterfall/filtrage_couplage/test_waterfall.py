import numpy as np
import pandas as pd
import plotly.express as px
from functions import rectangular_pulse
# Constants
SoB0 = 10**(5/10)
Pe = 10**((11-30)/10)

# FMCW parameters
fmin = 24.015e9
fmax = 24.235e9
fDoppler = 24.125e9  # Frequency for Doppler measurement

fc = (fmin + fmax) / 2
c = 3e8
lambda_val = c / fc

L = 26 * 1e-3  # Length in meters
S_area = L * L
Ge = 4 * np.pi * S_area / lambda_val**2
Gr = Ge

FFe = 1
RCS = 10**(-37/10)
Le = 10**(-1/10)
Lr = 1
Lrad = 1
Ltrt = 1
Llobe = 1
Attng = 1
FFr = FFe
k = 1.380649e-23  # Boltzmann constant
T = 290  # Temperature in Kelvin
B = fmax - fmin

# Calculate detection distance
D = ((Pe * Ge * Gr * lambda_val**2 * FFe**2 * RCS) /
     ((4 * np.pi)**3 * Le * Lr * Lrad * Ltrt * Llobe * Attng * SoB0 * FFr * k * T * B))**0.25

# Range resolution
deltaD = c / (2 * B)

# Load data
# df = pd.read_json("data.json", lines=True)
df=pd.read_json("data.json",lines=True)
I = np.array(df["I"].dropna().tolist())
Q = np.array(df["Q"].dropna().tolist())
buffer_size = np.size(I[0])

# Velocity resolution
Fs = 320e3
Ti = buffer_size / Fs
deltaV = lambda_val / (2 * Ti)

# Maximum velocity
vmax = buffer_size * deltaV / 2

# Normalization
nbit = 2**12
I = I - nbit / 2
Q = Q - nbit / 2
I = I / (nbit / 2)
Q = Q / (nbit / 2)

# Calculate amplitude
amplitude = np.linspace(-1, 1, num=2**12)

# Calculate S
S = np.conj(I + 1j * Q)

# Apply Hanning window
S = np.multiply(np.tile(np.hanning(len(I[0])), (len(I), 1)), S)

# Compute the inverse FT of S to get a time spectrum
s = np.fft.fftshift(np.fft.ifft(S, axis=1), axes=1)

# Calculate distance array
d = deltaD * (-buffer_size / 2 + np.arange(buffer_size))
sig_couplage = np.multiply(s,rectangular_pulse(d,-3,3))
s_filtered=np.copy(s-sig_couplage)
# Create the waterfall plot
fig = px.imshow(
    np.abs(s_filtered.T),
    aspect="auto",
    y=d,
    zmin=0.01,
    zmax=0.15,
    labels=dict(x="Sample number", y="Distance", color="Amplitude"),
    title="Waterfall Plot of Signal Amplitude"
)



# Set the y-axis range
fig.update_yaxes(range=[-10, 30], autorange=False)

# Show the figure
fig.show()
# import dash
# from dash.dependencies import Input, Output
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.exceptions import PreventUpdate

# from tempfile import gettempdir
# from os import getpid
# from os.path import join as pjoin

# tmpfile= pjoin(gettempdir(), f'time-{getpid()}')

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div([
#     dcc.Input(id='total', value=0),
#     html.Div(id='page-content'),
#     # the layout should update every 500 mili seconds
#     dcc.Interval(interval=500)
# ])

# @app.callback(Output('page-content', 'children'),
#               [Input('total', 'value')])
# def display_page(value):

#     if not value:
#         raise PreventUpdate

#     value = int(value)-1
#     with open(tmpfile, 'w') as f:
#         f.write(str(value))

#     with open(tmpfile) as f:
#         value= int(str(f.read(value)))

#     value+=1

#     print(value)

    
#     return html.Div([
#         html.Plaintext(f'Value: {value}')
#     ])


# if __name__ == '__main__':
#     app.run_server(debug=False)