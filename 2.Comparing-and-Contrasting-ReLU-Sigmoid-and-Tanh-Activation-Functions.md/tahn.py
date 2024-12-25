import plotly.graph_objects as go
import numpy as np

def tanh(x):
    return np.tanh(x)

x_values = np.linspace(-10, 10, 400)
y_values = tanh(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Tanh Curve'))
fig.update_layout(
    title="Tanh Function",
    xaxis_title="x",
    yaxis_title="Tanh(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
