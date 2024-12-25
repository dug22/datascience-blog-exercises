import plotly.graph_objects as go
import numpy as np

def relu(x):
    return np.maximum(0, x)

x_values = np.linspace(-10, 10, 400)
y_values = relu(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='ReLU Curve'))
fig.update_layout(
    title="ReLU Function",
    xaxis_title="x",
    yaxis_title="ReLU(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
