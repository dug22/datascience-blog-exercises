import plotly.graph_objects as go
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_values = np.linspace(-10, 10, 400)
y_values = sigmoid(x_values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Sigmoid Curve'))
fig.update_layout(
    title="Sigmoid Function",
    xaxis_title="x",
    yaxis_title="Sigmoid(x)",
    showlegend=True,
    width=1000,
    height=800
)

fig.show()
