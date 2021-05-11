import pandas as pd
import plotly.express as px
import numpy as np

def plot(df, title, x_axis, y_axis, opt):
    columns = df.columns
    y_data = []
    error_plus = []
    error_minus = []
    plot_data = pd.DataFrame()
    for c in columns:
        y = df[c]
        y_data.append(y.mean())
        error_plus.append(y.std())
        error_minus.append(-1 * y.std())
    x_data = np.linspace(0, opt.epoch, len(y_data))
    plot_data['y'] = y_data
    plot_data['x'] = x_data
    plot_data['plus_std'] = error_plus
    plot_data['minus_std'] = error_minus
    fig = px.scatter(plot_data, x='x', y='y', error_y="plus_std", error_y_minus="minus_std")
    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=y_axis)
    fig.show()
