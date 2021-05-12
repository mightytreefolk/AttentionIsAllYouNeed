import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import argparse

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


def plot_two(train, test, title, x_axis, y_axis):
    columns1 = train.columns
    columns2 = test.columns
    y1_data = []
    error_plus1 = []
    error_minus1 = []

    y2_data = []
    error_plus2 = []
    error_minus2 = []
    for c in columns1:
        y = train[c]
        y1_data.append(y.mean())
        error_plus1.append(y.std())
        error_minus1.append(-1 * y.std())

    for c in columns2:
        y = test[c]
        y2_data.append(y.mean())
        error_plus2.append(y.std())
        error_minus2.append(-1 * y.std())

    x_data = np.linspace(0, len(train.columns), len(y1_data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y1_data,
                             mode='lines+markers',
                             name='Train',))

    fig.add_trace(go.Scatter(x=x_data, y=y2_data,
                             mode='lines+markers',
                             name='test'))
    fig.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_acc')
    parser.add_argument('-train_loss')
    parser.add_argument('-test_loss')
    parser.add_argument('-test_acc')
    opt = parser.parse_args()

    train_acc = pd.read_csv(opt.train_acc)
    train_loss = pd.read_csv(opt.train_loss)
    test_acc = pd.read_csv(opt.test_acc)
    test_loss = pd.read_csv(opt.test_loss)


    plot_two(train_acc, test_acc, title='Training Accuracy', x_axis='Epochs', y_axis='Average accuracy')
    plot_two(train_loss, test_loss, title='Training loss', x_axis='Epochs', y_axis='Average loss')


if __name__ == "__main__":
    main()
