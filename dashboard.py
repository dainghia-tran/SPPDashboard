import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash_bootstrap_components._components.Container import Container
import dash_bootstrap_components as dbc
import json
import prediction as pred
import plotly.express as px
import asyncio
from binance.client import Client

from datetime import datetime

api_key = 'NwwA7xRmAq0juYxurtfqmAiW7asFDxB33zQknnaOEmEItWnbR0bVVtjoZLV6tQBy'
api_secret = 'Lqq5JzhBANnszyHblQwRhKDgrPjfmjDGdcZA7BZdg3e7pJZkir5vRnBTQh8k4ypp'
apiClient = Client(api_key, api_secret)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

stock_codes = ['AAPL', 'MSFT', 'FB', 'AMZN', 'GOOG', 'TWTR']

prediction_methods = [
    "XGBoost",
    "RNN",
    "LSTM",
]

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

stock_code_dropdown = html.Div(
    children=[
        html.Div(style={"flex": "1", "margin-right": "10px",
                 "text-align": "right"}, children=['Stock Code']),
        dcc.Dropdown(
            options=stock_codes, id="stock_code_dropdown",
            value=stock_codes[0],
            clearable=False,
            searchable=True,
            style={"border-color": "#e3c131", "background-color": "#e3c131",
                   "border-radius": "8px", "color": "black", "width": "100px"},
        )
    ],
    style={"display": "flex", "flex-direction": "row",
           "align-items": "center", "color": "white", "width": "100%", "margin-left": "10px"},
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand(
                            "Stock Price Analysis Dashboard", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                style={"textDecoration": "none"},
            ),
            dbc.Collapse(
                stock_code_dropdown,
                id="navbar-collapse",
                is_open=True,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)

prediction_method_dropdown = html.Div(
    [
        dcc.Dropdown(
            options=prediction_methods, id="prediction_method_dropdown",
            value=prediction_methods[0],
            clearable=False,
            searchable=True,
            style={"border-color": "white", "background": "linear-gradient(90deg, rgba(1,145,148,1) 0%, rgba(153,208,138,1) 100%)",
                   "border-radius": "8px"},
        ),
    ],
    style={"margin": "10px", "width": "150px"},
)


# @app.callback(
#     Output('graph', 'figure'),
#     [
#         Input('stock_code_dropdown', 'value'),
#         Input('prediction_method_dropdown', 'value')
#     ],
# )
# def update_graph(stock_code, method):
#     return updateFigure(stock_code, method)


@app.callback(
    Output('graph', 'figure'),
    Input('graph-update', 'n_intervals')
)
def update_graph_live(n):
    df = pd.DataFrame(apiClient.get_historical_klines(
        "BTCUSDT", Client.KLINE_INTERVAL_1MINUTE))
    df = df.iloc[:, :6]
    df.columns = ["Date", "o", "h", "l", "c", "v"]
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, unit="ms") + pd.Timedelta(hours=7)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'],
            )
        ],
        layout=go.Layout(uirevision='foo')
    )

    return fig


app.layout = html.Div([
    navbar,
    html.Div([
        prediction_method_dropdown,
        dcc.Graph(id="graph"),
        dcc.Interval(
            id='graph-update',
            interval=1000 * 60,
            n_intervals=0
        )
    ])
])


def updateFigure(stock_code, method):
    train, valid, pred_price = pred.get_predicted_price(
        stock_code, method,
    )
    trace1 = go.Scatter(
        name="Train",
        x=train.index,
        y=train['Close'],
    )
    trace2 = go.Scatter(
        name="Valid",
        x=valid.index,
        y=valid['Close'],
    )
    trace3 = go.Scatter(
        name="Predicted",
        x=valid.index,
        y=valid['Predictions'],
    )
    traces = [trace1, trace2, trace3]

    return go.Figure(data=traces, layout=go.Layout(
        title=go.layout.Title(
            text=f"{stock_code} Stock Price Prediction using {method}"),
    ))


app.run_server(debug=True, port='3000')
