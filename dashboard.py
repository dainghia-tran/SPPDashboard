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
    "LSTM",
    "RNN",
    "XGBoost",
]

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

stock_code_dropdown = html.Div(
    children=[
        dcc.Dropdown(
            options=stock_codes, id="stock_code_dropdown",
            value=stock_codes[0],
            clearable=False,
            searchable=True,
            style={"border-color": "#e3c131", "background-color": "#e3c131",
                   "border-radius": "8px", "color": "black", "width": "100px"},
        )
    ],
)

prediction_method_dropdown = html.Div(
    [
        html.Div(style={"flex": "1", "margin-right": "10px",
                 "text-align": "right"}, children=['Method']),
        dcc.Dropdown(
            options=prediction_methods, id="prediction_method_dropdown",
            value=prediction_methods[0],
            clearable=False,
            searchable=True,
            style={"background": "linear-gradient(90deg, rgba(1,145,148,1) 0%, rgba(153,208,138,1) 100%)",
                   "border-radius": "8px", "width": "150px", "color": "black"},
        ),
    ],
    style={"display": "flex", "flex-direction": "row", "flex": "1",
           "align-items": "center", "color": "white", "margin-left": "10px", },
)

features = html.Div(
    [
        dbc.RadioItems(
            id="feature-radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Close", "value": 1},
                {"label": "Price of change", "value": 2},
            ],
            value=1,
        ),
    ],
    className="radio-group",
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
                prediction_method_dropdown,
                id="navbar-collapse",
                is_open=True,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)


@app.callback(
    [
        Output('graph-pred', 'figure'),
        Output("loading-output-pred", "children")
    ],
    [
        Input('stock_code_dropdown', 'value'),
        Input('prediction_method_dropdown', 'value'),
        Input('feature-radios', 'value')
    ],
)
def update_graph(stock_code, method, feature):
    return updateFigure(stock_code, method, feature), ''


@app.callback(
    Output('graph-btc', 'figure'),
    Input('graph-btc-update', 'n_intervals')
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
        dcc.Tabs(
            id='tabs',
            children=[
                dcc.Tab(
                    label='Prediction',
                    children=[
                        html.Div([
                            html.Div(
                                children=[
                                    stock_code_dropdown,
                                    features
                                ],
                                style={"display": "flex",
                                       "flex-direction": "row",
                                       "justify-content": "space-between",
                                       "margin": "16px"}
                            ),
                            html.Div(
                                [
                                    dcc.Loading(
                                        id="loading-pred",
                                        type="default",
                                        children=html.Div(
                                            id="loading-output-pred"),
                                    ),
                                    dcc.Graph(id="graph-pred"),
                                ],
                                style={"position": "relative"}
                            ),
                            dcc.Interval(
                                id='graph-pred-update',
                                interval=1000 * 60,
                                n_intervals=0
                            )
                        ])
                    ]
                ),
                dcc.Tab(
                    label='Price',
                    children=[
                        html.Div([
                            html.H2(
                                "BTC/USDT price (update per mintute)",
                                style={"textAlign": "center", 'margin': "16px"}
                            ),
                            dcc.Graph(id="graph-btc"),
                            dcc.Interval(
                                id='graph-btc-update',
                                interval=1000 * 60,
                                n_intervals=0
                            )
                        ])
                    ]
                )
            ]
        )
    ]
    )
])


def updateFigure(stock_code, method, feature):
    train, actual, predicted = pred.get_predicted_price(
        stock_code, method, feature
    )

    trainTrace = go.Scatter(
        name="Train",
        x=train.index,
        y=train['Close'],
    )
    actualTrace = go.Scatter(
        name="Actual",
        x=actual.index,
        y=actual['Close'],
        line=dict(color='#f59e42')
    )
    predictedTrace = go.Scatter(
        name="Predicted",
        x=predicted.index,
        y=predicted["Close"],
    )
    traces = [trainTrace, actualTrace, predictedTrace]

    return go.Figure(data=traces, layout=go.Layout(
        title=go.layout.Title(
            text=f"{stock_code} Stock Price Prediction using {method} method",
        ),
    ))


app.run_server(debug=True, port='3000')
