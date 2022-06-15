import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash_bootstrap_components._components.Container import Container
import dash_bootstrap_components as dbc
import asyncio
import websockets
import csv
import json
import prediction as pred

from datetime import datetime

app = Dash(__name__)

stock_codes = ['AAPL', 'MSFT', 'FB', 'AMZN', 'GOOG', 'TWTR']
selected_stock_code = stock_codes[0]

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


@app.callback(
    [Output('stock_code_dropdown', 'value'),
     Output('stock_code_dropdown', 'value')],
    Input('stock_code_dropdown', 'value'),
)
def update_stock_code(code):
    selected_stock_code = code

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)
    return code, fig


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
                href="https://plotly.com",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                stock_code_dropdown,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


train, valid, pred_price = pred.get_predicted_price(code=selected_stock_code)
# final_data = []
# for i in range(len(train)):
#     final_data.append({"Date": train.index[i], "train": train[i], "valid": valid[i], "pred": pred_price[i]})
fig = go.Figure(
    [go.Scatter(y=[valid[['Close', 'Predictions']]], mode='lines')])


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

prediction_methods = [
    "XGBoost",
    "RNN",
    "LSTM",
]

dbc.DropdownMenu()

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


@app.callback(
    Output('prediction_method_dropdown', 'value'),
    Input('prediction_method_dropdown', 'value'),
)
def choose_prediction_method(method):
    return method


app.layout = html.Div([
    navbar,
    html.Div([
        prediction_method_dropdown,
        dcc.Graph(id="graph", figure=fig),
    ])
])


app.run_server(debug=True, port='3000')
