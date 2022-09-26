import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

app = Dash(__name__)
app.layout = html.Div([

    html.P("Hello World")
])






















if __name__ == "__main__":
    app.run_server(debug=True)