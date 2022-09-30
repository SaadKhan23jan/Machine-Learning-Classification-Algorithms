from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

css_sheet = [dbc.themes.COSMO]

app = Dash(__name__, use_pages=True, external_stylesheets=css_sheet)

app.title = "Machine Learning Classifications"

server = app.server

app.layout = html.Div([
    html.Div(f"Data Visualization and Machine Learning",
             style={'backgroundColor': 'Lightgreen', 'fontSize': 50, 'textAlign': 'center'}),
    html.Div([
        dcc.Link(page['name']+" | ", href=page['path'])
        for page in dash.page_registry.values()
        ], style={'fontSize': '20px'}),
    html.Hr(),

    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)
