from dash import Dash, html, dcc, DiskcacheManager, CeleryManager
import dash
import dash_bootstrap_components as dbc
import os

if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)


css_sheet = [dbc.themes.COSMO]

app = Dash(__name__, use_pages=True, external_stylesheets=css_sheet, background_callback_manager=background_callback_manager)

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
