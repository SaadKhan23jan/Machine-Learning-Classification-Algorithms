import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='Random Forest Classification'),

    html.Div(children='''
        This Page will be updated in Future
    '''),

])