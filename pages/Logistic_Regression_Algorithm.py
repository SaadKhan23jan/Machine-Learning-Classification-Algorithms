import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='Logistics Regression'),

    html.Div(children='''
        This Page will be updated in Future
    '''),

])