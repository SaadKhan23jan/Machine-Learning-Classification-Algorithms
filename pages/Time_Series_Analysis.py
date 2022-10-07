import dash
from dash import Dash, dcc, html, dash_table, ctx, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    # define data frame as global
    global df
    global dict_col
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    df = df.dropna()
    dict_col = []
    for col in df.columns:
        dict_col.append({'label': col, 'value': col})


dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='Random Forest Classification'),

    html.Div(children='''
        This Page will be updated in Future
    '''),

])