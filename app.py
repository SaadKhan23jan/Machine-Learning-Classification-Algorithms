import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from Algorithms import decision_tree
import base64

df = pd.read_csv('penguins_size.csv')
df = df.dropna()
df = df[df['sex']!='.']

css_sheet = [dbc.themes.COSMO]
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = Dash(__name__, external_stylesheets=css_sheet)

app.title = "Machine Learning Classifications"

server = app.server

image_filename = 'dt_tree.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


app.layout = html.Div([

    html.Div([
        dcc.Dropdown(id='df_actions', options=[{'label': 'Information', 'value': 'Information'},
                                            {'label': 'Is Null', 'value': 'Is Null'},
                                            {'label': 'Drop NA', 'value': 'Drop NA'},
                                            {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output'),
    ]),

    html.Div([
        html.Label('Show the DataFrame'),
        dcc.RadioItems(id='show_df', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No'),
    ]),

    html.Div(id='df_div',
        children = [
            dash_table.DataTable(id='dataframe', style_table={'overflowX': 'auto'},
                             style_cell={'height': 'auto', 'minWidth': '100px', 'width': '100px',
                                         'maxWidth': '180px', 'whiteSpace': 'normal'},
                             style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                             style_data={'color': 'black', 'backgroundColor': 'white', 'border': '2px solid blue'},
                             filter_action='native'
                             ),
             ], hidden=True),
    html.Br(),

    html.Div([
        html.Label('Show the Feature Importance DataFrame'),
        dcc.RadioItems(id='show_df_feature', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No'),
    ]),

    html.Div(id='df_feature_div',
             children=[
                 dash_table.DataTable(id='df_feature'),
             ], hidden=True),
    html.Br(),

    dbc.Button('Run DecisionTree', id='run_dt', n_clicks=0),
    html.Br(),

    html.Label('Show Confusion Matrix Figure'),
    dcc.RadioItems(id='show_cm', options=[{'label': 'Yes', 'value': 'Yes'},
                                          {'label': 'No', 'value': 'No'}],
                   value='No'),

    html.Div(id='show_cm_graph',
        children = [
            dcc.Graph(id='confusion_matrix'),
        ], hidden=True),
    html.Br(),

    html.Label('Show Tree Structure'),
    dcc.RadioItems(id='show_dt', options=[{'label': 'Yes', 'value': 'Yes'},
                                          {'label': 'No', 'value': 'No'}],
                   value='No'),
    html.Br(),

    html.Div(id='show_dt_fig',
             children=[
                 html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
             ], hidden=True),

])


@app.callback(Output('df_div', 'hidden'),
              Input('show_df', 'value'),
              prevent_initial_call=True)
def df_div(show_df):
    if show_df == "No":
        return True
    else:
        return False


@app.callback(Output('df_feature_div', 'hidden'),
              Input('show_df_feature', 'value'),
              prevent_initial_call=True)
def df_div(show_df_feature):
    if show_df_feature == "No":
        return True
    else:
        return False


@app.callback(Output('show_cm_graph', 'hidden'),
              Input('show_cm', 'value'),
              prevent_initial_call=True)
def cm_graph(show_cm):
    if show_cm == "Yes":
        return False
    else:
        return True

@app.callback(Output('show_dt_fig', 'hidden'),
              Input('show_dt', 'value'),
              prevent_initial_call=True)
def cm_graph(show_cm):
    if show_cm == "Yes":
        return False
    else:
        return True


@app.callback([Output('dataframe', 'data'),
               Output('dataframe', 'columns'),
               Output('confusion_matrix', 'figure'),
               Output('df_feature', 'data'),
               Output('df_feature', 'columns'), ],
              [Input('run_dt', 'n_clicks'),
               State('show_df', 'value'), ], )
def update_df(n_clicks, show_df):


    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')

    cm_fig, df_feature = decision_tree(df)

    df_feature_columns = [{'name': col, 'id': col} for col in df_feature.columns]
    df_feature_table = df_feature.to_dict(orient='records')


    return df_table, df_columns, cm_fig, df_feature_table, df_feature_columns





















if __name__ == "__main__":
    app.run_server(debug=True)