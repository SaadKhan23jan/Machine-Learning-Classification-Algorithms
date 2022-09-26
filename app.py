import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from Algorithms import decision_tree

df = pd.read_csv('penguins_size.csv')
df = df.dropna()
df = df[df['sex']!='.']

app = Dash(__name__)
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

    dcc.RadioItems(id='show_cm', options=[{'label': 'Yes', 'value': 'Yes'},
                                          {'label': 'No', 'value': 'No'}],
                   value='No'),

    html.Div(id='show_cm_graph',
        children = [
            dcc.Graph(id='confusion_matrix'),
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


@app.callback([Output('show_cm_graph', 'hidden'),
               Output('confusion_matrix', 'figure'), ],
              [Input('show_cm', 'value'), ],
              prevent_initial_call=True)
def cm_graph(show_cm):
    if show_cm == "No":
        return True
    else:
        fig = decision_tree(df)
        return False, fig


@app.callback([Output('dataframe', 'data'),
               Output('dataframe', 'columns'), ],
              [Input('show_df', 'value'), ], )
def update_df(show_df):


    columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')


    return df_table, columns





















if __name__ == "__main__":
    app.run_server(debug=True)