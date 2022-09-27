import dash
from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

import base64
import datetime
import io

from Algorithms import decision_tree, train_decision_tree
#from upload_df import parse_contents

#df = pd.read_csv('penguins_size.csv')
#df = df.dropna()
#df = df[df['sex']!='.']


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    # define data frame as global
    global df
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
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





css_sheet = [dbc.themes.SKETCHY]
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = Dash(__name__, external_stylesheets=css_sheet)

app.title = "Machine Learning Classifications"

server = app.server



image_filename = 'dt_tree.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


app.layout = html.Div([

    html.Div([
        dcc.Upload(id='upload-data',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px' },
                   # Allow multiple files to be uploaded
                   multiple=False
                   ),
        #html.Div(id='output-data-upload', hidden=True),
        dbc.Button('Upload File', id='upload_button'),
        html.Br(),
        html.Output(id='file_uploaded'),

    ], style={'background': 'white'}),

    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions', options=[{'label': 'Information', 'value': 'Information'},
                                            {'label': 'Is Null', 'value': 'Is Null'},
                                            {'label': 'Drop NA', 'value': 'Drop NA'},
                                            {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output'),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df', options=[{'label': 'Full   ', 'value': 'Full'},
                                              {'label': 'Head', 'value': 'Head'},
                                              {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
        dbc.Button('Show', id='show_df_button', n_clicks=0, style={'fontSize': '20px'})
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    html.Div(id='df_div',
             children=[
                 dash_table.DataTable(id='dataframe', style_table={'overflowX': 'auto'},
                                      style_cell={'height': 'auto', 'minWidth': '100px', 'width': '100px',
                                                  'maxWidth': '180px', 'whiteSpace': 'normal'},
                                      style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                      style_data={'color': 'black', 'backgroundColor': 'white',
                                                  'border': '2px solid blue'},
                                      filter_action='native'
                                      ),
             ], hidden=True),
    html.Br(),

    html.Div([
        html.Div(html.Label("Select Set of Parameters",
                            style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '25%', 'padding': '20px',
                                   'border': '2px solid black', 'borderRadius': '10px', 'display': 'flex'})),
        html.Br(),

        html.Div([
            html.Label("Enter the Label Column Name:", style={'fontWeight': 'bold', 'paddingRight': '20px'}),
            dcc.Input(id='labels', type='text', style={'width': '200px'}),
        ], style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '50%', 'padding': '20px',
                  'border': '2px solid black', 'borderRadius': '10px', 'display': 'flex'}),
        html.Br(),

        html.Div([
            html.Div([
                html.Label('Criterion', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='criterion', options=[{'label': 'gini', 'value': 'gini'},
                                                      {'label': 'entropy', 'value': 'entropy'},
                                                      #{'label': 'log_loss', 'value': 'log_loss'},
                                                      ],
                             value='gini'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Splitter', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='splitter', options=[{'label': 'best', 'value': 'best'},
                                                     {'label': 'random', 'value': 'random'}, ],
                             value='best'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Max Depth', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='max_depth', type='number', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Sample Split', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_samples_split', type='number', value=2),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Sample Leaf', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_samples_leaf', type='number', value=1),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Weight Fraction Leaf', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_weight_fraction_leaf', type='number', value=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Select Max Feature Method'),
                dcc.Dropdown(id='max_features', options=[{'label': 'auto', 'value': 'auto'},
                                                         {'label': 'sqrt', 'value': 'sqrt'},
                                                         {'label': 'log2', 'value': 'log2'}, ],
                             placeholder='default (None)',
                             value=None),
            ], style={'width': '12%'}),

        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        html.Div([
            html.Div([
                html.Label('Random State', style={'fontWeight': 'bold'}),
                dcc.Input(id='random_state', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Max Leaf Node', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='max_leaf_nodes', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Impurity decrease', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_impurity_decrease', type='number', value=0, placeholder='default 0'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Class Weight', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='class_weight', type='number', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Complexity parameter', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='ccp_alpha', type='number', value=0),
            ], style={'width': '12%'}),

        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        dbc.Button('Run DecisionTree', id='run_dt', n_clicks=0, style={'fontSize': '20px'}),
        html.Br(),
        html.Br(),



    ]),

    html.Div([
        html.Label('Show the Feature Importance', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_feature', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    html.Div(id='df_feature_div',
             children=[
                 dash_table.DataTable(id='df_feature'),
             ], hidden=True),
    html.Br(),


    html.Div([
        html.Label('Show Confusion Matrix Figure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_cm', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),

    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    html.Div(id='show_cm_graph',
             children=[
                 dcc.Graph(id='confusion_matrix'),
             ], hidden=True),
    html.Br(),

    html.Div([
        html.Label('Show Tree Structure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_dt', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    html.Div(id='show_dt_fig',
             children=[
                 html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
             ], hidden=True),
    html.Br(),







    html.Div([
        html.Label('Data Frame with dummy variables'),
        html.Div(id='dummy_features_df',
                 children=[
                     dash_table.DataTable(id='dummy_feature'),
                 ], hidden=False),
    ]),
    html.Br(),


    html.Label('Enter the values of all Features separated by comma'),
    dcc.Input(id='input_features', type='text'),
    dbc.Button('Train Model on All Data', id='train_model', n_clicks=0, style={'fontSize': '20px'}),

    html.Div([html.Label('Equation from Model 2: ', style={'fontWeight': 'bold', 'paddingRight': '10px'}),
              html.Div(id='prediction'), ], style={'display': 'flex'}),

], style={'background': 'Lightgreen'})


@app.callback(
              #Output('output-data-upload', 'children'),
              Output('file_uploaded', 'children'),
              Input('upload_button', 'n_clicks'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)
def update_output(n_clicks, content, filename, date):

    # print(type(df))#this will show data type as a pandas dataframe
    # print(df)

    if filename is not None:
        children = parse_contents(content, filename, date)
        return f'{filename} is Uploaded Succesfully...'
        #return children, f"{filename} File Uploaded Successfully"
    else:
        children = parse_contents(content, filename, date)
        return f'No File is Uploaded...'
        #return children, f"No File is Uploaded"


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
               Output('dataframe', 'columns'), ],
              [Input('show_df_button', 'n_clicks'),
               State('show_df', 'value'), ],
              prevent_initial_call=True)
def show_dataframe(n_clicks, show_df):

    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')
    if show_df == 'Head':
        df_table = df_table[:5]
        return df_table, df_columns
    else:
        return df_table, df_columns



@app.callback([Output('confusion_matrix', 'figure'),
               Output('df_feature', 'data'),
               Output('df_feature', 'columns'),
               Output('dummy_feature', 'data'),
               Output('dummy_feature', 'columns'), ],
              [Input('run_dt', 'n_clicks'),
               State('criterion', 'value'),
               State('splitter', 'value'),
               State('max_depth', 'value'),
               State('min_samples_split', 'value'),
               State('min_samples_leaf', 'value'),
               State('min_weight_fraction_leaf', 'value'),
               State('max_features', 'value'),
               State('random_state', 'value'),
               State('max_leaf_nodes', 'value'),
               State('min_impurity_decrease', 'value'),
               State('class_weight', 'value'),
               State('ccp_alpha', 'value'),
               State('labels', 'value'), ],
              prevent_initial_call=True, )
def update_df(n_clicks, criterion, splitter, max_depth, min_samples_split,
              min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
              min_impurity_decrease, class_weight, ccp_alpha, labels):


    if max_depth == 0:
        max_depth = None

    if labels not in df.columns:
        pass
    else:
        cm_fig, df_feature, dummy_features_df, dummy_features_df_columns\
            = decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                            min_impurity_decrease, class_weight, ccp_alpha, labels)

        df_feature_columns = [{'name': col, 'id': col} for col in df_feature.columns]
        df_feature_table = df_feature.to_dict(orient='records')

        dummy_features_df_columns = [{'name': col, 'id': col} for col in dummy_features_df.columns]
        dummy_features_df_table = dummy_features_df.to_dict(orient='records')

        return cm_fig, df_feature_table, df_feature_columns, dummy_features_df_table, dummy_features_df_columns


@app.callback(Output('prediction', 'children'),
              [Input('train_model', 'n_clicks'),
               State('criterion', 'value'),
               State('splitter', 'value'),
               State('max_depth', 'value'),
               State('min_samples_split', 'value'),
               State('min_samples_leaf', 'value'),
               State('min_weight_fraction_leaf', 'value'),
               State('max_features', 'value'),
               State('random_state', 'value'),
               State('max_leaf_nodes', 'value'),
               State('min_impurity_decrease', 'value'),
               State('class_weight', 'value'),
               State('ccp_alpha', 'value'),
               State('labels', 'value'),
               State('input_features', 'value'), ],
              prevent_initial_call=True, )
def predictions(n_clicks, criterion, splitter, max_depth, min_samples_split,
              min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
              min_impurity_decrease, class_weight, ccp_alpha, labels, input_features):


    if max_depth == 0:
        max_depth = None

    if labels not in df.columns:
        pass
    else:
        prediction = train_decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                                         min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                                         min_impurity_decrease, class_weight, ccp_alpha, labels, input_features)


        return str(prediction[0])










if __name__ == "__main__":
    app.run_server(debug=True)