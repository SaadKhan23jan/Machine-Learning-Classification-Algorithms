import dash
from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate

import base64
import datetime
import io

from Algorithms import decision_tree, train_decision_tree, get_dummy_variables

# from upload_df import parse_contents
# The following code was for df from local file on this machine
# df = pd.read_csv('penguins_size.csv')
# df = df.dropna()
# df = df[df['sex']!='.']

# This function create Pandas DataFrame from the uploaded file and make it global


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


css_sheet = [dbc.themes.COSMO]
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = Dash(__name__, external_stylesheets=css_sheet)

app.title = "Machine Learning Classifications"

server = app.server


# Here we will upload the image created from DecisionTree
image_filename = 'dt_tree.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


app.layout = html.Div([

    # This div container is for uploading file (uploaded when clicked the "Upload File" button
    html.Div([
        dcc.Upload(id='upload-data',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px' },
                   # Do not allow multiple files to upload
                   multiple=False
                   ),
        # The below line is for if we want to show the uploaded file as Data Table
        # html.Div(id='output-data-upload', hidden=True),
        dbc.Button('Upload File', id='upload_button'),
        html.Br(),
        html.Output(id='file_uploaded'),

    ], style={'background': 'white'}),
    html.Br(),

    # Thi div container is for taking action on the upload file for example, df.info, df.describe, df.isna, df.dropna
    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions', options=[{'label': 'Information', 'value': 'Information'},
                                            {'label': 'Is Null', 'value': 'Is Null'},
                                            {'label': 'Drop NA', 'value': 'Drop NA'},
                                            {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output'),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    # This div container is for options of hiding Data Frame, showing only head with 5 row on full Data Frame
    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df', options=[{'label': 'Full   ', 'value': 'Full'},
                                              {'label': 'Head', 'value': 'Head'},
                                              {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
        dbc.Button('Show', id='show_df_button', n_clicks=0, style={'fontSize': '20px'})
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    # This div container is for the results of obove options to print the Data Frame as per the chosen option
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

    # This Div has many divs inside and is for selecting variety of parameters for training the model
    html.Div([

        # This is just one div for printing the label "Select Set of Parameters"
        html.Div(html.Label("Select Set of Parameters",
                            style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '50%', 'padding': '20px',
                                   'border': '2px solid black', 'borderRadius': '10px', 'display': 'flex'})),
        html.Br(),

        # Thi div is for selecting a Label, the Label list is created from the columns of the Data Frame
        # This list is generated when the button is clicked to generate, so it is connected to a callback
        html.Div([
            html.Label("Select the Label to Predict:",
                       style={'width': '50%', 'fontWeight': 'bold', 'paddingRight': '20px'}),
            dcc.Dropdown(id='df_columns_dropdown_label', style={'width': '80%'}),
            html.Button('click to generate dropdown options', id='gen_dropdown',
                        style={'borderRadius': '20px', 'backgroundColor': ''}),
        ], style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '50%', 'padding': '20px',
                  'border': '2px solid black', 'borderRadius': '10px', 'display': 'flex'}),
        html.Br(),

        # This div has one div inside for each parameter
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
                dcc.Input(id='max_depth', type='number', value=None, placeholder='default (None)', min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Sample Split', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_samples_split', type='number', value=2, min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Sample Leaf', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_samples_leaf', type='number', value=1, min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Min Weight Fraction Leaf', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='min_weight_fraction_leaf', type='number', value=0, min=0, max=0.5, step=0.001),
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

        # This div has one div inside for each parameter in addition to the above
        html.Div([
            html.Div([
                html.Label('Random State', style={'fontWeight': 'bold'}),
                dcc.Input(id='random_state', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Max Leaf Node', style={'fontWeight': 'bold'}),
                dcc.Input(id='max_leaf_nodes', value=None, placeholder='default (None)'),
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

        # This button runs the calculations, so it is connected to a callback and runs the calculations
        dbc.Button('Run DecisionTree', id='run_dt', n_clicks=0, style={'fontSize': '20px'}),
        html.Br(),
        html.Br(),

    ]),

    # This div has radio button, if Yes, then it will show Data Frame of feature importance
    html.Div([
        html.Label('Show the Feature Importance', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_feature', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned Data Frame from the Decision model and also connected to callback from the above div Yes/No
    html.Div(id='df_feature_div',
             children=[
                 dash_table.DataTable(id='df_feature'),
             ], hidden=True),
    html.Br(),

    # This is almost the same div for confusion matrix as above for feature importance
    html.Div([
        html.Label('Show Confusion Matrix Figure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_cm', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),

    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned confusion matrix from model and the function heatmap_plot_confusion_matrix
    html.Div(id='show_cm_graph',
             children=[
                 dcc.Graph(id='confusion_matrix'),
             ], hidden=True),
    html.Br(),

    # Same radio buttons as above for Feature Importance Data Frame and Confusion mMatrix
    html.Div([
        html.Label('Show Tree Structure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_dt', options=[{'label': 'Yes', 'value': 'Yes'},
                                              {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This div has the plot for Confusion Matrix
    html.Div(id='show_dt_fig',
             children=[
                 html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
             ], hidden=True),
    html.Br(),

    # This table appears once the model is trained, it is editable
    # The values from these cells are converted into list in the call back and input into the model.predict
    html.Div([
        html.Label('Enter These variables values for Prediction', style={'fontSize': '20px', 'fontWeight': 'bold'}),
        dash_table.DataTable(id='dummy_feature', editable=True),
                 ], style={'backgroundColor': 'Lightblue'}),
    html.Br(),


    html.Label('Enter the values of all Features (as above dummy variables) separated by single white space',
               style={'fontSize': '20px', 'fontWeight': 'bold'}),
    html.Br(),
    dcc.Input(id='input_features', type='text'),
    html.Br(),
    html.Br(),
    dbc.Button('Train Model on All Data', id='train_model', n_clicks=0,
               style={'fontSize': '20px', 'fontWeight': 'bold'}),
    html.Br(),
    html.Div(),

    html.Div([html.Label('Prediction with the Selected Parameters set: ',
                         style={'fontSize': '20px', 'fontWeight': 'bold', 'paddingRight': '10px'}),
              html.Div([
                  html.Label("Prediction Label:"),
                  html.Div(id='target_label', style={'paddingRight': '20px', 'paddingLeft': '20px'}),
                  html.Label("Predicted Value:", style={'paddingRight': '20px'}),
                  html.Div(id='prediction'),
              ], style={'display': 'flex', 'border': '2px solid black', 'borderRadius': '10px'}),
              html.Br(),
              html.Br(),],
             style={'width': '50%', 'background': '#f3f2f5', 'fontSize': '20px', 'fontWeight': 'bold',
                    'paddingRight': '10px', 'paddingLeft': '200px', 'borderRadius': '50px'}),

    html.Div(),
    html.Div(),
    html.Div(),

    html.Button("CLick Here to generate input table", id="gen_inputs"),
    dash_table.DataTable(
        id='feature_input_table',
        editable=True

    )

], style={'background': 'Lightgreen'})


@app.callback(Output("feature_input_table", "columns"),
              Output("feature_input_table", "data"),
              [Input("gen_inputs", "n_clicks"),
               State('df_columns_dropdown_label', 'value'), ],
              prevent_initial_call=True)
def generate_inputs(n_clicks, df_columns_dropdown_label):

    columns, data = get_dummy_variables(df, df_columns_dropdown_label)
    # data = [{'input-data': i} for i in range(len(columns))]
    raise PreventUpdate

    return columns, data



# This is for Uploading the csv file
@app.callback(
              #Output('output-data-upload', 'children'),
              Output('file_uploaded', 'children'),
              Input('upload_button', 'n_clicks'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)
def upload_dataframe(n_clicks, content, filename, date):

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


@app.callback(
    Output('df_columns_dropdown_label', 'options'),
    Input('gen_dropdown', 'n_clicks'),
    prevent_initial_call=True
)
def generate_labels(click):
    # create dummy DataFrame
    #df = pd.DataFrame({f'col{i}': [1, 2, 3] for i in range(1, 5)})

    # initiate list
    options_for_dropdown = []
    for idx, colum_name in enumerate(df.columns):
        options_for_dropdown.append(
            {
                'label': colum_name,
                'value': colum_name
            }
        )
    return options_for_dropdown


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
def df_feature_div(show_df_feature):
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
               State('df_columns_dropdown_label', 'value'), ],
              prevent_initial_call=True, )
def update_df(n_clicks, criterion, splitter, max_depth, min_samples_split,
              min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
              min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label):

    if max_depth == 0:
        max_depth = None

    if df_columns_dropdown_label not in df.columns:
        pass
    else:
        cm_fig, df_feature, dummy_features_df, dummy_features_df_columns\
            = decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                            min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label)


        df_feature_columns = [{'name': col, 'id': col} for col in df_feature.columns]
        df_feature_table = df_feature.to_dict(orient='records')

        dummy_features_df_columns = [{'name': col, 'id': col} for col in dummy_features_df.columns]
        dummy_features_df_table = dummy_features_df.to_dict(orient='records')

        return cm_fig, df_feature_table, df_feature_columns, dummy_features_df_table, dummy_features_df_columns


@app.callback([Output('prediction', 'children'),
               Output('target_label', 'children'), ],
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
               State('df_columns_dropdown_label', 'value'),
               #State('input_features', 'value'),
               State('dummy_feature', 'data'), ],
              prevent_initial_call=True, )
def predictions(n_clicks, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                max_features, random_state, max_leaf_nodes, min_impurity_decrease, class_weight, ccp_alpha,
                df_columns_dropdown_label, input_features):

    data_list = []
    for key, value in input_features[0].items():
        data_list.append(float(value))

    input_features = data_list

    if max_depth == 0:
        max_depth = None

    if df_columns_dropdown_label not in df.columns:
        pass
    else:
        prediction = train_decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                                         min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                                         min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label,
                                         input_features)


        return str(prediction[0]), df_columns_dropdown_label



if __name__ == "__main__":
    app.run_server(debug=True)