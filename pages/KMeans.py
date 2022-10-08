import dash
from dash import dcc, html, dash_table, ctx, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
from Algorithms import run_kmeans_cluster, model_prediction
from plots import eda_graph_plot


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
layout = html.Div([

    # This div container is for uploading file (uploaded when clicked the "Upload File" button)
    # This is connected to "app.callback() 1"
    html.Div([
        dcc.Upload(id='upload_data_kmc',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                   # Do not allow multiple files to upload
                   multiple=False
                   ),

        dbc.Button('Upload File', id='upload_button_kmc'),
        html.Br(),
        html.Output(id='file_uploaded_kmc'),

    ], style={'background': 'white'}),
    html.Br(),

    # This Div is for Explanatory Data Analysis (EDA)
    # These all are connected to "app.callback() 1" while the graph is connected to callback() 2
    html.Div([
        html.Label('Explanatory Data Analysis', style={'fontSize': '50px', 'fontWeight': 'bold'}),
        html.Br(),

        html.Div([
            html.Label('Select X Feature', style={'width': '25%'}),
            html.Label('Select y Feature', style={'width': '25%'}),
            html.Label('Select Graph Type', style={'width': '25%'}),
            html.Label('Orientation', style={'width': '25%'}),
        ]),
        html.Div([
            dcc.Dropdown(id="x_axis_features_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="y_axis_features_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="graph_type_kmc", options=[{'label': 'Scatter', 'value': 'Scatter'},
                                                      {'label': 'Line', 'value': 'Line'},
                                                      {'label': 'Area', 'value': 'Area'},
                                                      {'label': 'Bar', 'value': 'Bar'},
                                                      {'label': 'Funnel', 'value': 'Funnel'},
                                                      {'label': 'Timeline', 'value': 'Timeline'},
                                                      {'label': 'Pie', 'value': 'Pie'},
                                                      {'label': 'Subburst', 'value': 'Subburst'},
                                                      {'label': 'Treemap', 'value': 'Treemap'},
                                                      {'label': 'Icicle', 'value': 'Icicle'},
                                                      {'label': 'Funnel Area', 'value': 'Funnel Area'},
                                                      {'label': 'Histogram', 'value': 'Histogram'},
                                                      {'label': 'Box', 'value': 'Box'},
                                                      {'label': 'Violin', 'value': 'Violin'},
                                                      {'label': 'Strip', 'value': 'Strip'},
                                                      {'label': 'ECDF', 'value': 'ECDF'},
                                                      {'label': 'Density Heatmap', 'value': 'Density Heatmap'},
                                                      {'label': 'Density Contour', 'value': 'Density Contour'},
                                                      {'label': 'Imshow', 'value': 'Imshow'}, ],
                         value='Histogram', style={'width': '25%'}),

            dcc.Dropdown(id="orientation_kmc", options=[{'label': 'Vertical', 'value': 'v'},
                                                       {'label': 'Horizontal', 'value': 'h'}, ],
                         style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Color', style={'width': '25%'}),
            html.Label('Symbol', style={'width': '25%'}),
            html.Label('Size', style={'width': '25%'}),
            html.Label('Hover Name', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="color_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="symbol_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="size_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="hover_name_kmc", style={'width': '25%'}, ),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Hover Data', style={'width': '25%'}),
            html.Label('Custom Data', style={'width': '25%'}),
            html.Label('Text', style={'width': '25%'}),
            html.Label('Facet Row', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="hover_data_kmc", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="custom_data_kmc", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="text_kmc", style={'width': '25%'}),
            dcc.Dropdown(id="facet_row_kmc", style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Facet Column', style={'width': '25%'}),
            html.Label('Width of Figure', style={'width': '25%'}),
            html.Label('Height of Figure', style={'width': '25%'}),
            html.Label('Sort the Data by'),
        ]),

        html.Div([
            dcc.Dropdown(id="facet_col_kmc", style={'width': '25%'}),
            dcc.Input(id='width_kmc', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Input(id='height_kmc', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Dropdown(id='sort_by_kmc', style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Label('Click to Plot', style={'width': '25%', 'fontSize': '20px'}),
        html.Br(),
        html.Button("Plot Graph", id="plot_graph_kmc", style={'fontSize': '20px'}),

        # This is connected to callback() 2
        dcc.Graph(id="eda_graph_kmc"),
    ], style={'backgroundColor': 'Lightblue'}),

    # Thi div container is for taking action on the upload file for example, df.info, df.describe, df.isna, df.dropna
    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions_kmc', options=[{'label': 'Information', 'value': 'Information'},
                                                  {'label': 'Is Null', 'value': 'Is Null'},
                                                  {'label': 'Drop NA', 'value': 'Drop NA'},
                                                  {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output_kmc'),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    # This div container is for options of hiding Data Frame, showing only head with 5 row on full Data Frame
    # This is connected to app.callback() 9
    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_kmc', options=[{'label': 'Full   ', 'value': 'Full'},
                                                 {'label': 'Head', 'value': 'Head'},
                                                 {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    # This div container is for the results of above options to print the Data Frame as per the chosen option
    # This is connected to "app.callback() 3"
    html.Div(id='df_div_kmc',
             children=[
                 dash_table.DataTable(id='dataframe_kmc', style_table={'overflowX': 'auto'},
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


        # This div has one div inside for each parameter
        html.Div([
            html.Div([
                html.Label('N Cluster', style={'fontWeight': 'bold'}),
                dcc.Input(id='n_clusters_kmc', type='number', value=8, step=1, inputMode='numeric', min=2),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Method for initialization', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='init_kmc', options=[{'label': 'k-means++', 'value': 'k-means++'},
                                                     {'label': 'Random', 'value': 'random'}, ],
                             value='k-means++'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Number for initialization', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='n_init_kmc', type='number', value=10, step=1, inputMode='numeric', min=1),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Maximum number of iterations', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='max_iter_kmc', type='number', value=300, step=1, inputMode='numeric', min=1),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Tol', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='tol_kmc', type='number', value=1, step=0.000001, inputMode='numeric', min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Random State', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='random_state_kmc', type='number', value=None, step=0.000001, inputMode='numeric',
                          min=0, max=0.5, placeholder='default(None)'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Copy X', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='copy_x_kmc', options=[{'label': 'True', 'value': True},
                                                       {'label': 'False', 'value': False}, ],
                             value=True),
            ], style={'width': '12%'}),

        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        # This div has one div inside for each parameter in addition to the above
        html.Div([
            html.Div([
                html.Label('Algorithm', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='algorithm_kmc', options=[{'label': 'elkan', 'value': 'elkan'},
                                                          {'label': 'auto', 'value': 'auto'},
                                                          {'label': 'full', 'value': 'full'}, ],
                             value='auto'),
            ], style={'width': '12%'}),
        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        # This button runs the calculations, so it is connected to a callback and runs the calculations
        html.Div([
            dbc.Button('Run KMeans Cluster', id='run_kmc', n_clicks=0,
                       style={'fontSize': '20px', 'marginRight': '20px'}),
        ], style={'display': 'flex'}),
        html.Br(),
        html.Br(),

    ]),

    # This is almost the same div for confusion matrix as above for feature importance
    html.Div([
        html.Label('Show Cluster Figure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_clusters_kmc', options=[{'label': 'Yes', 'value': 'Yes'},
                                                  {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),

    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned confusion matrix from model and the function heatmap_plot_confusion_matrix
    # app.callback() 7
    html.Div(id='show_cm_graph_kmc',
             children=[
                 dcc.Graph(id='confusion_matrix_kmc'),
             ], hidden=True),
    html.Br(),

    # This table appears once the model is trained, it is editable
    # The values from these cells are converted into list in the call back and input into the model.predict
    html.Div([
        html.Label('Enter These variables values for Prediction', style={'fontSize': '20px', 'fontWeight': 'bold'}),
        dash_table.DataTable(id='dummy_feature_kmc', editable=True),
    ], style={'backgroundColor': 'Lightblue'}),
    html.Br(),

    # This button is now only for prediction and is separated for training model on all data again and again
    # Model will be trained separately once, and this button will only do the prediction quickly
    # The values from these cells are converted into list in the call back and input into the model.predict
    html.Div([
        dbc.Button('Click to Predict', id='prediction_button_kmc', n_clicks=0,
                   style={'fontSize': '20px', 'fontWeight': 'bold'}),
    ]),
    html.Br(),

    # This Div is for showing the results of the Predictions from the model
    # id='prediction_kmc' is connected to callback 8
    html.Div([html.Label('Prediction with the Selected Parameters set: ',
                         style={'fontSize': '20px', 'fontWeight': 'bold', 'paddingRight': '10px'}),
              html.Div([
                  html.Label("Cluster Position:", style={'paddingRight': '20px'}),
                  html.Div(id='prediction_kmc'),
              ], style={'display': 'flex', 'border': '2px solid black', 'borderRadius': '10px'}),
              html.Br(),
              html.Br(), ],
             style={'width': '50%', 'background': '#f3f2f5', 'fontSize': '20px', 'fontWeight': 'bold',
                    'paddingRight': '10px', 'paddingLeft': '200px', 'borderRadius': '50px'}),

    html.Div(),
    html.Div(),
    html.Div(),

], style={'background': 'Lightgreen'})


# This is for Uploading the csv file. it will only upload if the button is clicked
# At the same time it will call the "parse_contents" function to make global Data Frame df
# app.callback() 1
@callback(
    [Output('file_uploaded_kmc', 'children'),
     Output('x_axis_features_kmc', 'options'),
     Output('y_axis_features_kmc', 'options'),
     Output('color_kmc', 'options'),
     Output('symbol_kmc', 'options'),
     Output('size_kmc', 'options'),
     Output('hover_name_kmc', 'options'),
     Output('hover_data_kmc', 'options'),
     Output('custom_data_kmc', 'options'),
     Output('text_kmc', 'options'),
     Output('facet_row_kmc', 'options'),
     Output('facet_col_kmc', 'options'),
     Output('sort_by_kmc', 'options'), ],
    Input('upload_button_kmc', 'n_clicks'),
    State('upload_data_kmc', 'contents'),
    State('upload_data_kmc', 'filename'),
    State('upload_data_kmc', 'last_modified'),
    prevent_initial_call=True)
def upload_dataframe(n_clicks, content, filename, date):
    if filename is not None:
        children = parse_contents(content, filename, date)

        # initiate list
        options_for_dropdown = []
        for idx, colum_name in enumerate(df.columns):
            options_for_dropdown.append(
                {
                    'label': colum_name,
                    'value': colum_name
                }
            )

        return [f'{filename} is Uploaded Successfully...', options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown,]
    else:
        children = parse_contents(content, filename, date)
        return f'No File is Uploaded...'


# This app.callback() is for generating Graph
# app.callback() 2
@callback(Output('eda_graph_kmc', 'figure'),
          [Input('plot_graph_kmc', 'n_clicks'),
           State('x_axis_features_kmc', 'value'),
           State('y_axis_features_kmc', 'value'),
           State('graph_type_kmc', 'value'),
           State('color_kmc', 'value'),
           State('symbol_kmc', 'value'),
           State('size_kmc', 'value'),
           State('hover_name_kmc', 'value'),
           State('hover_data_kmc', 'value'),
           State('custom_data_kmc', 'value'),
           State('text_kmc', 'value'),
           State('facet_row_kmc', 'value'),
           State('facet_col_kmc', 'value'),
           State('orientation_kmc', 'value'),
           State('width_kmc', 'value'),
           State('height_kmc', 'value'),
           State('sort_by_kmc', 'value'), ],
          prevent_initial_call=True)
def update_graph(n_clicks, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                 custom_data, text, facet_row, facet_col, orientation, width, height, sort_by):
    return eda_graph_plot(df, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                          custom_data, text, facet_row, facet_col, orientation, width, height, sort_by)


# This app.callback() is for showing the  Data Frame as per the Radio Button Options Choice
# Note here that in the first output "hidden" must be used no children "spent 2 hours on figuring out this
# app.callback() 3
@callback([Output('df_div_kmc', 'hidden'),
           Output('dataframe_kmc', 'data'),
           Output('dataframe_kmc', 'columns'), ],
          Input('show_df_kmc', 'value'),
          prevent_initial_call=True)
def show_dataframe(show_df_kmc):
    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')
    if show_df_kmc == 'No':
        return True, df_table, df_columns
    elif show_df_kmc == 'Head':
        df_table = df_table[:5]
        return False, df_table, df_columns
    else:
        return False, df_table, df_columns


# This app.callback() is for showing the Confusion Matrix as per the choice
# app.callback() 5
@callback(Output('show_cm_graph_kmc', 'hidden'),
          Input('show_clusters_kmc', 'value'),
          prevent_initial_call=True)
def cm_graph(show_cm):
    if show_cm == "Yes":
        return False
    else:
        return True


""" 
    This app.callback() will split the data into Train-Test Split, print the Feature Importance, Confusion Matrix
    and the Decision Tree, on this basis we will tune the Hyper parameters to increase the accuracy
    can be checked from the confusion matrix
"""


# callback 7
@callback([Output('prediction_kmc', 'children'),
           Output('confusion_matrix_kmc', 'figure'),
           Output('dummy_feature_kmc', 'data'),
           Output('dummy_feature_kmc', 'columns'), ],
          [Input('run_kmc', 'n_clicks'),
           Input('prediction_button_kmc', 'n_clicks'),
           State('n_clusters_kmc', 'value'),
           State('init_kmc', 'value'),
           State('n_init_kmc', 'value'),
           State('max_iter_kmc', 'value'),
           State('tol_kmc', 'value'),
           State('random_state_kmc', 'value'),
           State('copy_x_kmc', 'value'),
           State('algorithm_kmc', 'value'),
           State('dummy_feature_kmc', 'data'), ],
          prevent_initial_call=True, )
def run_randomforest_classifier(run_kmc, prediction_button_kmc, n_clusters_kmc, init_kmc, n_init_kmc, max_iter_kmc,
                                tol_kmc, random_state_kmc, copy_x_kmc, algorithm_kmc, input_features):

    triggered_id = ctx.triggered_id
    if triggered_id == 'run_kmc':
        global trained_model  # I made it gloabl so that it is available for return in both condition, else creates error
        global dummy_features_df_table
        global dummy_features_df_columns
        global fig

        fig, trained_model, dummy_features_df, dummy_features_df_columns = run_kmeans_cluster(df, n_clusters_kmc, init_kmc,
                                                                                      n_init_kmc, max_iter_kmc, tol_kmc,
                                                                                      random_state_kmc, copy_x_kmc,
                                                                                      algorithm_kmc)

        dummy_features_df_columns = [{'name': col, 'id': col} for col in dummy_features_df.columns]
        dummy_features_df_table = dummy_features_df.to_dict(orient='records')

        return "No Label Available", fig, dummy_features_df_table, dummy_features_df_columns
    elif triggered_id == 'prediction_button_kmc':

        # These line will take the data from the cells, convert it into float and make list from it.
        # It will be given as input for the prediction
        # It is inside in this KMeans, because it is all combined now in 1 callback
        # So if this input_features is out, error will be generated, so it must be created after model run first
        data_list = []
        for key, value in input_features[0].items():
            data_list.append(float(value))
        input_features = data_list

        prediction = model_prediction(trained_model, input_features)
        return str(prediction[0]), fig, dummy_features_df_table, dummy_features_df_columns