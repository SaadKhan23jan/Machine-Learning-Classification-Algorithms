import dash
from dash import dcc, html, dash_table, ctx, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
from Algorithms import logistic_regression, train_logistic_regression, model_prediction
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
        dcc.Upload(id='upload_data_lr',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                   # Do not allow multiple files to upload
                   multiple=False
                   ),

        dbc.Button('Upload File', id='upload_button_lr'),
        html.Br(),
        html.Output(id='file_uploaded_lr'),

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
            dcc.Dropdown(id="x_axis_features_lr", style={'width': '25%'}),
            dcc.Dropdown(id="y_axis_features_lr", style={'width': '25%'}),
            dcc.Dropdown(id="graph_type_lr", options=[{'label': 'Scatter', 'value': 'Scatter'},
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

            dcc.Dropdown(id="orientation_lr", options=[{'label': 'Vertical', 'value': 'v'},
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
            dcc.Dropdown(id="color_lr", style={'width': '25%'}),
            dcc.Dropdown(id="symbol_lr", style={'width': '25%'}),
            dcc.Dropdown(id="size_lr", style={'width': '25%'}),
            dcc.Dropdown(id="hover_name_lr", style={'width': '25%'}, ),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Hover Data', style={'width': '25%'}),
            html.Label('Custom Data', style={'width': '25%'}),
            html.Label('Text', style={'width': '25%'}),
            html.Label('Facet Row', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="hover_data_lr", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="custom_data_lr", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="text_lr", style={'width': '25%'}),
            dcc.Dropdown(id="facet_row_lr", style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Facet Column', style={'width': '25%'}),
            html.Label('Width of Figure', style={'width': '25%'}),
            html.Label('Height of Figure', style={'width': '25%'}),
            html.Label('Sort the Data by'),
        ]),

        html.Div([
            dcc.Dropdown(id="facet_col_lr", style={'width': '25%'}),
            dcc.Input(id='width_lr', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Input(id='height_lr', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Dropdown(id='sort_by_lr', style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Label('Click to Plot', style={'width': '25%', 'fontSize': '20px'}),
        html.Br(),
        html.Button("Plot Graph", id="plot_graph_lr", style={'fontSize': '20px'}),

        # This is connected to callback() 2
        dcc.Graph(id="eda_graph_lr"),
    ], style={'backgroundColor': 'Lightblue'}),

    # Thi div container is for taking action on the upload file for example, df.info, df.describe, df.isna, df.dropna
    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions_lr', options=[{'label': 'Information', 'value': 'Information'},
                                                  {'label': 'Is Null', 'value': 'Is Null'},
                                                  {'label': 'Drop NA', 'value': 'Drop NA'},
                                                  {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output_lr'),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    # This div container is for options of hiding Data Frame, showing only head with 5 row on full Data Frame
    # This is connected to app.callback() 9
    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_lr', options=[{'label': 'Full   ', 'value': 'Full'},
                                                 {'label': 'Head', 'value': 'Head'},
                                                 {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    # This div container is for the results of above options to print the Data Frame as per the chosen option
    # This is connected to "app.callback() 3"
    html.Div(id='df_div_lr',
             children=[
                 dash_table.DataTable(id='dataframe_lr', style_table={'overflowX': 'auto'},
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
        # It is connected to app.callback() 1
        html.Div([
            html.Label("Select the Label to Predict:",
                       style={'width': '50%', 'fontWeight': 'bold', 'paddingRight': '20px'}),
            dcc.Dropdown(id='df_columns_dropdown_label_lr', style={'width': '80%'}),
        ], style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '50%', 'padding': '20px',
                  'border': '2px solid black', 'borderRadius': '10px', 'display': 'flex'}),
        html.Br(),

        # This div has one div inside for each parameter
        html.Div([
            html.Div([
                html.Label('Penalty', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='penalty_lr', options=[{'label': 'L1', 'value': 'l1'},
                                                       {'label': 'L2', 'value': 'l2'},
                                                       {'label': 'ElasticNet', 'value': 'elasticnet'},
                                                       {'label': 'None', 'value': 'none'}, ],
                             value='l2'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Dual', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='dual_lr', options=[{'label': 'True', 'value': True},
                                                    {'label': 'False', 'value': False}, ],
                             value=False),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Tolerance', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='tol_lr', type='number', value=0.0001, step=0.000001, inputMode='numeric', min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('C', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='c_lr', type='number', value=1.0, step=0.000001, inputMode='numeric', min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Fit Intercept', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Dropdown(id='fit_intercept_lr', options=[{'label': 'True', 'value': True},
                                                             {'label': 'False', 'value': False}, ],
                             value=True),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Intercept Scaling', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='intercept_scaling_lr', type='number', value=1.0, step=0.000001, inputMode='numeric',
                          min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Random State', style={'fontWeight': 'bold'}),
                dcc.Input(id='random_state_lr', value=None, placeholder='default (None)'),
            ], style={'width': '12%'}),

        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        # This div has one div inside for each parameter in addition to the above
        html.Div([
            html.Div([
                html.Label('Solver', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='solver_lr', options=[{'label': 'newton-cg', 'value': 'newton-cg'},
                                                      {'label': 'lbfgs', 'value': 'lbfgs'},
                                                      {'label': 'liblinear', 'value': 'liblinear'},
                                                      {'label': 'sag', 'value': 'sag'},
                                                      {'label': 'saga', 'value': 'saga'}, ],
                             value='lbfgs'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Max Iterations', style={'fontWeight': 'bold'}),
                dcc.Input(id='max_iter_lr', type='number', value=100, step=1, inputMode='numeric', min=0),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('Multi Class', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Dropdown(id='multi_class_lr', options=[{'label': 'auto', 'value': 'auto'},
                                                           {'label': 'ovr', 'value': 'ovr'},
                                                           {'label': 'multinomial', 'value': 'multinomial'}, ],
                             value='auto'),
            ], style={'width': '12%'}),

            html.Div([
                html.Label('L1 Ratio', style={'fontWeight': 'bold'}),
                html.Br(),
                dcc.Input(id='l1_ratio_lr', type='number', value=None, placeholder='default (None)',
                          min=0, max=1, step=0.000001),
            ], style={'width': '12%'}),

        ], style={'background': 'Lightblue', 'display': 'flex'}),
        html.Br(),

        # This button runs the calculations, so it is connected to a callback and runs the calculations
        # The inside div is for showing the accuracy of the Model
        html.Div([
            dbc.Button('Run LogisticRegression', id='run_lr', n_clicks=0,
                       style={'fontSize': '20px', 'marginRight': '20px'}),
            html.Div([
                html.Label("The Accuracy of the Model is",
                           style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label(id='model_accuracy_score_lr', style={'fontSize': '20px', 'fontWeight': 'bold',
                                                                'marginLeft': '20px', 'color': 'green'}),
            ], style={'backgroundColor': 'Lightblue', 'width': '25%', 'border': '2px solid black',
                      'borderRadius': '20px'}, ),
        ], style={'display': 'flex'}),
        html.Br(),
        html.Br(),

    ]),

    # This div has radio button, if Yes, then it will show Data Frame of feature importance
    html.Div([
        html.Label('Show the Feature Importance', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_feature_lr', options=[{'label': 'Yes', 'value': 'Yes'},
                                                         {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned Data Frame from the Decision model and also connected to callback from the above div Yes/No
    # app.callback() 4
    html.Div(id='df_feature_div_lr',
             children=[
                 dash_table.DataTable(id='df_feature_lr'),
             ], hidden=True),
    html.Br(),

    # This is almost the same div for confusion matrix as above for feature importance
    html.Div([
        html.Label('Show Confusion Matrix Figure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_cm_lr', options=[{'label': 'Yes', 'value': 'Yes'},
                                                 {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),

    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned confusion matrix from model and the function heatmap_plot_confusion_matrix
    # app.callback() 5
    html.Div(id='show_cm_graph_lr',
             children=[
                 dcc.Graph(id='confusion_matrix_lr'),
             ], hidden=True),
    html.Br(),

    # Same radio buttons for DecisionTree Diagram as above for Feature Importance Data Frame and Confusion Matrix
    # app.callback() 8
    html.Div([
        html.Label('Show Tree Structure', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_dt_lr', options=[{'label': 'Yes', 'value': 'Yes'},
                                                 {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This button is for training the model on whole data and then predict the value
    html.Div([
        dbc.Button('Train Model on All Data', id='train_model_lr', n_clicks=0,
                   style={'fontSize': '20px', 'fontWeight': 'bold'}),
        html.P("Model Not Trained", id="message_lr", style={'fontWeight': 'bold', 'padding': '20px'}),
    ], style={'display': 'flex'}),
    html.Br(),

    # This div has radio button, if Yes, then it will show Data Frame of feature importance
    html.Div([
        html.Label('Show the Feature Importance of Trained Model', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_feature_trained_lr', options=[{'label': 'Yes', 'value': 'Yes'},
                                                                 {'label': 'No', 'value': 'No'}],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '25%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),

    # This is a returned Data Frame from the Decision model and also connected to callback from the above div Yes/No
    # app.callback() 7
    html.Div(id='df_feature_div_trained_lr',
             children=[
                 dash_table.DataTable(id='df_feature_trained_lr'),
             ], hidden=False),
    html.Br(),

    # This table appears once the model is trained, it is editable
    # The values from these cells are converted into list in the call back and input into the model.predict
    html.Div([
        html.Label('Enter These variables values for Prediction', style={'fontSize': '20px', 'fontWeight': 'bold'}),
        dash_table.DataTable(id='dummy_feature_lr', editable=True),
    ], style={'backgroundColor': 'Lightblue'}),
    html.Br(),

    # This button is now only for prediction and is separated for training model on all data again and again
    # Model will be trained separately once, and this button will only do the prediction quickly
    # The values from these cells are converted into list in the call back and input into the model.predict
    html.Div([
        dbc.Button('Click to Predict', id='prediction_button_lr', n_clicks=0,
                   style={'fontSize': '20px', 'fontWeight': 'bold'}),
    ]),
    html.Br(),

    # This Div is for showing the results of the Predictions from the model
    # id='prediction_lr' is connected to callback 9
    html.Div([html.Label('Prediction with the Selected Parameters set: ',
                         style={'fontSize': '20px', 'fontWeight': 'bold', 'paddingRight': '10px'}),
              html.Div([
                  html.Label("Prediction Label:"),
                  html.Div(id='target_label_lr', style={'paddingRight': '20px', 'paddingLeft': '20px'}),
                  html.Label("Predicted Value:", style={'paddingRight': '20px'}),
                  html.Div(id='prediction_lr'),
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
    [Output('file_uploaded_lr', 'children'),
     Output('x_axis_features_lr', 'options'),
     Output('y_axis_features_lr', 'options'),
     Output('color_lr', 'options'),
     Output('symbol_lr', 'options'),
     Output('size_lr', 'options'),
     Output('hover_name_lr', 'options'),
     Output('hover_data_lr', 'options'),
     Output('custom_data_lr', 'options'),
     Output('text_lr', 'options'),
     Output('facet_row_lr', 'options'),
     Output('facet_col_lr', 'options'),
     Output('sort_by_lr', 'options'),
     Output('df_columns_dropdown_label_lr', 'options'), ],
    Input('upload_button_lr', 'n_clicks'),
    State('upload_data_lr', 'contents'),
    State('upload_data_lr', 'filename'),
    State('upload_data_lr', 'last_modified'),
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
                options_for_dropdown, options_for_dropdown, options_for_dropdown]
    else:
        children = parse_contents(content, filename, date)
        return f'No File is Uploaded...'


# This app.callback() is for generating Graph
# app.callback() 2
@callback(Output('eda_graph_lr', 'figure'),
          [Input('plot_graph_lr', 'n_clicks'),
           State('x_axis_features_lr', 'value'),
           State('y_axis_features_lr', 'value'),
           State('graph_type_lr', 'value'),
           State('color_lr', 'value'),
           State('symbol_lr', 'value'),
           State('size_lr', 'value'),
           State('hover_name_lr', 'value'),
           State('hover_data_lr', 'value'),
           State('custom_data_lr', 'value'),
           State('text_lr', 'value'),
           State('facet_row_lr', 'value'),
           State('facet_col_lr', 'value'),
           State('orientation_lr', 'value'),
           State('width_lr', 'value'),
           State('height_lr', 'value'),
           State('sort_by_lr', 'value'), ],
          prevent_initial_call=True)
def update_graph(n_clicks, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                 custom_data, text, facet_row, facet_col, orientation, width, height, sort_by):
    return eda_graph_plot(df, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                          custom_data, text, facet_row, facet_col, orientation, width, height, sort_by)


# This app.callback() is for showing the  Data Frame as per the Radio Button Options Choice
# Note here that in the first output "hidden" must be used no children "spent 2 hours on figuring out this
# app.callback() 3
@callback([Output('df_div_lr', 'hidden'),
           Output('dataframe_lr', 'data'),
           Output('dataframe_lr', 'columns'), ],
          Input('show_df_lr', 'value'),
          prevent_initial_call=True)
def show_dataframe(show_df_lr):
    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')
    if show_df_lr == 'No':
        return True, df_table, df_columns
    elif show_df_lr == 'Head':
        df_table = df_table[:5]
        return False, df_table, df_columns
    else:
        return False, df_table, df_columns


# This app.callback() is for showing the Feature Importance Data Frame as per the choice
# app.callback() 4
@callback(Output('df_feature_div_lr', 'hidden'),
          Input('show_df_feature_lr', 'value'),
          prevent_initial_call=True)
def df_feature_div(show_df_feature):
    if show_df_feature == "No":
        return True
    else:
        return False


# This app.callback() is for showing the Confusion Matrix as per the choice
# app.callback() 5
@callback(Output('show_cm_graph_lr', 'hidden'),
          Input('show_cm_lr', 'value'),
          prevent_initial_call=True)
def cm_graph(show_cm):
    if show_cm == "Yes":
        return False
    else:
        return True


# app.callback() 7
@callback(Output('df_feature_div_trained_lr', 'children'),
          Input('show_df_feature_trained_lr', 'value'),
          prevent_initial_call=True)
def df_feature_div(show_df_feature_trained):
    if show_df_feature_trained == "No":
        return True
    else:
        return False


""" 
    This app.callback() will split the data into Train-Test Split, print the Feature Importance, Confusion Matrix
    and the Decision Tree, on this basis we will tune the Hyper parameters to increase the accuracy
    can be checked from the confusion matrix
"""


# callback 8
@callback([Output('confusion_matrix_lr', 'figure'),
           Output('df_feature_lr', 'data'),
           Output('df_feature_lr', 'columns'),
           Output('dummy_feature_lr', 'data'),
           Output('dummy_feature_lr', 'columns'),
           Output('model_accuracy_score_lr', 'children'), ],
          [Input('run_lr', 'n_clicks'),
           State('penalty_lr', 'value'),
           State('dual_lr', 'value'),
           State('tol_lr', 'value'),
           State('c_lr', 'value'),
           State('fit_intercept_lr', 'value'),
           State('intercept_scaling_lr', 'value'),
           State('random_state_lr', 'value'),
           State('solver_lr', 'value'),
           State('max_iter_lr', 'value'),
           State('multi_class_lr', 'value'),
           State('l1_ratio_lr', 'value'),
           State('df_columns_dropdown_label_lr', 'value'), ],
          prevent_initial_call=True, )
def run_decision_tree(n_clicks, penalty_lr, dual_lr, tol_lr, c_lr, fit_intercept_lr, intercept_scaling_lr,
                      random_state_lr, solver_lr, max_iter_lr, multi_class_lr, l1_ratio_lr, df_columns_dropdown_label):

    cm_fig, df_feature, dummy_features_df, dummy_features_df_columns, model_accuracy_score =\
        logistic_regression(df, penalty_lr, dual_lr, tol_lr, c_lr, fit_intercept_lr, intercept_scaling_lr,
                            random_state_lr, solver_lr, max_iter_lr, multi_class_lr, l1_ratio_lr,
                            df_columns_dropdown_label)

    df_feature_columns = [{'name': col, 'id': col} for col in df_feature.columns]
    df_feature_table = df_feature.to_dict(orient='records')

    dummy_features_df_columns = [{'name': col, 'id': col} for col in dummy_features_df.columns]
    dummy_features_df_table = dummy_features_df.to_dict(orient='records')

    return [cm_fig, df_feature_table, df_feature_columns, dummy_features_df_table,
            dummy_features_df_columns, model_accuracy_score]


"""
    This app.callback() will train the model on all the data and then also predict the value
"""


# callback 9
@callback([Output('prediction_lr', 'children'),
           Output('target_label_lr', 'children'),
           Output('message_lr', 'children'),
           Output('df_feature_trained_lr', 'data'),
           Output('df_feature_trained_lr', 'columns'), ],
          [Input('train_model_lr', 'n_clicks'),
           Input('prediction_button_lr', 'n_clicks'),
           State('penalty_lr', 'value'),
           State('dual_lr', 'value'),
           State('tol_lr', 'value'),
           State('c_lr', 'value'),
           State('fit_intercept_lr', 'value'),
           State('intercept_scaling_lr', 'value'),
           State('random_state_lr', 'value'),
           State('solver_lr', 'value'),
           State('max_iter_lr', 'value'),
           State('multi_class_lr', 'value'),
           State('l1_ratio_lr', 'value'),
           State('df_columns_dropdown_label_lr', 'value'),
           State('dummy_feature_lr', 'data'), ],
          prevent_initial_call=True, )
def predictions(n_clicks_train_model, n_clicks_prediction_button, penalty_lr, dual_lr, tol_lr, c_lr, fit_intercept_lr,
                intercept_scaling_lr, random_state_lr, solver_lr, max_iter_lr, multi_class_lr, l1_ratio_lr,
                df_columns_dropdown_label, input_features):
    triggered_id = ctx.triggered_id

    # These line will take the data from the cells, convert it into float and make list from it.
    # It will be given as input for the prediction
    data_list = []
    for key, value in input_features[0].items():
        data_list.append(float(value))
    input_features = data_list

    if triggered_id == 'train_model_lr':
        global trained_model  # I made it gloabl so that it is available for return in both condition, else creates error
        global df_feature_trained_columns
        global df_feature_trained_table
        trained_model, df_feature_trained = train_logistic_regression(df, penalty_lr, dual_lr, tol_lr, c_lr,
                                                                      fit_intercept_lr, intercept_scaling_lr,
                                                                      random_state_lr, solver_lr, max_iter_lr,
                                                                      multi_class_lr, l1_ratio_lr,
                                                                      df_columns_dropdown_label)

        df_feature_trained_columns = [{'name': col, 'id': col} for col in df_feature_trained.columns]
        df_feature_trained_table = df_feature_trained.to_dict(orient='records')

        return "No Label Available", "No Prediction Available", f"Model Trained {n_clicks_train_model} times", df_feature_trained_table, df_feature_trained_columns
    elif triggered_id == 'prediction_button_lr':
        prediction = model_prediction(trained_model, input_features)
        return str(prediction[0]), df_columns_dropdown_label, f"Model Trained {n_clicks_train_model} times", df_feature_trained_table, df_feature_trained_columns
