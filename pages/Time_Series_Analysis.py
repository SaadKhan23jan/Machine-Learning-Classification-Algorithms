import pandas as pd
import base64
import io
import dash
from dash import html, dcc, callback, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from Time_series_Analysis_Functions import sarimax_pred_tsa, decomposition_tsa
from plots import eda_graph_plot


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
            df = df.dropna()

        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df = df.dropna()

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
    # Upload CSV/Excel file
    # connected to callback 1
    html.Div([
        dcc.Upload(id='upload_data_tsa',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                   # Do not allow multiple files to upload
                   multiple=False
                   ),
        # The below line is for if we want to show the uploaded file as Data Table
        # connected to callback 1
        dbc.Button('Upload File', id='upload_button_tsa'),
        html.Br(),
        html.Output(id='file_uploaded_tsa'),

    ], style={'background': 'white'}),
    html.Br(),

    # Show DataFrame
    # This div container is for options of hiding Data Frame, showing only head with 5 row on full Data Frame
    # This is connected to app.callback() 2 for Input from this Radio Button
    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_tsa', options=[{'label': 'Full   ', 'value': 'Full'},
                                                  {'label': 'Head', 'value': 'Head'},
                                                  {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    # This div container is for the results of above options to print the Data Frame as per the chosen option
    # The div connected to "callback() 2" based on the above Radio Button options based on hidden property
    html.Div(id='df_div_tsa',
             children=[
                 dash_table.DataTable(id='dataframe_tsa', style_table={'overflowX': 'auto'},
                                      style_cell={'height': 'auto', 'minWidth': '100px', 'width': '100px',
                                                  'maxWidth': '180px', 'whiteSpace': 'normal'},
                                      style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                      style_data={'color': 'black', 'backgroundColor': 'white',
                                                  'border': '2px solid blue'},
                                      filter_action='native'
                                      ),
             ], hidden=False),
    html.Br(),

    # This Div is for Explanatory Data Analysis (EDA)
    # This is connected to "callback() 3"
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
            dcc.Dropdown(id="x_axis_features_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="y_axis_features_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="graph_type_tsa", options=[{'label': 'Scatter', 'value': 'Scatter'},
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

            dcc.Dropdown(id="orientation_tsa", options=[{'label': 'Vertical', 'value': 'v'},
                                                        {'label': 'Horizontal', 'value': 'h'}, ],
                         value='v',
                         style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Color', style={'width': '25%'}),
            html.Label('Symbol', style={'width': '25%'}),
            html.Label('Size', style={'width': '25%'}),
            html.Label('Hover Name', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="color_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="symbol_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="size_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="hover_name_tsa", style={'width': '25%'}, ),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Hover Data', style={'width': '25%'}),
            html.Label('Custom Data', style={'width': '25%'}),
            html.Label('Text', style={'width': '25%'}),
            html.Label('Facet Row', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="hover_data_tsa", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="custom_data_tsa", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="text_tsa", style={'width': '25%'}),
            dcc.Dropdown(id="facet_row_tsa", style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Facet Column', style={'width': '25%'}),
            html.Label('Width of Figure', style={'width': '25%'}),
            html.Label('Height of Figure', style={'width': '25%'}),
            html.Label('Sort the Data by'),
        ]),

        html.Div([
            dcc.Dropdown(id="facet_col_tsa", style={'width': '25%'}),
            dcc.Input(id='width_tsa', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Input(id='height_tsa', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Dropdown(id='sort_by_tsa', style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Label('Click to Plot', style={'width': '25%', 'fontSize': '20px'}),
        html.Br(),
        html.Button("Plot Graph", id="plot_graph_tsa", style={'fontSize': '20px'}),

        # This is connected to callback 3
        dcc.Graph(id="eda_graph_tsa"),
    ], style={'backgroundColor': 'Lightblue'}),

    # Thi div container is for taking action on the upload file for example, df.info, df.describe, df.isna, df.dropna
    # This is not yet functional
    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions_tsa', options=[{'label': 'Information', 'value': 'Information'},
                                               {'label': 'Is Null', 'value': 'Is Null'},
                                               {'label': 'Drop NA', 'value': 'Drop NA'},
                                               {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output_tsa'),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    # Thi div is for selecting a Label, the Label list is created from the columns of the Data Frame
    # This list is generated when the button is clicked to generate, so it is connected to a callback
    # It is connected to app.callback() 1, as I have updated it and callback 1 will generate dropdown options for all
    html.Div([
        html.Div([
            html.Label("Select the Label:",
                       style={'width': '50%', 'fontWeight': 'bold', 'paddingRight': '20px'}),
            html.Label("Select the Date Column:",
                       style={'width': '50%', 'fontWeight': 'bold', 'paddingRight': '20px'}),
        ], style={'display': 'flex'}),
        html.Br(),
        html.Div([
            dcc.Dropdown(id='df_columns_dropdown_label_tsa', style={'width': '80%'}),
            dcc.Dropdown(id='df_columns_dropdown_date_tsa', style={'width': '80%'}),
        ], style={'display': 'flex'}),

    ], style={'background': '#f3f2f5', 'fontSize': '20px', 'width': '50%', 'padding': '20px',
              'border': '2px solid black', 'borderRadius': '10px'}),
    html.Br(),

    # This div is for generating decomposed graphs
    # It is connected to callback 7
    html.Div([
        html.Button('Generate Decomposed Graphs', id='gen_decomposed_graphs_tsa', n_clicks=0,
                    style={'fontWeight': 'bold', 'fontSize': '20px', 'borderRadius': '20px', 'backgroundColor': ''}),
        dcc.Graph(id='decomposed_graphs_tsa')
    ]),


    # This Div has many divs inside and for SARIMAX Models
    html.Div([
        html.H2("SARIMAX Models Predictions"),
        html.Br(),

        html.Label('Select one of SARIMAX Model:   '),
        dcc.Dropdown(id='sarimax_model_tsa', options=[{'label': 'MA', 'value': 'MA'},
                                                      {'label': 'AR', 'value': 'AR'},
                                                      {'label': 'ARMA', 'value': 'ARMA'},
                                                      {'label': 'ARIMA', 'value': 'ARIMA'},
                                                      {'label': 'SARIMAX', 'value': 'SARIMAX'},
                                                      {'label': 'Auto ARIMA', 'value': 'Auto ARIMA'} ],
                     style={'width': '50%'},
                     value='ARIMA'
                     ),
        html.Br(),

        html.Div(
            id='auto_arima_container_tsa',
            children=html.Div([
                html.Div([
                    html.Div([
                        html.Label('Starting value of p:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start_p_order_tsa', type='number', value=0, inputMode='numeric', min=0, )
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Starting value of i:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start_i_order_tsa', type='number', value=0, inputMode='numeric', min=0)
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Starting value of q:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start_q_order_tsa', type='number', value=0, inputMode='numeric', min=0)
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    # the error-max ids are connected to the callback() 5
                    html.Div([
                        html.Label('Maximum value of p:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max_p_order_tsa', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error_max_p_tsa', children=[]),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Maximum value of i:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max_i_order_tsa', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error_max_i_tsa'),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Maximum value of q:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max_q_order_tsa', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error_max_q_tsa'),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                ], style={'display': 'flex'}),
                html.Br(),
                html.Div([
                    html.Div([

                        html.Label('Select the Seasonal Factor', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto_arima_seasonal_factor_tsa',
                                     options=[{'label': 'Quarterly', 'value': 3},
                                              {'label': '4-Monthly', 'value': 4},
                                              {'label': 'Bi-Yearly', 'value': 6},
                                              {'label': 'Yearly', 'value': 12}, ],
                                     value=12),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px',
                              'marginLeft': '10px', 'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False),

                    html.Div([
                        html.Label('Seasonal', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto_arima_seasonal_tsa',
                                     options=[{'label': 'True', 'value': 'True'},
                                              {'label': 'False', 'value': 'False'}, ],
                                     value='False'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px',
                              'marginLeft': '10px', 'paddingRight': '10px', 'paddingTop': '15px'},),
                    html.Div([
                        html.Label('Stationary', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto_arima_stationary_tsa',
                                     options=[{'label': 'True', 'value': 'True'},
                                              {'label': 'False', 'value': 'False'}, ],
                                     value='False'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px',
                              'marginLeft': '10px', 'paddingRight': '10px', 'paddingTop': '15px'},),

                    html.Div([
                        html.Label('Information Criteria', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto_arima_information_criterion_tsa',
                                     options=[{'label': 'AIC', 'value': 'aic'},
                                              {'label': 'BIC', 'value': 'bic'},
                                              {'label': 'HQIC', 'value': 'hqic'},
                                              {'label': 'OOB', 'value': 'oob'}, ],
                                     value='aic'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px',
                              'marginLeft': '10px', 'paddingRight': '10px', 'paddingTop': '15px'},),

                    html.Div([
                        html.Label('Method to be used', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto_arima_method_tsa',
                                     options=[{'label': 'Newton-Raphson', 'value': 'newton'},
                                              {'label': 'Nelder-Mead', 'value': 'nm'},
                                              {'label': ' Broyden-Fletcher-Goldfarb-Shanno (BFGS)', 'value': 'bfgs'},
                                              {'label': 'limited-memory BFGS with optional box constraints',
                                               'value': 'lbfgs'},
                                              {'label': 'modified Powell’s method', 'value': 'powell'},
                                              {'label': 'conjugate gradient', 'value': 'cg'},
                                              {'label': 'Newton-conjugate gradient', 'value': 'ncg'},
                                              {'label': 'global basin-hopping solver', 'value': 'basinhopping'}, ],
                                     value='lbfgs')
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px',
                              'marginLeft': '10px', 'paddingRight': '10px', 'paddingTop': '15px'},)
                ], style={'display': 'flex'}),

            ]),
            hidden=True,
        ),

        html.Div(
            id='sarimax_container_tsa',
            children=html.Div([
                html.Div([
                    html.Label('Enter the order of P:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax_p_order_tsa', type='number', value=0, inputMode='numeric', min=0, )
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Enter the order of I:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax_i_order_tsa', type='number', value=0, inputMode='numeric', min=0)
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Enter the order of Q:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax_q_order_tsa', type='number', value=0, inputMode='numeric', min=0, )
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Select the Seasonal Factor'),
                    dcc.Dropdown(id='seasonal_factor_tsa', options=[{'label': 'Quarterly', 'value': 3},
                                                                    {'label': '4-Monthly', 'value': 4},
                                                                    {'label': 'Bi-Yearly', 'value': 6},
                                                                    {'label': 'Yearly', 'value': 12}, ],
                                 value=12),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False),

            ], style={'display': 'flex'}),
            hidden=True,
        ),

        html.Br(),
        html.Br(),

        html.Div([
            html.Div(
                id='p_order_container_tsa',
                children=[
                    html.Label('Enter the order of p:', style={'paddingRight': '20px'}),
                    dcc.Input(id='p_order_tsa', type='number', placeholder='Enter the order of P', value=0,
                              inputMode='numeric', min=0, required=True),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),

            html.Div(
                id='i_order_container_tsa',
                children=[
                    html.Label('Enter the order of i:', style={'paddingRight': '20px'}),
                    dcc.Input(id='i_order_tsa', type='number', placeholder='Enter the order of I', value=0,
                              inputMode='numeric', min=0),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),

            html.Div(
                id='q_order_container_tsa',
                children=[
                    html.Label('Enter the order of q:', style={'paddingRight': '20px'}),
                    dcc.Input(id='q_order_tsa', type='number', placeholder='Enter the order of Q', value=0,
                              inputMode='numeric', min=0),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),
            html.Div([
                html.Label('Days for Forecast:', style={'paddingRight': '20px'}),
                dcc.Input(id='days_tsa', type='number', placeholder='Enter the order of Q', value=14,
                          inputMode='numeric', min=0,),
            ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                      'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False
            ),

        ], style={'display': 'flex'}),

        html.Br(),

        # This is connected to the dash.callback 4
        html.Div([
            html.Div([html.P(id='run_calc_tsa', children=['Calculations are not yet Started'])]),
            html.Button(id='run_pred_tsa', children='Run Forecast',
                        style={'weight': 'bold', 'height': '100px', 'borderRadius': '30px', 'paddingRight': '20px',
                               'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Button(id='stop_pred_tsa', children='Stop Forecast',
                        style={'weight': 'bold', 'height': '100px', 'borderRadius': '30px', 'paddingLeft': '20px',
                               'fontWeight': 'bold', 'fontSize': '20px'}),

        ]),


        html.Br(),
        html.Br(),

        # This is for showing the results of SARIMAX
        # Connected to callback 8
        html.Div(id='sarimax_results_tsa'),

    ]),


    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.H1([f'Predictions Through ', html.Label(id='model_used_tsa', style={'color': 'blue', 'weight': 'bold'}),
                 ' Model']),

    ], style={'borderRadius': '20px', 'width': '800px',
              'backgroundColor': '#e6f5f4', 'padding': '30px'}),


    dcc.Graph(id='fig_pred_tsa'),

    html.P('***Note***: This is for demo purpose and the results are never claimed to be correct, nor can be used'
           ' for real time prediction of the actual data', style={'color': 'white', 'backgroundColor': 'black'}),



], style={'background-color': 'Lightgreen'})


# This callback is for upload the CSV/Excel file
# I have updated this callback, and it will generate dropdown options for every Dropdown component
# when upload button is clicked
# app.callback() 1
@callback([
    Output('file_uploaded_tsa', 'children'),
    Output('x_axis_features_tsa', 'options'),
    Output('y_axis_features_tsa', 'options'),
    Output('color_tsa', 'options'),
    Output('symbol_tsa', 'options'),
    Output('size_tsa', 'options'),
    Output('hover_name_tsa', 'options'),
    Output('hover_data_tsa', 'options'),
    Output('custom_data_tsa', 'options'),
    Output('text_tsa', 'options'),
    Output('facet_row_tsa', 'options'),
    Output('facet_col_tsa', 'options'),
    Output('sort_by_tsa', 'options'),
    Output('df_columns_dropdown_label_tsa', 'options'),
    Output('df_columns_dropdown_date_tsa', 'options'), ],
    Input('upload_button_tsa', 'n_clicks'),
    State('upload_data_tsa', 'contents'),
    State('upload_data_tsa', 'filename'),
    State('upload_data_tsa', 'last_modified'),
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

        return [f'{filename} is Uploaded Successfully...',
                options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
                options_for_dropdown, options_for_dropdown]
    else:
        children = parse_contents(content, filename, date)
        return f'No File is Uploaded...'


# This app.callback() is for showing the  Data Frame as per the Radio Button Options Choice
# Note here that in the first output "hidden" must be used no children "spent 2 hours on figuring out this
# app.callback() 2
@callback([Output('df_div_tsa', 'hidden'),
           Output('dataframe_tsa', 'data'),
           Output('dataframe_tsa', 'columns'), ],
          [Input('show_df_tsa', 'value'), ],
          prevent_initial_call=True)
def show_dataframe(show_df_tsa):
    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')
    if show_df_tsa == 'No':
        return True, df_table, df_columns
    elif show_df_tsa == 'Head':
        df_table = df_table[:5]
        return False, df_table, df_columns
    else:
        return False, df_table, df_columns


# This app.callback() is for generating Graph
# app.callback() 3
@callback(Output('eda_graph_tsa', 'figure'),
          [Input('plot_graph_tsa', 'n_clicks'),
           State('x_axis_features_tsa', 'value'),
           State('y_axis_features_tsa', 'value'),
           State('graph_type_tsa', 'value'),
           State('color_tsa', 'value'),
           State('symbol_tsa', 'value'),
           State('size_tsa', 'value'),
           State('hover_name_tsa', 'value'),
           State('hover_data_tsa', 'value'),
           State('custom_data_tsa', 'value'),
           State('text_tsa', 'value'),
           State('facet_row_tsa', 'value'),
           State('facet_col_tsa', 'value'),
           State('orientation_tsa', 'value'),
           State('width_tsa', 'value'),
           State('height_tsa', 'value'),
           State('sort_by_tsa', 'value'), ],
          prevent_initial_call=True)
def update_graph(n_clicks, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                 custom_data, text, facet_row, facet_col, orientation, width, height, sort_by):
    return eda_graph_plot(df, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                          custom_data, text, facet_row, facet_col, orientation, width, height, sort_by)


# This is disabling the run-pred button, so that it is not clickable before the previous calculations completed
# dash.callback() 4 Note here it is dash.callback() instead of callback because of its special functionality
@dash.callback(
    output=Output('run_calc_tsa', 'children'),
    inputs=Input('run_pred_tsa', 'n_clicks'),
    background=True,
    running=[
        (Output('run_pred_tsa', 'disabled'), True, False),
        (Output('stop_pred_tsa', 'disabled'), False, True),
    ],
    cancel=[Input('stop_pred_tsa', 'n_clicks')],
    prevent_initial_call=True, )
def run_calculations(n_clicks):
    # time.sleep(2.0)
    return [f"Click on Run to Start Calculations or Stop to Cancel the Calculations or Refresh the page"]


# For SARIMAX max orders must be greater than the starting values, therefore it will alert to the user
# callback() 5
@callback([Output('error_max_p_tsa', 'children'),
           Output('error_max_i_tsa', 'children'),
           Output('error_max_q_tsa', 'children'), ],
          [Input('max_p_order_tsa', 'value'),
           Input('max_i_order_tsa', 'value'),
           Input('max_q_order_tsa', 'value'),
           Input('start_p_order_tsa', 'value'),
           Input('start_i_order_tsa', 'value'),
           Input('start_q_order_tsa', 'value'), ]
          )
def error_alert(max_p_order, max_i_order, max_q_order, start_p_order,  start_i_order,  start_q_order):

    # This code is short form, else it can be written in 6 lines of if-else statements
    alert_p, alert_i, alert_q, = ('', '', '')

    if max_p_order <= start_p_order:
        alert_p = 'Should be greater than Starting p value'
    if max_i_order <= start_i_order:
        alert_i = 'Should be greater than Starting i value'
    if max_q_order <= start_q_order:
        alert_q = 'Should be greater than Starting q value'

    return alert_p, alert_i, alert_q


# This callback has the functionality of hiding and un-hiding the inputs according to the type of SARIMAX Model
# callback() 6
@callback([Output('auto_arima_container_tsa', 'hidden'),
           Output('sarimax_container_tsa', 'hidden'),
           Output('p_order_container_tsa', 'hidden'),
           Output('i_order_container_tsa', 'hidden'),
           Output('q_order_container_tsa', 'hidden'), ],
          [Input('sarimax_model_tsa', 'value'), ],
          prevent_initial_call=True)
def update_output(sarimax_model):
    if sarimax_model == 'SARIMAX':
        return True, False, False, False, False
    elif sarimax_model == 'MA':
        return True, True, True, True, False
    elif sarimax_model == 'AR':
        return True, True, False, True, True
    elif sarimax_model == 'ARMA':
        return True, True, False, True, False
    elif sarimax_model == 'ARIMA':
        return True, True, False, False, False
    elif sarimax_model == 'Auto ARIMA':
        return False, True, True, True, True
    else:
        return False, False, True, True, True


# It will create decomposed graph of Observed, Trend, Seasonality, Residuals
# app.callback() 7
@callback(Output('decomposed_graphs_tsa', 'figure'),
          [Input('gen_decomposed_graphs_tsa', 'n_clicks'),
           State('df_columns_dropdown_label_tsa', 'value'),
           State('df_columns_dropdown_date_tsa', 'value'), ],
          prevent_initial_call=True)
def update_graph(n_clicks, df_columns_dropdown_label_tsa, df_columns_dropdown_date_tsa):

    fig_seasonality_decompose = decomposition_tsa(df, df_columns_dropdown_label_tsa, df_columns_dropdown_date_tsa)

    return fig_seasonality_decompose


# This callback is for running calculations'
# app.callback() 8`
@callback([Output('sarimax_results_tsa', 'children'),
           Output('fig_pred_tsa', 'figure'),
           Output('model_used_tsa', 'children'), ],
          [Input('run_pred_tsa', 'n_clicks'),
           State('df_columns_dropdown_label_tsa', 'value'),
           State('p_order_tsa', 'value'),
           State('i_order_tsa', 'value'),
           State('q_order_tsa', 'value'),
           State('sarimax_model_tsa', 'value'),
           State('days_tsa', 'value'),
           State('sarimax_p_order_tsa', 'value'),
           State('sarimax_i_order_tsa', 'value'),
           State('sarimax_q_order_tsa', 'value'),
           State('seasonal_factor_tsa', 'value'),
           State('start_p_order_tsa', 'value'),
           State('start_i_order_tsa', 'value'),
           State('start_q_order_tsa', 'value'),
           State('max_p_order_tsa', 'value'),
           State('max_i_order_tsa', 'value'),
           State('max_q_order_tsa', 'value'),
           State('auto_arima_seasonal_factor_tsa', 'value'),
           State('auto_arima_seasonal_tsa', 'value'),
           State('auto_arima_stationary_tsa', 'value'),
           State('auto_arima_information_criterion_tsa', 'value'),
           State('auto_arima_method_tsa', 'value'), ],
          prevent_initial_call=True)
def predictions(n_clicks, df_columns_dropdown_label_tsa, p, i, q, sarimax_model, days, sp, si, sq, seasonal_factor,
                start_p_order, start_i_order, start_q_order, max_p_order, max_i_order, max_q_order,
                auto_arima_seasonal_factor, auto_arima_seasonal, auto_arima_stationary,
                auto_arima_information_criterion, auto_arima_method):

    # Here we will call our function for SARIMAX Model

    if auto_arima_seasonal == 'True':
        auto_arima_seasonal = True
    else:
        auto_arima_seasonal = False

    if auto_arima_stationary == 'True':
        auto_arima_stationary = True
    else:
        auto_arima_stationary = False

    results, pred_fig = sarimax_pred_tsa(df, df_columns_dropdown_label_tsa, p, i, q, sarimax_model, days, sp, si, sq,
                                         seasonal_factor, start_p_order, start_i_order, start_q_order, max_p_order,
                                         max_i_order, max_q_order, auto_arima_seasonal_factor, auto_arima_seasonal,
                                         auto_arima_stationary, auto_arima_information_criterion, auto_arima_method)

    return results, pred_fig, sarimax_model
