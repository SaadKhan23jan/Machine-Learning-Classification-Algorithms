import dash
from dash import Dash, dcc, html, dash_table, ctx, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

import base64
import io

from Algorithms import decision_tree, train_decision_tree, get_dummy_variables, eda_graph_plot, model_prediction


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
        dcc.Upload(id='upload-data_da',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                   # Do not allow multiple files to upload
                   multiple=False
                   ),
        # The below line is for if we want to show the uploaded file as Data Table
        # html.Div(id='output-data-upload_1', hidden=True),
        dbc.Button('Upload File', id='upload_button_da'),
        html.Br(),
        html.Output(id='file_uploaded_da'),

    ], style={'background': 'white'}),
    html.Br(),

    # This Div is for Explanatory Data Analysis (EDA)
    # This is connected to "app.callback() 2" and "app.callback() 3"
    html.Div([
        html.Label('Explanatory Data Analysis', style={'fontSize': '50px', 'fontWeight': 'bold'}),
        html.Br(),
        html.Button('Click to Generate Dropdown options', id='eda_gen_options_da',
                    style={'fontSize': '20px'}),

        html.Div([
            html.Label('Select X Feature', style={'width': '25%'}),
            html.Label('Select y Feature', style={'width': '25%'}),
            html.Label('Select Graph Type', style={'width': '25%'}),
            html.Label('Orientation', style={'width': '25%'}),
        ]),
        html.Div([
            dcc.Dropdown(id="x_axis_features_da", style={'width': '25%'}),
            dcc.Dropdown(id="y_axis_features_da", style={'width': '25%'}),
            dcc.Dropdown(id="graph_type_da", options=[{'label': 'Scatter', 'value': 'Scatter'},
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

            dcc.Dropdown(id="orientation_da", options=[{'label': 'Vertical', 'value': 'v'},
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
            dcc.Dropdown(id="color_da", style={'width': '25%'}),
            dcc.Dropdown(id="symbol_da", style={'width': '25%'}),
            dcc.Dropdown(id="size_da", style={'width': '25%'}),
            dcc.Dropdown(id="hover_name_da", style={'width': '25%'}, ),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Hover Data', style={'width': '25%'}),
            html.Label('Custom Data', style={'width': '25%'}),
            html.Label('Text', style={'width': '25%'}),
            html.Label('Facet Row', style={'width': '25%'}),
        ]),

        html.Div([
            dcc.Dropdown(id="hover_data_da", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="custom_data_da", style={'width': '25%'}, multi=True),
            dcc.Dropdown(id="text_da", style={'width': '25%'}),
            dcc.Dropdown(id="facet_row_da", style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.Label('Facet Column', style={'width': '25%'}),
            html.Label('Width of Figure', style={'width': '25%'}),
            html.Label('Height of Figure', style={'width': '25%'}),
            html.Label('Sort the Data by'),
        ]),

        html.Div([
            dcc.Dropdown(id="facet_col_da", style={'width': '25%'}),
            dcc.Input(id='width_da', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Input(id='height_da', style={'width': '25%'}, type='number', inputMode='numeric', step=1, min=0),
            dcc.Dropdown(id='sort_by_da', style={'width': '25%'}),
        ], style={'display': 'flex'}),

        html.Label('Click to Plot', style={'width': '25%', 'fontSize': '20px'}),
        html.Br(),
        html.Button("Plot Graph", id="plot_graph_da", style={'fontSize': '20px'}),

        dcc.Graph(id="eda_graph_da"),
    ], style={'backgroundColor': 'Lightblue'}),

    # Thi div container is for taking action on the upload file for example, df.info, df.describe, df.isna, df.dropna
    html.Div([
        html.Label('Select an Action to perform', style={'fontSize': '20px'}),
        dcc.Dropdown(id='df_actions_da', options=[{'label': 'Information', 'value': 'Information'},
                                               {'label': 'Is Null', 'value': 'Is Null'},
                                               {'label': 'Drop NA', 'value': 'Drop NA'},
                                               {'label': 'Head', 'value': 'Head'}]),
        dash_table.DataTable(id='df_actions_output_da'),
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px'}),
    html.Br(),

    # This div container is for options of hiding Data Frame, showing only head with 5 row on full Data Frame
    # This is connected to app.callback() 9
    html.Div([
        html.Label('Show the DataFrame', style={'fontSize': '20px', 'paddingRight': '20px'}),
        dcc.RadioItems(id='show_df_da', options=[{'label': 'Full   ', 'value': 'Full'},
                                              {'label': 'Head', 'value': 'Head'},
                                              {'label': 'No', 'value': 'No'}, ],
                       value='No', style={'fontSize': '20px'}, inputStyle={'marginRight': '10px'}),
        dbc.Button('Show', id='show_df_button_da', n_clicks=0, style={'fontSize': '20px'})
    ], style={'background': '#f3f2f5', 'width': '50%', 'padding': '20px', 'border': '2px solid black',
              'borderRadius': '10px', 'display': 'flex'}),
    html.Br(),

    # This div container is for the results of above options to print the Data Frame as per the chosen option
    # This is connected to "app.callback() 5" and # app.callback() 9 # can be one, will edit later
    html.Div(id='df_div_da',
             children=[
                 dash_table.DataTable(id='dataframe_da', style_table={'overflowX': 'auto'},
                                      style_cell={'height': 'auto', 'minWidth': '100px', 'width': '100px',
                                                  'maxWidth': '180px', 'whiteSpace': 'normal'},
                                      style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                      style_data={'color': 'black', 'backgroundColor': 'white',
                                                  'border': '2px solid blue'},
                                      filter_action='native'
                                      ),
             ], hidden=True),
    html.Br(),



], style={'background': 'Lightgreen'})


# This is for Uploading the csv file. it will only upload if the button is clicked
# At the same time it will call the "parse_contents" function to make global Data Frame df
# app.callback() 1
@callback(
    # Output('output-data-upload', 'children'),
    Output('file_uploaded_da', 'children'),
    Input('upload_button_da', 'n_clicks'),
    State('upload-data_da', 'contents'),
    State('upload-data_da', 'filename'),
    State('upload-data_da', 'last_modified'),
    prevent_initial_call=True)
def upload_dataframe(n_clicks, content, filename, date):
    # print(type(df))#this will show data type as a pandas dataframe
    # print(df)

    if filename is not None:
        children = parse_contents(content, filename, date)
        return f'{filename} is Uploaded Successfully...'
        # return children, f"{filename} File Uploaded Successfully"
    else:
        children = parse_contents(content, filename, date)
        return f'No File is Uploaded...'
        # return children, f"No File is Uploaded"


# This will create labels for EDA Analysis
# app.callback() 2
@callback(
    Output('x_axis_features_da', 'options'),
    Output('y_axis_features_da', 'options'),
    Output('color_da', 'options'),
    Output('symbol_da', 'options'),
    Output('size_da', 'options'),
    Output('hover_name_da', 'options'),
    Output('hover_data_da', 'options'),
    Output('custom_data_da', 'options'),
    Output('text_da', 'options'),
    Output('facet_row_da', 'options'),
    Output('facet_col_da', 'options'),
    Output('sort_by_da', 'options'),
    Input('eda_gen_options_da', 'n_clicks'),
    prevent_initial_call=True
)
def generate_labels_eda(click):
    # create dummy DataFrame
    # df = pd.DataFrame({f'col{i}': [1, 2, 3] for i in range(1, 5)})

    # initiate list
    options_for_dropdown = []
    for idx, colum_name in enumerate(df.columns):
        options_for_dropdown.append(
            {
                'label': colum_name,
                'value': colum_name
            }
        )
    return [options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
            options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown,
            options_for_dropdown, options_for_dropdown, options_for_dropdown, options_for_dropdown]


# This app.callback() is for generating Graph
# app.callback() 3
@callback(Output('eda_graph_da', 'figure'),
              [Input('plot_graph_da', 'n_clicks'),
               State('x_axis_features_da', 'value'),
               State('y_axis_features_da', 'value'),
               State('graph_type_da', 'value'),
               State('color_da', 'value'),
               State('symbol_da', 'value'),
               State('size_da', 'value'),
               State('hover_name_da', 'value'),
               State('hover_data_da', 'value'),
               State('custom_data_da', 'value'),
               State('text_da', 'value'),
               State('facet_row_da', 'value'),
               State('facet_col_da', 'value'),
               State('orientation_da', 'value'),
               State('width_da', 'value'),
               State('height_da', 'value'),
               State('sort_by_da', 'value'), ],
              prevent_initial_call=True)
def update_graph(n_clicks, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                 custom_data, text, facet_row, facet_col, orientation, width, height, sort_by):
    return eda_graph_plot(df, x_axis_features, y_axis_features, graph_type, color, symbol, size, hover_name, hover_data,
                          custom_data, text, facet_row, facet_col, orientation, width, height, sort_by)


# This app.callback() is for showing the Data Frame as per the choice
# app.callback() 5
@callback(Output('df_div_da', 'hidden'),
              Input('show_df_da', 'value'),
              prevent_initial_call=True)
def df_div(show_df):
    if show_df == "No":
        return True
    else:
        return False


# This app.callback() is for showing the  Data Frame as per the choice
# app.callback() 9
@callback([Output('dataframe_da', 'data'),
               Output('dataframe_da', 'columns'), ],
              [Input('show_df_button_da', 'n_clicks'),
               State('show_df_da', 'value'), ],
              prevent_initial_call=True)
def show_dataframe(n_clicks, show_df):
    df_columns = [{'name': col, 'id': col} for col in df.columns]
    df_table = df.to_dict(orient='records')
    if show_df == 'Head':
        df_table = df_table[:5]
        return df_table, df_columns
    else:
        return df_table, df_columns