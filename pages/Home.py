import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='Welcome to the Home Page'),
    html.P(["My Name is ",
            html.Span("Saad Khan.", style={'fontWeight': 'bold'}),
            " I am self-taught Data Analyst and Python Programmer. I studied Bachelor of Mechanical Engineering from "
            "Pakistan and Currently I am studying Master's of Mechanical Engineering with speciality in "
            "'Production and Logistics' From University of Duisburg Essen, Germany. This project is open source and "
            "we can contribute to its betterment in any "
            "way. "
            "I Thank you once again for visiting here and you can contact me through LinkedIn by clicking below button",
            ], style={'fontSize': '20px', 'backgroundColor': 'Lightblue'}),
    html.Br(),

    html.Div(html.P("Contact Me", style={'textAlign': 'center', 'border': '6px solid black', 'borderRadius': '20px',
                                'width':'20%'}),
             style={'display': 'flex', 'justifyContent': 'center', 'alignItem': 'center'}),
    html.Br(),

    html.Div([
        dbc.Button("LinkedIn!", target="_blank", href="https://www.linkedin.com/in/saad-khan-167704163/",
                   style={'marginRight': '10px'}),
        dbc.Button("Github!", target="_blank", href="https://github.com/SaadKhan23jan",
                   style={'marginLeft': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItem': 'center'})

])
