from dash import html, dcc, Input, Output, State, callback, register_page, no_update
import dash_bootstrap_components as dbc
import dash
from tkinter import filedialog as fd
from tkinter import Tk
from skimage import io
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# from app import book_keep
register_page(__name__, path="/")

empty_figure = go.Figure(data=[go.Scatter(x=[], y=[])])
empty_figure.update_xaxes(visible=False)
empty_figure.update_yaxes(visible=False)
empty_figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")

layout = html.H1("Historical Archive")
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Button(
                "Upload Image",
                id="upload-data",
                n_clicks=0,
                style={"width": "30%"},
                # Allow multiple files to be uploaded
            ),
        ),
        dbc.Row(dcc.Graph(id="output-upload", figure=empty_figure)),
        dbc.Row(
            [
                dbc.Col(width=3),
                dbc.Col(
                    dbc.Button(
                        "Crop Image", n_clicks=0, id="crop-btn-upload", href="/crop"
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        "Run Analysis",
                        n_clicks=0,
                        id="analysis-btn-upload",
                        href="/analysis",
                    ),
                    align="center",
                ),
            ]
        ),
    ]
)


# @callback(
#     Output("crop-btn-upload", "href"),
#     Output("analysis-btn-upload", "href"),
#     Input("upload-data", "n_clicks"),
# )
# def move_crop(n_clicks):
#     if n_clicks == 0:
#         return no_update, no_update
#     return "/crop", "/analysis"


@callback(
    Output("store-image", "data"),
    Output("output-upload", "figure"),
    Input("upload-data", "n_clicks"),
)
def update_figure(n_clicks):
    if n_clicks == 0:
        return no_update, no_update
    root = Tk()
    root.withdraw()
    root.focus_force()
    root.update()
    filenames = fd.askopenfilenames(
        title="Open files",
        initialdir="/home/adityadhall/Bosch",
    )

    root.destroy()
    if len(filenames) == 0:
        return no_update
    filename = filenames[-1]
    img = io.imread(filename)
    img_list = img.tolist()
    fig = px.imshow(img)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    io.imsave("./temp/image.tif", img)
    return "image_uploaded", fig
