from dash import html, dcc, Input, Output, State, callback, register_page, no_update
import dash_bootstrap_components as dbc
import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage import io

dash.register_page(__name__, path="/analysis")

empty_figure = go.Figure(data=[go.Scatter(x=[], y=[])])
empty_figure.update_xaxes(visible=False)
empty_figure.update_yaxes(visible=False)
empty_figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")


layout = dbc.Container(
    [
        dbc.Row(dcc.Graph(id="output-depth", figure=empty_figure)),
        dbc.Row(dcc.Graph(id="output-depth-res", figure=empty_figure)),
        dbc.Row(
            [
                dbc.Col(dbc.Button("Run Depth Algo", n_clicks=0, id="run-depth")),
                dbc.Col(
                    dbc.Button("Run Segmentation Algo", n_clicks=0, id="run-segment")
                ),
                dbc.Col(
                    dbc.Button(
                        "Home Page", n_clicks=0, id="upload-page-depth", href="/"
                    )
                ),
            ]
        ),
    ]
)


@callback(
    Output("output-depth", "figure"),
    Input("store-crop", "data"),
    Input("store-image", "data"),
)
def update_depth_input(data_crop, data_orig):
    if data_crop == {} and data_orig == {}:
        return no_update
    img_path = "./temp/crop.tif" if data_crop != {} else "./temp/image.tif"
    # img_arr = np.array(data, dtype=np.uint8)
    img_arr = io.imread(img_path)
    fig = px.imshow(img_arr)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


@callback(
    Output("output-depth-res", "figure"),
    Input("run-depth", "n_clicks"),
    State("store-crop", "data"),
    State("store-image", "data"),
)
def run_depth_algo(n_clicks, data_crop, data_orig):
    if data_crop == {} and data_orig == {}:
        return no_update
    img_path = "./temp/crop.tif" if data_crop != {} else "./temp/image.tif"
    if n_clicks == 0:
        return no_update
    # img_arr = np.array(data, dtype=np.uint8)
    img_arr = io.imread(img_path)
    img_arr = 255 - img_arr
    fig = px.imshow(img_arr)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    io.imsave("./temp/depth.tif", img_arr)
    return fig
