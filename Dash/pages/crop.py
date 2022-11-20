from dash import html, dcc, Input, Output, State, callback, register_page, no_update
import dash_bootstrap_components as dbc
import dash
import numpy as np
import plotly.express as px
from skimage import io
import plotly.graph_objects as go

dash.register_page(__name__, path="/crop")

empty_figure = go.Figure(data=[go.Scatter(x=[], y=[])])
empty_figure.update_xaxes(visible=False)
empty_figure.update_yaxes(visible=False)
empty_figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")

fig_config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}


layout = dbc.Container(
    [
        dbc.Row(dcc.Graph(id="output-crop", figure=empty_figure)),
        dbc.Row(dcc.Graph(id="output-crop-res", figure=empty_figure)),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Run Analysis",
                        n_clicks=0,
                        id="analysis-btn-crop",
                        href="/analysis",
                    )
                ),
                dbc.Col(
                    dbc.Button("Home Page", n_clicks=0, id="upload-page-crop", href="/")
                ),
            ]
        ),
    ]
)


@callback(
    Output("store-crop", "data"),
    Output("output-crop-res", "figure"),
    Input("output-crop", "relayoutData"),
    State("store-image", "data"),
    prevent_initial_call=True,
)
def update_crop_res(relayout_data, data):
    if (
        relayout_data is None
        or "shapes" not in relayout_data.keys()
        or len(relayout_data) == 0
    ):
        return no_update, no_update
    last_shape = relayout_data["shapes"][-1]
    x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
    x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
    # img_arr = np.array(data, dtype=np.uint8)
    img_arr = io.imread("./temp/image.tif")
    img_crop = img_arr[y0:y1, x0:x1]
    io.imsave("./temp/crop.tif", img_crop)
    fig = px.imshow(img_crop)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return "image_cropped", fig


@callback(Output("output-crop", "figure"), Input("store-image", "data"))
def update_crop_input(data):
    if data == {}:
        return no_update
    # img_arr = np.array(data, dtype=np.uint8)
    img_arr = io.imread("./temp/image.tif")
    fig = px.imshow(img_arr)
    fig.update_layout(dragmode="drawrect")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
