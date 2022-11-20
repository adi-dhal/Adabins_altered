import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    [
        dbc.Row(html.H1("Multipage Trial App"), style={"textAlign": "center"}),
        dbc.Row(dash.page_container),
        dbc.Row(
            [
                dcc.Store(id="store-image", data={}),
                dcc.Store(id="store-crop", data={}),
                dcc.Store(id="store-depth", data={}),
            ]
        ),
    ],
    className="dbc",
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=True)
