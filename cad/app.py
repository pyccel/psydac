# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq             as daq
from dash.dependencies import Output, Input, State
import dash_table
import pandas as pd
import sys
import os
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten
from collections import OrderedDict
from collections import namedtuple

# bsplines
from bsplines_utilities import point_on_bspline_curve
from bsplines_utilities import point_on_bspline_surface
from bsplines_utilities import insert_knot_bspline_curve
from bsplines_utilities import elevate_degree_bspline_curve
from bsplines_utilities import insert_knot_bspline_surface
from bsplines_utilities import elevate_degree_bspline_surface

# nurbs
from bsplines_utilities import point_on_nurbs_curve
from bsplines_utilities import point_on_nurbs_surface
from bsplines_utilities import insert_knot_nurbs_curve
from bsplines_utilities import insert_knot_nurbs_surface
from bsplines_utilities import elevate_degree_nurbs_curve
from bsplines_utilities import elevate_degree_nurbs_surface

# bsplines
from bsplines_utilities import translate_bspline_curve
from bsplines_utilities import rotate_bspline_curve
from bsplines_utilities import homothetic_bspline_curve

# nurbs
from bsplines_utilities import translate_nurbs_curve
from bsplines_utilities import rotate_nurbs_curve
from bsplines_utilities import homothetic_nurbs_curve

from datatypes import SplineCurve
from datatypes import SplineSurface
from datatypes import SplineVolume
from datatypes import NurbsCurve
from datatypes import NurbsSurface
from datatypes import NurbsVolume

from gallery import make_line
from gallery import make_arc
from gallery import make_square
from gallery import make_circle
from gallery import make_half_annulus_cubic
from gallery import make_L_shape_C1

# ... global variables
namespace = OrderedDict()
model_id = 0
# ...

# ... global dict for time stamps
d_timestamp = OrderedDict()
d_timestamp['load']      = -10000
d_timestamp['refine']    = -10000
d_timestamp['transform'] = -10000
d_timestamp['edit']      = -10000

d_timestamp['line']     = -10000
d_timestamp['arc']      = -10000
d_timestamp['square']   = -10000
d_timestamp['circle']   = -10000
d_timestamp['half_annulus_cubic'] = -10000
d_timestamp['L_shape_C1'] = -10000
d_timestamp['cube']     = -10000
d_timestamp['cylinder'] = -10000

d_timestamp['insert']    = -10000
d_timestamp['elevate']   = -10000
d_timestamp['subdivide'] = -10000

d_timestamp['translate']  = -10000
d_timestamp['rotate']     = -10000
d_timestamp['homothetie'] = -10000
# ...

# ...
def plot_curve(crv, nx=101, control_polygon=False):
    knots  = crv.knots
    degree = crv.degree
    P      = crv.points

    n  = len(knots) - degree - 1

    # ... curve
    xs = np.linspace(0., 1., nx)

    Q = np.zeros((nx, 2))

    if isinstance(crv, SplineCurve):
        for i,x in enumerate(xs):
            Q[i,:] = point_on_bspline_curve(knots, P, x)

    elif isinstance(crv, NurbsCurve):
        W = crv.weights
        for i,x in enumerate(xs):
            Q[i,:] = point_on_nurbs_curve(knots, P, W, x)

    line_marker = dict(color='#0066FF', width=2)
    x = Q[:,0] ; y = Q[:,1]

    trace_crv = go.Scatter(
        x=x,
        y=y,
        mode = 'lines',
        name='Curve',
        line=line_marker,
    )
    # ...

    if not control_polygon:
        return [trace_crv]

    # ... control polygon
    line_marker = dict(color='#ff7f0e', width=2)

    x = P[:,0] ; y = P[:,1]

    trace_ctrl = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Control polygon',
        line=line_marker,
    )
    # ...

    return [trace_crv, trace_ctrl]
# ...

# ...
def plot_surface(srf, Nu=101, Nv=101, control_polygon=False):
    Tu, Tv = srf.knots
    pu, pv = srf.degree
    P      = srf.points

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    us = np.linspace(0., 1., Nu)
    vs = np.linspace(0., 1., Nv)

    lines = []
    line_marker = dict(color='#0066FF', width=2)

    # ...
    Q = np.zeros((len(gridu), Nv, 2))
    if isinstance(srf, SplineSurface):
        for i,u in enumerate(gridu):
            for j,v in enumerate(vs):
                Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    elif isinstance(srf, NurbsSurface):
        W = srf.weights
        for i,u in enumerate(gridu):
            for j,v in enumerate(vs):
                Q[i,j,:] = point_on_nurbs_surface(Tu, Tv, P, W, u, v)

    for i in range(len(gridu)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[i,:,0],
                             y=Q[i,:,1])
                 ]
    # ...

    # ...
    Q = np.zeros((Nu, len(gridv), 2))
    if isinstance(srf, SplineSurface):
        for i,u in enumerate(us):
            for j,v in enumerate(gridv):
                Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    elif isinstance(srf, NurbsSurface):
        W = srf.weights
        for i,u in enumerate(us):
            for j,v in enumerate(gridv):
                Q[i,j,:] = point_on_nurbs_surface(Tu, Tv, P, W, u, v)

    for j in range(len(gridv)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[:,j,0],
                             y=Q[:,j,1])
                 ]
    # ...

    if not control_polygon:
        return lines

    # ... control polygon
    line_marker = dict(color='#ff7f0e', width=2)

    for i in range(nu):
        lines += [go.Scatter(mode = 'lines+markers',
                             line=line_marker,
                             x=P[i,:,0],
                             y=P[i,:,1])
                 ]

    for j in range(nv):
        lines += [go.Scatter(mode = 'lines+markers',
                             line=line_marker,
                             x=P[:,j,0],
                             y=P[:,j,1])
                 ]
    # ...

    return lines
# ...

# ...
def model_from_data(data):
    # ...
    weights = None
    try:
        knots, degree, points = data

        points = np.asarray(points)

    except:
        try:
            knots, degree, points, weights = data

            points = np.asarray(points)
            weights = np.asarray(weights)

        except:
            raise ValueError('Could not retrieve data')
    # ...

    if isinstance(knots, (tuple, list)):
        knots = [np.asarray(T) for T in knots]

    if isinstance(degree, int):
        if weights is None:
            current_model = SplineCurve(knots=knots,
                                        degree=degree,
                                        points=points)

        else:
            current_model = NurbsCurve(knots=knots,
                                       degree=degree,
                                       points=points,
                                       weights=weights)

    elif len(degree) == 2:
        if weights is None:
            current_model = SplineSurface(knots=knots,
                                          degree=degree,
                                          points=points)

        else:
            current_model = NurbsSurface(knots=knots,
                                         degree=degree,
                                         points=points,
                                         weights=weights)

    return current_model
# ...

# =================================================================
tab_line = dcc.Tab(label='line', children=[
                              html.Label('origin'),
                              dcc.Input(id='line_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('end'),
                              dcc.Input(id='line_end',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='line_submit',
                                          n_clicks_timestamp=0),

])

# =================================================================
tab_arc = dcc.Tab(label='arc', children=[
                              html.Label('center'),
                              dcc.Input(id='arc_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('radius'),
                              dcc.Input(id='arc_radius',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('angle'),
                              dcc.Dropdown(id="arc_angle",
                                           options=[{'label': '90', 'value': '90'},
                                                    {'label': '120', 'value': '120'},
                                                    {'label': '180', 'value': '180'}],
                                           value=[],
                                           multi=False),
                              html.Button('Submit', id='arc_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_square = dcc.Tab(label='square', children=[
                              html.Label('origin'),
                              dcc.Input(id='square_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('length'),
                              dcc.Input(id='square_length',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='square_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_circle = dcc.Tab(label='circle', children=[
                              html.Label('center'),
                              dcc.Input(id='circle_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('radius'),
                              dcc.Input(id='circle_radius',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='circle_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_half_annulus_cubic = dcc.Tab(label='half_annulus_cubic', children=[
                              html.Label('center'),
                              dcc.Input(id='half_annulus_cubic_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('rmax'),
                              dcc.Input(id='half_annulus_cubic_rmax',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('rmin'),
                              dcc.Input(id='half_annulus_cubic_rmin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='half_annulus_cubic_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_L_shape_C1 = dcc.Tab(label='L_shape_C1', children=[
                              html.Label('center'),
                              dcc.Input(id='L_shape_C1_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='L_shape_C1_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_cube = dcc.Tab(label='cube', children=[
                              html.Label('origin'),
                              dcc.Input(id='cube_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='cube_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_cylinder = dcc.Tab(label='cylinder', children=[
                              html.Label('origin'),
                              dcc.Input(id='cylinder_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='cylinder_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_geometry_1d = dcc.Tab(label='1D', children=[
                          dcc.Tabs(children=[
                                   tab_line,
                                   tab_arc,
                          ]),
])

# =================================================================
tab_geometry_2d = dcc.Tab(label='2D', children=[
                          dcc.Tabs(children=[
                                   tab_square,
                                   tab_circle,
                                   tab_half_annulus_cubic,
                                   tab_L_shape_C1,
                          ]),
])

# =================================================================
tab_geometry_3d = dcc.Tab(label='3D', children=[
                          dcc.Tabs(children=[
                                   tab_cube,
                                   tab_cylinder,
                          ]),
])


# =================================================================
tab_loader = dcc.Tab(label='Load', children=[
                     html.Button('load', id='button_load',
                                 n_clicks_timestamp=0),
                     dcc.Store(id='loaded_model'),
                     dcc.Tabs(children=[
                              tab_geometry_1d,
                              tab_geometry_2d,
                              tab_geometry_3d
                     ]),
])


# =================================================================
tab_insert_knot = dcc.Tab(label='Insert knot', children=[
                          html.Div([
                              html.Label('Knot'),
                              dcc.Input(id='insert_knot_value',
                                        placeholder='Enter a value ...',
                                        value='',
                                        # we use text rather than number to avoid
                                        # having the incrementation/decrementation
                                        type='text'
                              ),
                              html.Label('times'),
                              daq.NumericInput(id='insert_knot_times',
                                               min=1,
                                               value=0
                              ),
                              html.Button('Submit', id='insert_submit',
                                          n_clicks_timestamp=0),
                          ]),
])

# =================================================================
tab_elevate_degree = dcc.Tab(label='Elevate degree', children=[
                             html.Div([
                                 html.Label('times'),
                                 daq.NumericInput(id='elevate_degree_times',
                                                  min=0,
                                                  value=0
                                 ),
                              html.Button('Submit', id='elevate_submit',
                                          n_clicks_timestamp=0),
                             ]),
])

# =================================================================
tab_subdivision = dcc.Tab(label='Subdivision', children=[
                             html.Div([
                                 html.Label('times'),
                                 daq.NumericInput(id='subdivision_times',
                                                  min=0,
                                                  value=0
                                 ),
                              html.Button('Submit', id='subdivide_submit',
                                          n_clicks_timestamp=0),
                             ]),
])

# =================================================================
tab_refinement = dcc.Tab(label='Refinement', children=[
                         dcc.Store(id='refined_model'),
                         html.Div([
                             # ...
                             html.Label('Axis'),
                             dcc.Dropdown(id="axis",
                                          options=[{'label': 'u', 'value': '0'},
                                                   {'label': 'v', 'value': '1'},
                                                   {'label': 'w', 'value': '2'}],
                                          value=[],
                                          multi=True),
                             html.Button('Apply', id='button_refine',
                                         n_clicks_timestamp=0),
                             html.Hr(),
                             # ...

                             # ...
                             dcc.Tabs(children=[
                                      tab_insert_knot,
                                      tab_elevate_degree,
                                      tab_subdivision
                             ]),
                             # ...
                         ])
])

# =================================================================
tab_translate = dcc.Tab(label='Translate', children=[
                             html.Div([
                                 html.Label('displacement'),
                                 dcc.Input(id='translate_disp',
                                           placeholder='Enter a value ...',
                                           value='',
                                           type='text'),
                                 html.Button('Submit', id='translate_submit',
                                             n_clicks_timestamp=0),
                             ]),
])

# =================================================================
tab_rotate = dcc.Tab(label='Rotate', children=[
                             html.Div([
                                 html.Label('center'),
                                 dcc.Input(id='rotate_center',
                                           placeholder='Enter a value ...',
                                           value='',
                                           type='text'),
                                 html.Label('angle'),
                                 dcc.Input(id='rotate_angle',
                                           placeholder='Enter a value ...',
                                           value='',
                                           type='text'),
                                 html.Button('Submit', id='rotate_submit',
                                             n_clicks_timestamp=0),
                             ]),
])

# =================================================================
tab_homothetie = dcc.Tab(label='Homothetie', children=[
                             html.Div([
                                 html.Label('center'),
                                 dcc.Input(id='homothetie_center',
                                           placeholder='Enter a value ...',
                                           value='',
                                           type='text'),
                                 html.Label('scale'),
                                 dcc.Input(id='homothetie_alpha',
                                           placeholder='Enter a value ...',
                                           value='',
                                           type='text'),
                              html.Button('Submit', id='homothetie_submit',
                                          n_clicks_timestamp=0),
                             ]),
])


# =================================================================
tab_transformation = dcc.Tab(label='Transformation', children=[
                             dcc.Store(id='transformed_model'),
                             html.Div([
                                 # ...
                                 html.Button('Apply', id='button_transform',
                                             n_clicks_timestamp=0),
                                 dcc.Tabs(children=[
                                          tab_translate,
                                          tab_rotate,
                                          tab_homothetie,
                                 ]),
                                 # ...
                             ])
])

# =================================================================
tab_editor = dcc.Tab(label='Editor', children=[
                     html.Button('Edit', id='button_edit',
                                 n_clicks_timestamp=0),
                     dcc.Store(id='edited_model'),
                     html.Div([
                         html.Div(id='editor-Tu'),
                         html.Div(id='editor-Tv'),
                         html.Div(id='editor-Tw'),
                         html.Div(id='editor-degree'),
                         dash_table.DataTable(id='editor-table',
                                              columns=[],
                                              editable=True),
                     ])
])

# =================================================================
tab_viewer = dcc.Tab(label='Viewer', children=[

                    html.Label('Geometry'),
                    dcc.Dropdown(id="model",
                                 options=[{'label':name, 'value':name}
                                          for name in namespace.keys()],
                                 value=[],
                                 multi=True),

                     html.Div([
                         daq.BooleanSwitch(label='Control polygon',
                           id='control_polygon',
                           on=False
                         ),
                         # ...
                         html.Div([
                             dcc.Graph(id="graph")]),
                         # ...
                     ])
])

# =================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # ...
    html.H1(children='CAID'),
    # ...

    # ...
    dcc.Tabs(id="tabs", children=[
        tab_viewer,
        tab_loader,
        tab_refinement,
        tab_transformation,
        tab_editor,
    ]),
    html.Div(id='tabs-content-example')
    # ...
])

# =================================================================
@app.callback(
    Output("loaded_model", "data"),
    [Input('button_load', 'n_clicks_timestamp'),
     Input('line_origin', 'value'),
     Input('line_end',    'value'),
     Input('line_submit', 'n_clicks_timestamp'),
     Input('arc_center', 'value'),
     Input('arc_radius', 'value'),
     Input('arc_angle',  'value'),
     Input('arc_submit', 'n_clicks_timestamp'),
     Input('square_origin', 'value'),
     Input('square_length', 'value'),
     Input('square_submit', 'n_clicks_timestamp'),
     Input('circle_center', 'value'),
     Input('circle_radius', 'value'),
     Input('circle_submit', 'n_clicks_timestamp'),
     Input('half_annulus_cubic_center', 'value'),
     Input('half_annulus_cubic_rmax', 'value'),
     Input('half_annulus_cubic_rmin', 'value'),
     Input('half_annulus_cubic_submit', 'n_clicks_timestamp'),
     Input('L_shape_C1_center', 'value'),
     Input('L_shape_C1_submit', 'n_clicks_timestamp')]
)
def load_model(time_clicks,
               line_origin, line_end,
               line_submit_time,
               arc_center, arc_radius, arc_angle,
               arc_submit_time,
               square_origin, square_length,
               square_submit_time,
               circle_center, circle_radius,
               circle_submit_time,
               half_annulus_cubic_center,
               half_annulus_cubic_rmax,
               half_annulus_cubic_rmin,
               half_annulus_cubic_submit_time,
               L_shape_C1_center,
               L_shape_C1_submit_time):

    global d_timestamp

    if time_clicks <= d_timestamp['load']:
        return None

    d_timestamp['load'] = time_clicks

    if ( not( line_origin is '' ) and
         not( line_end is '' ) and
         not( line_submit_time <= d_timestamp['line'] )
       ):
        # ...
        try:
            line_origin = [float(i) for i in line_origin.split(',')]

        except:
            raise ValueError('Cannot convert line_origin')
        # ...

        # ...
        try:
            line_end = [float(i) for i in line_end.split(',')]

        except:
            raise ValueError('Cannot convert line_end')
        # ...

        d_timestamp['line'] = line_submit_time

        return make_line(origin=line_origin,
                         end=line_end)

    elif ( not( arc_center is '' ) and
           not( arc_radius is '' ) and
           arc_angle and
           not( arc_submit_time <= d_timestamp['arc'] )
         ):
        # ...
        try:
            arc_center = [float(i) for i in arc_center.split(',')]

        except:
            raise ValueError('Cannot convert arc_center')
        # ...

        # ...
        try:
            arc_radius = float(arc_radius)

        except:
            raise ValueError('Cannot convert arc_radius')
        # ...

        # ...
        try:
            arc_angle = float(arc_angle)

        except:
            raise ValueError('Cannot convert arc_angle')
        # ...

        d_timestamp['arc'] = arc_submit_time

        return make_arc(center=arc_center,
                        radius=arc_radius,
                        angle=arc_angle)


    elif ( not( square_origin is '' ) and
           not( square_length is '' ) and
           not( square_submit_time <= d_timestamp['square'] )
        ):
        # ...
        try:
            square_origin = [float(i) for i in square_origin.split(',')]

        except:
            raise ValueError('Cannot convert square_origin')
        # ...

        # ...
        try:
            square_length = float(square_length)

        except:
            raise ValueError('Cannot convert square_length')
        # ...

        d_timestamp['square'] = square_submit_time

        return make_square(origin=square_origin,
                           length=square_length)

    elif ( not( circle_center is '' ) and
           not( circle_radius is '' ) and
           not( circle_submit_time <= d_timestamp['circle'] )
         ):
        # ...
        try:
            circle_center = [float(i) for i in circle_center.split(',')]

        except:
            raise ValueError('Cannot convert circle_center')
        # ...

        # ...
        try:
            circle_radius = float(circle_radius)

        except:
            raise ValueError('Cannot convert circle_radius')
        # ...

        d_timestamp['circle'] = circle_submit_time

        return make_circle(center=circle_center,
                           radius=circle_radius)

    elif ( not( half_annulus_cubic_center is '' ) and
           not( half_annulus_cubic_rmax is '' ) and
           not( half_annulus_cubic_rmin is '' ) and
           not( half_annulus_cubic_submit_time <= d_timestamp['half_annulus_cubic'] )
         ):
        # ...
        try:
            half_annulus_cubic_center = [float(i) for i in half_annulus_cubic_center.split(',')]

        except:
            raise ValueError('Cannot convert half_annulus_cubic_center')
        # ...

        # ...
        try:
            half_annulus_cubic_rmax = float(half_annulus_cubic_rmax)

        except:
            raise ValueError('Cannot convert half_annulus_cubic_rmax')
        # ...

        # ...
        try:
            half_annulus_cubic_rmin = float(half_annulus_cubic_rmin)

        except:
            raise ValueError('Cannot convert half_annulus_cubic_rmin')
        # ...

        d_timestamp['half_annulus_cubic'] = half_annulus_cubic_submit_time

        return make_half_annulus_cubic(center=half_annulus_cubic_center,
                           rmax=half_annulus_cubic_rmax,
                           rmin=half_annulus_cubic_rmin)

    elif ( not( L_shape_C1_center is '' ) and
           not( L_shape_C1_submit_time <= d_timestamp['L_shape_C1'] )
         ):
        # ...
        try:
            L_shape_C1_center = [float(i) for i in L_shape_C1_center.split(',')]

        except:
            raise ValueError('Cannot convert L_shape_C1_center')
        # ...

        d_timestamp['L_shape_C1'] = L_shape_C1_submit_time

        return make_L_shape_C1(center=L_shape_C1_center)

    else:
        return None

# =================================================================
@app.callback(
    Output("refined_model", "data"),
    [Input("model", "value"),
     Input('button_refine', 'n_clicks_timestamp'),
     Input('insert_knot_value', 'value'),
     Input('insert_knot_times', 'value'),
     Input('insert_submit', 'n_clicks_timestamp'),
     Input('elevate_degree_times', 'value'),
     Input('elevate_submit', 'n_clicks_timestamp'),
     Input('subdivision_times', 'value'),
     Input('subdivide_submit', 'n_clicks_timestamp')]
)
def apply_refine(models,
                 time_clicks,
                 t, t_times,
                 insert_submit_time,
                 m,
                 elevate_submit_time,
                 levels,
                 subdivide_submit_time):

    global d_timestamp

    if time_clicks <= d_timestamp['refine']:
        return None

    d_timestamp['refine'] = time_clicks

    if len(models) == 0:
        return None

    if len(models) > 1:
        return None

    name  = models[0]
    model = namespace[name]

    # ... insert knot
    if not( t is '' ) and not( insert_submit_time <= d_timestamp['insert'] ):
        times = int(t_times)
        t = float(t)

        if isinstance(model, (SplineCurve, NurbsCurve)):
            t_min = model.knots[ model.degree]
            t_max = model.knots[-model.degree]
            if t > t_min and t < t_max:
                if isinstance(model, SplineCurve):
                    knots, degree, P = insert_knot_bspline_curve( model.knots,
                                                          model.degree,
                                                          model.points,
                                                          t, times=times )

                    model = SplineCurve(knots=knots,
                                        degree=degree,
                                        points=P)

                elif isinstance(model, NurbsCurve):
                    knots, degree, P, W = insert_knot_nurbs_curve( model.knots,
                                                                model.degree,
                                                                model.points,
                                                                model.weights,
                                                                t, times=times )

                    model = NurbsCurve(knots=knots,
                                       degree=degree,
                                       points=P,
                                       weights=W)


        elif isinstance(model, (SplineSurface, NurbsSurface)):
            u_min = model.knots[0][ model.degree[0]]
            u_max = model.knots[0][-model.degree[0]]
            v_min = model.knots[1][ model.degree[1]]
            v_max = model.knots[1][-model.degree[1]]
            condition = False
            # TODO
            if t > u_min and t < u_max:
                if isinstance(model, SplineSurface):
                    Tu, Tv, pu, pv, P = insert_knot_bspline_surface( *model.knots,
                                                                     *model.degree,
                                                                      model.points,
                                                                      t,
                                                                      times=times,
                                                                      axis=None)

                    model = SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

                elif isinstance(model, NurbsSurface):
                    Tu, Tv, pu, pv, P, W = insert_knot_nurbs_surface( *model.knots,
                                                                      *model.degree,
                                                                       model.points,
                                                                       model.weights,
                                                                       t,
                                                                       times=times,
                                                                       axis=None)

                    model = NurbsSurface(knots=(Tu, Tv),
                                         degree=(pu, pv),
                                         points=P,
                                         weights=W)

        d_timestamp['insert'] = insert_submit_time
    # ...

    # ... degree elevation
    if m > 0 and not( elevate_submit_time <= d_timestamp['elevate'] ) :
        m = int(m)

        if isinstance(model, SplineCurve):
            knots, degree, P = elevate_degree_bspline_curve( model.knots,
                                                             model.degree,
                                                             model.points,
                                                             m=m)

            model = SplineCurve(knots=knots,
                                degree=degree,
                                points=P)

        elif isinstance(model, NurbsCurve):
            knots, degree, P, W = elevate_degree_nurbs_curve( model.knots,
                                                              model.degree,
                                                              model.points,
                                                              model.weights,
                                                              m=m)

            model = NurbsCurve(knots=knots,
                               degree=degree,
                               points=P,
                               weights=W)

        elif isinstance(model, SplineSurface):
            Tu, Tv, pu, pv, P = elevate_degree_bspline_surface( *model.knots,
                                                               *model.degree,
                                                                model.points,
                                                                m=m)

            model = SplineSurface(knots=(Tu, Tv),
                                  degree=(pu, pv),
                                  points=P)

        elif isinstance(model, NurbsSurface):
            Tu, Tv, pu, pv, P, W = elevate_degree_nurbs_surface( *model.knots,
                                                                *model.degree,
                                                                 model.points,
                                                                 model.weights,
                                                                 m=m)

            model = NurbsSurface(knots=(Tu, Tv),
                                 degree=(pu, pv),
                                 points=P,
                                 weights=W)

        d_timestamp['elevate'] = elevate_submit_time
    # ...

    # ...subdivision
    if levels > 0 and not( subdivide_submit_time <= d_timestamp['subdivide'] ):
        levels = int(levels)

        for level in range(levels):
            grid = np.unique(model.knots)
            for a,b in zip(grid[:-1], grid[1:]):
                t = (a+b)/2.

                knots, degree, P = insert_knot_bspline_curve( model.knots,
                                                              model.degree,
                                                              model.points,
                                                              t, times=1 )

                model = SplineCurve(knots=knots, degree=degree, points=P)

        d_timestamp['subdivide'] = subdivide_submit_time
    # ...

    print('refinement done')
    return model


# =================================================================
@app.callback(
    Output("transformed_model", "data"),
    [Input("model", "value"),
     Input('button_transform', 'n_clicks_timestamp'),
     Input('translate_disp', 'value'),
     Input('translate_submit', 'n_clicks_timestamp'),
     Input('rotate_center', 'value'),
     Input('rotate_angle', 'value'),
     Input('rotate_submit', 'n_clicks_timestamp'),
     Input('homothetie_alpha', 'value'),
     Input('homothetie_center', 'value'),
     Input('homothetie_submit', 'n_clicks_timestamp')]
)
def apply_transform(models,
                    time_clicks,
                    translate_disp,
                    translate_submit_time,
                    rotate_center,
                    rotate_angle,
                    rotate_submit_time,
                    homothetie_alpha,
                    homothetie_center,
                    homothetie_submit_time):

    global d_timestamp

    if time_clicks <= d_timestamp['transform']:
        return None

    d_timestamp['transform'] = time_clicks

    if len(models) == 0:
        return None

    if len(models) > 1:
        return None

    name  = models[0]
    model = namespace[name]

    if not( translate_disp is '' ) and not( translate_submit_time <= d_timestamp['translate'] ):
        # ...
        try:
            displ = [float(i) for i in translate_disp.split(',')]

        except:
            raise ValueError('Cannot convert translate_disp')
        # ...

        displ = np.asarray(displ)

        if isinstance(model, SplineCurve):
            knots, P = translate_bspline_curve(model.knots,
                                               model.points,
                                               displ)

            model = SplineCurve(knots=knots,
                                degree=model.degree,
                                points=P)

        elif isinstance(model, NurbsCurve):
            knots, P, W = translate_nurbs_curve(model.knots,
                                                model.points,
                                                model.weights,
                                                displ)

            model = NurbsCurve(knots=knots,
                               degree=model.degree,
                               points=P,
                               weights=W)

    elif not( rotate_center is '' ) and not( rotate_angle is '' ) and not( rotate_submit_time <= d_timestamp['rotate'] ):
        # ...
        try:
            center = [float(i) for i in rotate_center.split(',')]
            center = np.asarray(center)

        except:
            raise ValueError('Cannot convert rotate_center')
        # ...

        # ...
        try:
            angle = float(rotate_angle)
            angle *= np.pi / 180

        except:
            raise ValueError('Cannot convert rotate_angle')
        # ...

        if isinstance(model, SplineCurve):
            knots, P = rotate_bspline_curve(model.knots,
                                            model.points,
                                            angle,
                                            center=center)

            model = SplineCurve(knots=knots,
                                degree=model.degree,
                                points=P)

        elif isinstance(model, NurbsCurve):
            knots, P, W = rotate_nurbs_curve(model.knots,
                                             model.points,
                                             model.weights,
                                             angle,
                                             center=center)

            model = NurbsCurve(knots=knots,
                               degree=model.degree,
                               points=P,
                               weights=W)

    elif not( homothetie_center is '' ) and not( homothetie_alpha is '' ) and not( homothetie_submit_time <= d_timestamp['homothetie'] ):
        # ...
        try:
            center = [float(i) for i in homothetie_center.split(',')]
            center = np.asarray(center)

        except:
            raise ValueError('Cannot convert homothetie_center')
        # ...

        # ...
        try:
            alpha = float(homothetie_alpha)

        except:
            raise ValueError('Cannot convert homothetie_alpha')
        # ...

        if isinstance(model, SplineCurve):
            knots, P = homothetic_bspline_curve(model.knots,
                                            model.points,
                                            alpha,
                                            center=center)

            model = SplineCurve(knots=knots,
                                degree=model.degree,
                                points=P)

        elif isinstance(model, NurbsCurve):
            knots, P, W = homothetic_nurbs_curve(model.knots,
                                             model.points,
                                             model.weights,
                                             alpha,
                                             center=center)

            model = NurbsCurve(knots=knots,
                               degree=model.degree,
                               points=P,
                               weights=W)


    print('transformation done')
    return model

# =================================================================
@app.callback(
    [Output("edited_model", "data"),
     Output("editor-Tu", "children"),
     Output("editor-Tv", "children"),
     Output("editor-Tw", "children"),
     Output("editor-degree", "children")],
    [Input("model", "value"),
     Input('button_edit', 'n_clicks_timestamp')]
)
def show_model(models,
               time_clicks):

    model = None
    Tu = None
    Tv = None
    Tw = None
    degree = None

    global d_timestamp

    if time_clicks <= d_timestamp['edit']:
        return model, Tu, Tv, Tw, degree

    d_timestamp['edit'] = time_clicks

    if len(models) == 0:
        return model, Tu, Tv, Tw, degree

    if len(models) > 1:
        return model, Tu, Tv, Tw, degree

    name  = models[0]
    model = namespace[name]

    knots  = ''
    degree = ''

    if isinstance(model, (SplineCurve, NurbsCurve)):
        Tu = ', '.join(str(i) for i in model.knots)
        Tu = '[{}]'.format(Tu)
        Tu = 'u = {}'.format(Tu)

        degree = str(model.degree)
        degree = 'degree = {}'.format(degree)

    elif isinstance(model, (SplineSurface, NurbsSurface)):
        Tu = ', '.join(str(i) for i in model.knots[0])
        Tu = '[{}]'.format(Tu)
        Tu = 'u = {}'.format(Tu)

        Tv = ', '.join(str(i) for i in model.knots[1])
        Tv = '[{}]'.format(Tv)
        Tv = 'v = {}'.format(Tv)

        degree = ', '.join(str(i) for i in model.degree)
        degree = '[{}]'.format(degree)
        degree = 'degrees = {}'.format(degree)

    print('show model done')

    return model, Tu, Tv, Tw, degree

# =================================================================
@app.callback(
    [Output("editor-table", "columns"),
     Output("editor-table", "data")],
    [Input('edited_model', 'data')]
)
def update_editor_data(data):
    if data is None:
        return [], []

    model = model_from_data(data)

    dim = model.points.shape[-1]
    xyz_names = ['x', 'y', 'z'][:dim]

    if isinstance(model, (SplineCurve, NurbsCurve)):
        names = ['i']
        data = []
        nu = model.points.shape[0]
        for i in range(nu):
            d = {'i': i}

            xyz = model.points[i,:]
            for k,v in zip(xyz_names, xyz):
                d[k] = v

            data += [OrderedDict(d)]

    elif isinstance(model, (SplineSurface, NurbsSurface)):
        names = ['i', 'j']
        data = []
        nu, nv = model.points.shape[:-1]
        for i in range(nu):
            for j in range(nv):
                d = {'i': i, 'j': j}

                xyz = model.points[i,j,:]
                for k,v in zip(xyz_names, xyz):
                    d[k] = v

                data += [OrderedDict(d)]

    names += xyz_names
    columns = [{"name": i, "id": i} for i in names]

    return columns, data

# =================================================================
@app.callback(
    [Output("model", "options"),
     Output("loaded_model", "clear_data"),
     Output("refined_model", "clear_data"),
     Output("transformed_model", "clear_data")],
    [Input('loaded_model', 'data'),
     Input('refined_model', 'data'),
     Input('transformed_model', 'data')]
)
def update_namespace(loaded_model, refined_model, transformed_model):
    data = None
    clear_load      = False
    clear_refine    = False
    clear_transform = False

    if not( loaded_model is None ):
        data = loaded_model
        clear_load = True

    elif not( refined_model is None ):
        data = refined_model
        clear_refine = True

    elif not( transformed_model is None ):
        data = transformed_model
        clear_transform = True

    if data is None:
        options = [{'label':name, 'value':name} for name in namespace.keys()]
        return options, clear_load, clear_refine, clear_transform

    # ...
    weights = None
    try:
        knots, degree, points = data

        points = np.asarray(points)

    except:
        try:
            knots, degree, points, weights = data

            points = np.asarray(points)
            weights = np.asarray(weights)

        except:
            raise ValueError('Could not retrieve data')
    # ...

    if isinstance(knots, (tuple, list)):
        knots = [np.asarray(T) for T in knots]

    if isinstance(degree, int):
        if weights is None:
            current_model = SplineCurve(knots=knots,
                                        degree=degree,
                                        points=points)

        else:
            current_model = NurbsCurve(knots=knots,
                                       degree=degree,
                                       points=points,
                                       weights=weights)

    elif len(degree) == 2:
        if weights is None:
            current_model = SplineSurface(knots=knots,
                                          degree=degree,
                                          points=points)

        else:
            current_model = NurbsSurface(knots=knots,
                                         degree=degree,
                                         points=points,
                                         weights=weights)


    # ...
    global model_id
    namespace['model_{}'.format(model_id)] = current_model
    model_id += 1
    # ...

    options = [{'label':name, 'value':name} for name in namespace.keys()]

    return options, clear_load, clear_refine, clear_transform


# =================================================================
@app.callback(
    Output("graph", "figure"),
    [Input("model", "value"),
     Input('control_polygon', 'on')]
)
def update_graph(models, control_polygon):

    if len(models) == 0:
        return {'data': []}

    # ...
    _models = []
    for model in models:
        if isinstance(model, str):
            _models += [namespace[model]]

        else:
            _models += [model]

    models = _models
    # ...

    # ...
    traces = []
    for model in models:
        if isinstance(model, (SplineCurve, NurbsCurve)):
            traces += plot_curve(model,
                                 nx=101,
                                 control_polygon=control_polygon)

        elif isinstance(model, (SplineSurface, NurbsSurface)):
            traces += plot_surface(model,
                                   Nu=101,
                                   Nv=101,
                                   control_polygon=control_polygon)

        else:

            raise TypeError('Only SplineCurve is available, given {}'.format(type(model)))

    # showlegend is True only for curves
    showlegend = len([i for i in models if not isinstance(i, SplineCurve)]) == 0

    layout = go.Layout( yaxis=dict(scaleanchor="x", scaleratio=1),
                        showlegend=showlegend )
    # ...

    return {'data': traces, 'layout': layout}


###########################################################
if __name__ == '__main__':

    app.run_server(debug=True)
