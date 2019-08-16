# -*- coding: utf-8 -*-
from collections import namedtuple

SplineCurve   = namedtuple('SplineCurve',   'knots, degree, points')
SplineSurface = namedtuple('SplineSurface', 'knots, degree, points')
SplineVolume  = namedtuple('SplineVolume',  'knots, degree, points')
NurbsCurve    = namedtuple('NurbsCurve',    'knots, degree, points, weights')
NurbsSurface  = namedtuple('NurbsSurface',  'knots, degree, points, weights')
NurbsVolume   = namedtuple('NurbsVolume',   'knots, degree, points, weights')
