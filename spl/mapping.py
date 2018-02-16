# -*- coding: UTF-8 -*-
import numpy as np

from clapp.core.basic  import Basic

class Mapping(Basic):
    """
    A Class for Splines/NURBS mapping from SPL.
    """

    def __init__(self, filename=None, p_dim=None, geometry=None, other=None):
        """
        Mapping constructor.

        Creates a mapping object from a file or **caid** geomtry.

        filename : str
              name of a file to read.

        p_dim : int
              parametric dimension. Must be given if filename is provided.

        geoemtry : caid.cad_geometry.cad_geometry
              a **caid** geometry.

        other : clapp.spl.mapping.Mapping
            another mapping for compostion

        """

        # ... first we create the clapp object
        Basic.__init__(self)
        # ...

        # ... then we set the new id
        self._id = self.com.newID("mapping")
        # ...

        # ...
        self._filename = filename
        self._geometry = geometry

        if self.filename is not None:
            if p_dim is not None:
                self._p_dim = p_dim
                self.com.core.mapping_allocate(self.id, self.p_dim)
            else:
                raise ValueError("create mapping: p_dim must be given:")
            self.com.core.mapping_read_from_file(self.id, filename)
        elif self.geometry is not None:
            # TODO handle multi patchs
            nrb = self.geometry[0]
            d_dim          = len(nrb.shape)
            if d_dim == 1:
                p_u            = nrb.degree[0]
                n_u            = nrb.shape[0]
                knots_u        = nrb.knots[0]
                weights        = nrb.weights
                points         = np.zeros((d_dim, n_u))
                for d in range(0, d_dim):
                    points[d,:] = nrb.points[:,d]

                self._p_dim = d_dim
                self.com.core.mapping_allocate(self.id, d_dim)
                self.com.core.mapping_create_1d(self.id, \
                                                d_dim, \
                                                p_u, \
                                                n_u, \
                                                n_u+p_u+1, \
                                                knots_u, \
                                                points, weights)

            elif d_dim == 2:
                p_u            = nrb.degree[0]
                p_v            = nrb.degree[1]
                n_u            = nrb.shape[0]
                n_v            = nrb.shape[1]
                knots_u        = nrb.knots[0]
                knots_v        = nrb.knots[1]
                weights        = nrb.weights
                points         = np.zeros((d_dim, n_u, n_v))
                for d in range(0, d_dim):
                    points[d,:,:] = nrb.points[:,:,d]

                self._p_dim = d_dim
                self.com.core.mapping_allocate(self.id, d_dim)

                if other is None:
                    self.com.core.mapping_create_2d_0(self.id, \
                                                d_dim, \
                                                p_u, p_v, \
                                                n_u, n_v, \
                                                n_u+p_u+1, n_v+p_v+1, \
                                                knots_u, knots_v, \
                                                points, weights)
                else:
                    self.com.core.mapping_create_2d_1(self.id, \
                                                d_dim, \
                                                p_u, p_v, \
                                                n_u, n_v, \
                                                n_u+p_u+1, n_v+p_v+1, \
                                                knots_u, knots_v, \
                                                points, weights,\
                                                other.id)

            elif d_dim == 3:
                p_u            = nrb.degree[0]
                p_v            = nrb.degree[1]
                p_w            = nrb.degree[2]
                n_u            = nrb.shape[0]
                n_v            = nrb.shape[1]
                n_w            = nrb.shape[2]
                knots_u        = nrb.knots[0]
                knots_v        = nrb.knots[1]
                knots_w        = nrb.knots[2]
                weights        = nrb.weights
                points         = np.zeros((d_dim, n_u, n_v, n_w))
                for d in range(0, d_dim):
                    points[d,:,:,:] = nrb.points[:,:,:,d]

                self._p_dim = d_dim
                self.com.core.mapping_allocate(self.id, d_dim)
                self.com.core.mapping_create_3d(self.id, \
                                                d_dim, \
                                                p_u, p_v, p_w, \
                                                n_u, n_v, n_w, \
                                                n_u+p_u+1, n_v+p_v+1, n_w+p_w+1, \
                                                knots_u, knots_v, knots_w, \
                                                points, weights)

        else:
            raise NotImplemented("create mapping not yet implemented ")
        # ...

    def __del__(self):
        """
        destroys the current object
        """
        self.com.core.mapping_free(self.id)
        self.com.freeID("mapping", self.id)
        self._id = None

    @property
    def p_dim(self):
        """
        Returns the parametric dimension.
        """
        return self._p_dim

    @property
    def filename(self):
        """
        Returns the associated filename, used at the creation.
        """
        return self._filename

    @property
    def geometry(self):
        """
        Returns the **caid** geometry used at the creation.
        """
        return self._geometry

    @property
    def d_dim(self):
        """
        Returns the manifold dimension.
        """
        return self.com.core.mapping_get_d_dim(self.id)

    def __str__(self):
        """
        prints info about this object.
        """
        self.com.core.mapping_print_info(self.id)
        line = ""
        for d in self.infoData:
            line += str(d + " : " + str(self.infoData[d]))
            line += "\n"
        return line

    def export(self, filename):
        """
        Exports the mapping to a file.

        filename : str
                name of the file where to store the mapping.
        """
        _fmt = 0
        self.com.core.mapping_export(self.id, filename, _fmt)

    def to_cad_geometry(self):
        """
        converts a mpping to a caid.cad_geoemtry class.
        """
        # TODO no use a tmp file, but pass all the arguments from fortran
        _filename = "tmp_mapping.nml"
        self.export(_filename)
        from caid.cad_geometry import cad_geometry
        geo = cad_geometry(_filename)
        return geo

    def evaluate(self, u=None, v=None, w=None):
        """
        evaluates the mapping over 1d arrays.

        u : list or numpy.array
          a list of numpy array containing the 1d sites for the first parametric dimension

        v : list or numpy.array
          a list of numpy array containing the 1d sites for the second parametric dimension

        w : list or numpy.array
          a list of numpy array containing the 1d sites for the third parametric dimension

        """
        if self.p_dim == 1:
            if u is not None:
                u = np.asarray(u)
                n_points_u = u.shape[0]
                return self.com.core.mapping_evaluate_1d(self.id, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         u)
            else:
                print("You must give sites for evaluation.")
        elif self.p_dim == 2:
            if (u is not None) and (v is not None):
                u = np.asarray(u)
                v = np.asarray(v)
                n_points_u = u.shape[0]
                n_points_v = v.shape[0]
                return self.com.core.mapping_evaluate_2d(self.id, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         n_points_v, \
                                                         u, v)
            else:
                print("You must give sites for evaluation.")
        elif self.p_dim == 3:
            if (u is not None) and (v is not None) and (w is not None):
                u = np.asarray(u)
                v = np.asarray(v)
                w = np.asarray(w)
                n_points_u = u.shape[0]
                n_points_v = v.shape[0]
                n_points_w = w.shape[0]
                return self.com.core.mapping_evaluate_3d(self.id, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         n_points_v, \
                                                         n_points_w, \
                                                         u, v, w)
            else:
                print("You must give sites for evaluation.")
        else:
            print("evaluate mapping not yet implemented")

    def evaluate_deriv(self, u=None, v=None, w=None):
        # TODO: to test
        """
        evaluates derivatives of the mapping over 1d arrays.

        u : list or numpy.array
          a list of numpy array containing the 1d sites for the first parametric dimension

        v : list or numpy.array
          a list of numpy array containing the 1d sites for the second parametric dimension

        w : list or numpy.array
          a list of numpy array containing the 1d sites for the third parametric dimension

        """
        if self.p_dim == 1:
            if u is not None:
                u = np.asarray(u)
                n_points_u = u.shape[0]
                return self.com.core.mapping_evaluate_deriv_1_1d(self.id, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         u)
            else:
                print("You must give sites for evaluation.")
        elif self.p_dim == 2:
            if (u is not None) and (v is not None):
                u = np.asarray(u)
                v = np.asarray(v)
                n_points_u = u.shape[0]
                n_points_v = v.shape[0]
                n_total_deriv = 2
                return self.com.core.mapping_evaluate_deriv_1_2d(self.id, \
                                                         n_total_deriv, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         n_points_v, \
                                                         u, v)
            else:
                print("You must give sites for evaluation.")
        elif self.p_dim == 3:
            if (u is not None) and (v is not None) and (w is not None):
                u = np.asarray(u)
                v = np.asarray(v)
                w = np.asarray(w)
                n_points_u = u.shape[0]
                n_points_v = v.shape[0]
                n_points_w = w.shape[0]
                n_total_deriv = 3
                return self.com.core.mapping_evaluate_deriv_1_3d(self.id, \
                                                         n_total_deriv, \
                                                         self.d_dim, \
                                                         n_points_u, \
                                                         n_points_v, \
                                                         n_points_w, \
                                                         u, v, w)
            else:
                print("You must give sites for evaluation.")
        else:
            print("evaluate_deriv mapping not yet implemented")

    def compute_aerea(self, space=None, discretization=None, verbose=False):
        """
        computes the aera of the mapped domain.

        space : clapp.disco.space.Space
             A Finite Elements Space.
        discretization : clapp.disco.parameters.bspline
             The discretization parameters.
        verbose : bool
             If true, the **Fortran** assembler will print some info.
        """

        if space is  None:
            if discretization_params is not None:
                _context = Context(dirname="input", \
                              discretization_params=discretization)
                _trial_space = Space(context=context, type_space="h1")
                _test_space  = Space(context=context, type_space="h1")
                _ddm_params  = _context.ddm_params
            else:
                raise ValueError("> compute_aerea: expecting more parmeters")
        else:
            _trial_space = space
            _test_space  = space
            _ddm_params  = space.context.ddm_params
        # ...

        from clapp.fema.assembler      import Assembler
        from clapp.plaf.vector         import Vector
        from clapp.disco.field         import Field

        # ...
        norms = Vector(n_size=1, n_blocks=1)
        phi   = Field(_trial_space, name="phi")
        x = np.ones(phi.n_size)
        phi.set(x)
        # ...

        # ...
        assembler = Assembler(spaces =  [_test_space, _trial_space], \
                              fields = [phi], \
                              mapping = self, \
                              norms=norms, \
                              ddm_parameters = _ddm_params, \
                              enable_sqrt_norms = False, \
                              verbose = verbose)
        # ...

        # ...
        assembler.set_fields_evaluation()
        assembler.set_norms_evaluation()
        assembler.assemble()

        x = norms.get()
        return x
        # ...


