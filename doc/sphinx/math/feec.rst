Exterior Algebra 
****************

.. sectionauthor:: A. Ratnani

Let :math:`V` be a real vector space of dimension :math:`n`. 

.. topic:: Definition, Alternating algebraic forms:

  For each :math:`k`, we define :math:`\Alt^k V` as the space of alternating :math:`k`-linear maps :math:`V \times \cdots \times V \rightarrow \mathbb{R}`.

.. note:: * :math:`\Alt^0 = \mathbb{R}`,
          * :math:`\Alt^1 = V^{*}` is the dual space of :math:`V` (the space of covectors)


.. topic:: Definition, Exterior product:

  For :math:`\omega \in \Alt^j` and :math:`\eta \in \Alt^k`, their exterior (wedge) product is given by:

  .. math::

    (\omega \wedge \eta ) (v_1, \cdots, v_{j+k}) = \sum_{\sigma} (\mathrm{sign}~ \sigma) 
    \omega (v_{\sigma(1)}, \cdots, v_{\sigma(j)}) 
    \eta (v_{\sigma(j+1)}, \cdots, v_{\sigma(j+k)})

  for all :math:`v_i \in V`. Where the sum is over all permutations :math:`\sigma` of :math:`\{ 1,\cdots,j+k \}`, 
  for which :math:`\sigma(1)< \cdots <\sigma(j)` and :math:`\sigma(j+1)< \cdots <\sigma(j+k)`.

.. note:: * The exterior product is **bilinear**, **associative**,
          * **anti-commutative**: :math:`\eta \wedge \omega = (-1)^{jk} \omega \wedge \eta` for all :math:`\omega \in \Alt^j` and :math:`\eta \in \Alt^k`.


.. topic:: Definition, Grassmann Algebra:

  Grassmann Algebra is defined by:

  .. math::

    \Alt V := \bigoplus_k \Alt^k V

  This is a **anti-commutative graded algebra**. Also called **Exterior Algebra** of :math:`V^{*}`

In the case of :math:`V=\mathbb{R}^n`, we have:

* :math:`\Alt V^0 \sim \mathbb{R}`,
* :math:`\Alt V^1 \sim \mathbb{R}^n`,
* :math:`\Alt V^{n-1} \sim \mathbb{R}^n`, using Riesz representation theorem,
* :math:`\Alt V^n \sim \mathbb{R}`, using the map :math:`v \longmapsto \det(v,v_1,\cdots,v_{n-1})`.


Basis
^^^^^

Let :math:`v_1,\cdots,v_n` be a basis of :math:`V` and :math:`\mu_1,\cdots,\mu_n` the associated dual basis for :math:`V^*` (:math:`\mu_i(v_j) = \delta_{ij}`). 

For any increasing permutations :math:`\sigma, \rho : \{ 1,\cdots,k \} \longrightarrow \{ 1,\cdots,n \}`, we have:

.. math::

  \mu_{\sigma(1)} \wedge \cdots \wedge \mu_{\sigma(k)} (v_{\rho(1)}, \cdots, v_{\rho(k)}) = \chi_{\sigma,\rho}

thus the :math:`\binom {n}{k}` algebraic :math:`k`-forms :math:`\mu_{\sigma(1)} \wedge \cdots \wedge \mu_{\sigma(k)}`, 
form a basis for :math:`\Alt^k V` and :math:`\dim \Alt^k V = \binom {n}{k}`.


.. topic:: Definition, Interior product:

  Let :math:`\omega` be a :math:`k`-form, and :math:`v \in V`. The **interior product** of :math:`\omega` and :math:`v` is the :math:`(k-1)`-form  :math:`\omega \lrcorner v` defined by:

  .. math::
    \omega \lrcorner v (v_1,\cdots,v_{k-1}) = \omega (v,v_1,\cdots,v_{k-1})


* We have for :math:`\omega \in \Alt^k V`, :math:`\eta \in \Alt^l V` and :math:`v \in V`: 

.. math::

  (\omega \wedge \eta) \lrcorner v = (\omega \lrcorner v)\wedge \eta + (-1)^k \omega \wedge (\eta \lrcorner v)


.. topic:: Definition, Inner product:

  If :math:`V` is has an inner product, then :math:`\Alt^k V` is endowed with an inner product given by:

  .. math::

    (\omega , \eta) = \sum_{\rho} \omega (v_{\rho(1)}, \cdots, v_{\rho(k)}) \eta (v_{\rho(1)}, \cdots, v_{\rho(k)}), ~~~\forall \omega, \eta \in \Alt^k V. 

  where the sum is over increasing sequences :math:`\rho : \{ 1,\cdots,k \} \longrightarrow \{ 1,\cdots,n \}`, and :math:`v_1, \cdots,v_n` is any orthonormal basis.


Orientation and Volume form
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: add Orientation and Volume form 



.. topic:: Definition, Pullback:

  A linear transformation of vector spaces :math:`L: V \rightarrow W` induces a transformation 
  :math:`L^{*}: \Alt W \rightarrow \Alt V`, called the **pullback**, and given by:

  .. math::

    L^{*} \omega (v_{1}, \cdots, v_{k}) = \omega (L v_{1}, \cdots, L v_{k}),~~~~~ \forall \omega \in \Alt^k W,~~~ v_{1}, \cdots, v_{k} \in V


* The pullback acts **contravariantly**: if :math:`U \xrightarrow{~K~} V \xrightarrow{~L~} W` then,

  .. math::

    \Alt W \xrightarrow{~K^{*}~} \Alt V \xrightarrow{~L^{*}~} \Alt U 

* :math:`L^{*} (\omega \wedge \eta) = L^{*} \omega \wedge L^{*} \eta`


Let V be a subspace of W. For the inclusion :math:`\imath_V : V \longrightarrow W`, we can define its pullback :math:`\imath_V^{*}`: 
this is a **surjection** of :math:`\Alt W` onto :math:`\Alt V`.

If W has an inner product and :math:`\pi_V : W \longrightarrow V` is the orthogonal projection. We can define its pullback :math:`\pi_V^{*}` : 
this an **injection** of :math:`\Alt V` onto :math:`\Alt W`.

Let us consider the composition :  :math:`W` \shortstack{:math:`\pi_V` \\ :math:`\longrightarrow`} :math:`V` \shortstack{:math:`\imath_V` \\ :math:`\longrightarrow`} :math:`W`, and its pullback :math:`\pi_V^* \imath_V^*`.


.. topic:: Definition, The tangential and normal parts:

  * :math:`\pi_V^* \imath_V^*` associates for each :math:`\omega \in \Alt^k` its **tangential** part :math:`\omega_{\parallel}` with respect to :math:`V` :

  .. math::

    (\pi_V^* \imath_V^* \omega) (v_1,\cdots,v_k) = \omega (\pi_V v_1, \cdots, \pi_V v_k), ~~~~~\forall v_1,\cdots,v_k \in W.

  * :math:`\omega - \pi_V^* \imath_V^* \omega` associates for each :math:`\omega \in \Alt^k` its **normal** part :math:`\omega_{\perp}` with respect to :math:`V`.


The **tangential part** of :math:`\omega` vanishes if and only if the image of :math:`\omega` in :math:`\Alt^k V` vanishes.

Let :math:`V` be an oriented inner product space, with volume form :math:`\mbox{vol}`. Let :math:`\omega \in \Alt^k V`. 
We can define a new linear map :math:`L_{\omega}` as the composition of :math:`\Alt^{n-k} V \longrightarrow \Alt^n V` such as:

.. math::

  \mu \longmapsto \omega \wedge \mu

and the canonical isomorphism of :math:`\Alt^n V` onto :math:`\mathbb{R}`, and using the Riesz representation theorem, 
there exists an element :math:`\star \omega \in \Alt^{n-k} V` such that : :math:`L_{\omega} (\mu) = (\star \omega , \mu)`, *i.e.*:

.. math::

  \omega \wedge \mu = (\star \omega , \mu) \mbox{vol}, ~~~\omega \in \Alt^{k}, ~\mu \in \Alt^{n-k}


.. topic:: Definition, The Hodge star operation:

  The linear map which maps :math:`\Alt^k V` onto :math:`\Alt^{n-k} V` :math:`\omega \longmapsto \star \omega` is called the **Hodge star** operator.


* If :math:`e_1,\cdots,e_n` is any positively oriented orthonormal basis, and :math:`\sigma` a permutation, we have

.. math::

  \omega(e_{\sigma(1)}, \cdots, e_{\sigma(k)}) = (\mathrm{sign} \sigma) \star \omega(e_{\sigma(k+1)}, \cdots, e_{\sigma(n)})

* :math:`\star \star \omega = (-1)^{k(n-k)} \omega, ~~~\forall \omega \in \Alt^k V`, thus the Hodge star is an **isometry**.
* :math:`(\star \omega)_{\parallel} = \star (\omega_{\perp})` and :math:`(\star \omega)_{\perp} = \star (\omega_{\parallel})`
* the image of :math:`\star \omega` in :math:`\Alt^k V` vanishes if and only if :math:`\omega_{\perp}` vanishes.


.. math::

  \begin{tabular}{|c|l|}
    \hline
   $\Alt^0 \mathbb{R}^3 \cong \mathbb{R}$ &  $c \leftrightarrow c$ \\
  %   \hline
   $\Alt^1 \mathbb{R}^3 \cong \mathbb{R}^3$ & $u_1 \diff x_1 + u_2 \diff x_2 + u_3 \diff x_3 \leftrightarrow u$ \\
  %    \hline
   $\Alt^2 \mathbb{R}^3 \cong \mathbb{R}^3$ & $u_3 \diff x_1 \wedge \diff x_2 - u_2 \diff x_1 \wedge \diff x_3 + u_1 \diff x_2 \wedge \diff x_3 +  \leftrightarrow u$ \\
  %   \hline
   $\Alt^3 \mathbb{R}^3 \cong \mathbb{R}$ & $c \diff x_1 \wedge \diff x_2 \wedge \diff x_3 \leftrightarrow c$  \\
    \hline
  \end{tabular}
..   \caption{Correspondence}

.. math::

  \begin{tabular}{|c|l|}
    \hline
   $ \wedge : \Alt^1 \mathbb{R}^3 \times \Alt^1 \mathbb{R}^3 \longrightarrow \Alt^2 \mathbb{R}^3$ 
  &  $\times : \mathbb{R}^3 \times \mathbb{R}^3 \longrightarrow \mathbb{R}^3$ 
  \\
   $ \wedge : \Alt^1 \mathbb{R}^3 \times \Alt^2 \mathbb{R}^3 \longrightarrow \Alt^3 \mathbb{R}^3$ 
  &  $\cdot : \mathbb{R}^3 \times \mathbb{R}^3 \longrightarrow \mathbb{R}$  
  \\
    \hline
  \end{tabular}
..   \caption{Exterior product}

.. math::

  \begin{tabular}{|c|l|}
    \hline
   $ L^* : \Alt^0 \mathbb{R}^3 \longrightarrow \Alt^0 \mathbb{R}^3 $ & $\id : \mathbb{R} \longrightarrow \mathbb{R}$
  \\
   $ L^* : \Alt^1 \mathbb{R}^3 \longrightarrow \Alt^1 \mathbb{R}^3 $ & $L^T : \mathbb{R}^3 \longrightarrow \mathbb{R}^3$
  \\
   $ L^* : \Alt^2 \mathbb{R}^3 \longrightarrow \Alt^2 \mathbb{R}^3 $ & $(\det L )L^{-1} : \mathbb{R}^3 \longrightarrow \mathbb{R}^3$
  \\
   $ L^* : \Alt^3 \mathbb{R}^3 \longrightarrow \Alt^3 \mathbb{R}^3 $ & $(\det L) : \mathbb{R} \longrightarrow \mathbb{R}$ ~~~($c \longmapsto c \det L$)
  \\
    \hline
  \end{tabular}
..   \caption{Pullback by a linear map L :$\mathbb{R}^3 \longrightarrow \mathbb{R}^3$}

.. math::

  \begin{tabular}{|c|l|}
    \hline
   $ \lrcorner v : \Alt^1 \mathbb{R}^3 \longrightarrow \Alt^0 \mathbb{R}^3 $ & $v \cdot : \mathbb{R}^3 \longrightarrow \mathbb{R}$
  \\
   $ \lrcorner v : \Alt^2 \mathbb{R}^3 \longrightarrow \Alt^1 \mathbb{R}^3 $ & $v \times : \mathbb{R}^3 \longrightarrow \mathbb{R}^3$
  \\
   $ \lrcorner v : \Alt^3 \mathbb{R}^3 \longrightarrow \Alt^2 \mathbb{R}^3 $ & $v : \mathbb{R} \longrightarrow \mathbb{R}^3$ ~~~($c \longmapsto c v$)
  \\
    \hline
  \end{tabular}
..   \caption{Interior product with a vector $v \in \mathbb{R}^3$}

.. math::

  \begin{tabular}{|c|l|}
    \hline
   inner product on $\Alt^k \mathbb{R}^3$ induced  & dot product on $\mathbb{R}$ and $\mathbb{R}^3$
  \\
   by dot product on $\mathbb{R}^3$ & 
  \\
   $\volume = \diff x_1 \wedge \diff x_2 \wedge \diff x_3$ & $(v_1,v_2,v_3) \longmapsto \det(v_1|v_2|v_3)$
  \\
    \hline
  \end{tabular}
..   \caption{Inner product and volume form}

.. math::

  \begin{tabular}{|c|l|}
    \hline
   $ \star : \Alt^0 \mathbb{R}^3 \longrightarrow \Alt^3 \mathbb{R}^3 $ & $\id : \mathbb{R} \longrightarrow \mathbb{R}$
  \\
   $ \star : \Alt^1 \mathbb{R}^3 \longrightarrow \Alt^2 \mathbb{R}^3 $ & $\id : \mathbb{R}^3 \longrightarrow \mathbb{R}^3$
  \\
    \hline
  \end{tabular}
..   \caption{Hodge star}



Exterior Calculus on manifolds and Differential forms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\Omega` be a smooth manifold, of dimension :math:`n`.

*  :math:`\forall x \in \Omega` we denote by :math:`T_x \Omega` the tangent space. This is a vector space of dimension :math:`n`,
*  tangent bundle :math:`\{ (x,v), ~~ x \in \Omega, v \in T_x \Omega \}`,
*  Applying the exterior algebra to the tangent spaces, we obtain the exterior forms bundle, whose elements are pairs :math:`(x,\mu)` with :math:`x \in \Omega` and :math:`\mu \in \Alt^k T_x \Omega`.
*  a **differential** :math:`k`-form :math:`\omega` is a section of this bundle. This is a map which associates to each :math:`x \in \Omega` an element :math:`\omega_x \in \Alt^k T_x \Omega`,
*  if the map :math:`\mathcal{L}_{\omega}^k : x \longmapsto \omega_x (v_1(x), \cdots, v_k(x))` is smooth (whenever :math:`v_i` are smooth), we say that :math:`\omega` is a smooth differential :math:`k`-form,
*  we define :math:`\Lambda^k(\Omega)` the space of all smooth :math:`k`-forms on :math:`\Omega`,
*  :math:`\Lambda^0(\Omega) = \mathcal{C}^{\infty}(\Omega)`,
*  if the map :math:`\mathcal{L}_{\omega}^k` is :math:`\mathcal{C}^{m}(\Omega)`, we define differential :math:`k`-forms with less smoothness :math:`\mathcal{C}^{m} \Lambda^k (\Omega)`.

Let :math:`\Omega` be a smooth manifold, of dimension :math:`n`.

.. topic:: Exterior product:

  if :math:`\omega \in \Lambda^k(\Omega)` and :math:`\eta \in \Lambda^j(\Omega)`, we may define :math:`\omega \wedge \eta` as  :math:`(\omega \wedge \eta)_x = \omega_x \wedge \eta_x` and the Grassmann algebra :math:`\Lambda(\Omega) := \bigoplus_k \Lambda^k(\Omega)`

Differential forms can be differentiated and integrated, without recourse to any additional structure, such as a metric or a measure.

.. topic:: Exterior differentiation:

   For each :math:`\omega \in \Lambda^k(\Omega)`, can define the :math:`(k+1)`-form :math:`\diff \omega \in \Lambda^{k+1}(\Omega)`, such as:

  .. math::

    \diff\omega_x(v_1,\cdots,v_{k+1}) = \sum_{j=1}^{k+1} (-1)^j \partial_{v_j} \omega_x(v_1,\cdots,\hat{v_j},\cdots,v_{k+1})

  where the hat is used to indicated a suppressed argument.

  This defines a graded linear operator of degree :math:`+1`, of :math:`\Lambda(\Omega)` onto :math:`\Lambda(\Omega)`.

We have the following properties:

*  :math:`\diff \circ \diff = 0`
*  :math:`\diff (\omega \wedge \eta) = \diff \omega \wedge \eta + (-1)^k \omega \wedge \diff \eta, ~~\forall \omega \in \Lambda^k(\Omega), \eta \in \Lambda^j(\Omega)`,
*  (Pullback) let :math:`\phi` be a smooth map of :math:`\Omega` onto :math:`\Omega^{\prime}`. Then :math:`\phi^*(\omega \wedge \eta) = \phi^*(\omega) \wedge \phi^*(\eta)` and :math:`\phi^* (\diff \omega) = \diff (\phi^* \omega)`,
*  (Interior product) the interior product of a differential :math:`k`-form :math:`\omega` with a vector field :math:`v`, 
*  we obtain a :math:`(k-1)`-form by : :math:`(\omega \lrcorner v)_x := \omega_x \lrcorner v_x`,
*  (Trace operator) the pullback :math:`i_{\partial \Omega}^*` of :math:`i_{\partial \Omega}` is the trace operator :math:`\trace`


.. topic:: Integration:

  * If :math:`f` is an oriented, piecewise smooth :math:`k`-dimensional submanifold of :math:`\Omega`, and :math:`\omega` is a continuous :math:`k`-form, then th integral :math:`\int_f \omega` is well defined :

    * [0-forms] can be evaluated at points,
    * [1-forms] can be integrated over directed curves,
    * [2-forms] can be integrated over directed surfaces,

  *  (Inner product) The :math:`L^2`-inner product of two differential :math:`k`-forms on an oriented Riemannian manifold :math:`\Omega` is defined as :

  .. math::

    (\omega,\eta)_{L^2 \Lambda^k} = \int_{\Omega} (\omega_x,\eta_x) \volume = \int \omega \wedge \star \eta

  The completion of :math:`\Lambda^k(\Omega)` in the corresponding norm defines the Hilbert space :math:`L^2 \Lambda^k(\Omega)`.

We have the following results:

* (Integration) if :math:`\phi` is an orientation-preserving diffeomorphism, then 

.. math::

  \int_{\Omega} \phi^* \omega = \int_{\Omega^{\prime}} \omega, ~~~ \forall \omega \in \Lambda^n(\Omega^{\prime})

.. topic:: Theorem, Stokes theorem:

  If :math:`\Omega` is an oriented :math:`n`-manifold with boundary :math:`\partial \Omega`, then

  .. math::

    \int_{\Omega} \diff \omega = \int_{\partial \Omega} \trace \omega, ~~~ \forall \omega \in \Lambda^{n-1}(\Omega)

.. topic:: Theorem, Integration by parts:

  If :math:`\Omega` is an oriented :math:`n`-manifold with boundary :math:`\partial \Omega`, then

  .. math::

    \int_{\Omega} \diff \omega \wedge \eta = (-1)^{k-1} \int_{\Omega} \omega \wedge \diff \eta + \int_{\partial \Omega} \trace \omega \wedge \trace \eta, ~~~ \forall \omega \in \Lambda^{k}(\Omega), \eta \in \Lambda^{n-k-1}(\Omega)



Sobolev spaces of differential forms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As for the classical case, we can define the Sobolev spaces as:

*  :math:`H^s \Lambda^k(\Omega)` is the space of differential :math:`k`-forms such that :math:`\mathcal{L}_{\omega}^k \in H^s(\Omega)`.
*  :math:`H \Lambda^k(\Omega) = \{ \omega \in L^2 \Lambda^k(\Omega),~~ \diff \omega \in L^2 \Lambda^{k+1}(\Omega) \}`. The associated norm is :

  .. math::

    \| \omega \|_{H \Lambda^k}^2 = \| \omega \|_{H \Lambda}^2 := \| \omega \|_{L^2 \Lambda^k}^2 + \| \diff \omega \|_{L^2 \Lambda^{k+1}}^2

*  :math:`H \Lambda^{0}(\Omega)` coincides with :math:`H^1 \Lambda^{0}(\Omega)`,
*  :math:`H \Lambda^{n}(\Omega)` coincides with :math:`L^2 \Lambda^{n}(\Omega)`,
*  for :math:`0 < k < n`, we have :math:`H^1 \Lambda^k(\Omega) \subset H \Lambda^k(\Omega) \subset L^2 \Lambda^k(\Omega)`, strictly.

.. math::

  \begin{tabular}{|c|c c c c c|}
    \hline
   $k$ & $\Lambda^k$ & $H \Lambda^k$ & $\diff \omega$ & $\int_f \omega$ & $\kappa \omega$ 
  \\
   \hline
  & & & & & \\
   0 & $\mathcal{C}^{\infty}$ & $H^1$ & $\nabla \omega$ & $\omega(f)$ & $0$
  \\
   1 & $\mathcal{C}^{\infty}(\mathbb{R}^3)$ & $H(\rots,\mathbb{R}^3)$ & $\rots \omega$ & $\int_f \omega \cdot t \diff \mathcal{H}_1$ & $x \longmapsto x \cdot \omega(x)$
  \\
   2 & $\mathcal{C}^{\infty}(\mathbb{R}^3)$ & $H(\divs, \mathbb{R}^3)$ & $\divs \omega$ & $\int_f \omega \cdot n \diff \mathcal{H}_2$ & $x \longmapsto x \times \omega(x)$
  \\
   3 & $\mathcal{C}^{\infty}$ & $L^2$ & $0$ & $\int_f \omega \diff \mathcal{H}_3$ & $x \longmapsto x \omega(x)$
  \\
  & & & & & \\
    \hline
  \end{tabular}
..   \caption{Correspondences between differential forms in $3$D, and scalar/vector fields.}

Cohomology and De Rham Complex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The De Rham complex is the sequence of spaces and mappings

.. math::

  0 \xrightarrow{\quad} \Lambda^0(\Omega)  \xrightarrow{~\diff~}  \Lambda^1(\Omega)  \xrightarrow{~\diff~}   \cdots  \xrightarrow{~\diff~}  \Lambda^n(\Omega)  \xrightarrow{\quad} 0  

Since, :math:`\diff \circ \diff = 0`, we have 

.. math::

  \mathcal{R}(\diff : \Lambda^{k-1}(\Omega) \longrightarrow \Lambda^k(\Omega)) \subset \mathcal{N}(\diff : \Lambda^{k}(\Omega) \longrightarrow \Lambda^{k+1}(\Omega))

If :math:`\Omega` is an oriented Riemannian manifold, we have the following cohomology:

.. math::

  0 \xrightarrow{\quad} H \Lambda^0(\Omega)  \xrightarrow{~\diff~}  H \Lambda^1(\Omega)  \xrightarrow{~\diff~}   \cdots  \xrightarrow{~\diff~}  H \Lambda^n(\Omega)  \xrightarrow{\quad} 0  

The *coderivative operator* :math:`\delta : \Lambda^{k}(\Omega) \longrightarrow \Lambda^{k-1}(\Omega)` is defined as:

.. math::

  \star \delta \omega = (-1)^k \diff \star \omega,~~~ \omega \in \Lambda^k(\Omega)

*  we have 

  .. math::

    (\diff \omega , \eta ) = (\omega , \delta \eta )  + \int_{\partial \Omega} \trace \omega \wedge \trace \eta,  ~~~ \forall \omega \in \Lambda^{k}(\Omega), \eta \in \Lambda^{k+1}(\Omega),

*  :math:`\delta` is a graded linear operator of degree :math:`-1`.
*  :math:`\delta` is the formal adjoint of :math:`\diff` whenever :math:`\omega` or :math:`\eta` vanishes near the boundary.
*  we define the spaces 

  .. math::

    H^* \Lambda^k(\Omega) = \{ \omega \in L^2 \Lambda^k(\Omega),~~ \delta \omega \in L^2 \Lambda^{k-1}(\Omega) \}.

  we have :math:`H^* \Lambda^k(\Omega) = \star H \Lambda^{n-k}(\Omega)`.

*  we obtain the dual complex

  .. math::

    0 \xleftarrow{\quad} H^* \Lambda^0(\Omega)  \xleftarrow{~\delta~}  H^* \Lambda^1(\Omega)  \xleftarrow{~\delta~}   \cdots  \xleftarrow{~\delta~}  H^* \Lambda^n(\Omega)  \xleftarrow{\quad} 0  


Cohomology with boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\Lambda_0^k(\Omega)` be the subspace of :math:`\Lambda^k(\Omega)` of smooth :math:`k`-forms with compact support. We have :math:`\diff \Lambda_0^k \subset \Lambda_0^{k+1}`.

The De Rham complex with the compact support is 

.. math::

  0 \xrightarrow{\quad} \Lambda^0_0(\Omega)  \xrightarrow{~\diff~}  \Lambda^1_0(\Omega)  \xrightarrow{~\diff~}   \cdots  \xrightarrow{~\diff~}  \Lambda^n_0(\Omega)  \xrightarrow{\quad} 0  

Recall that the closure of :math:`\Lambda_0^k(\Omega)` in :math:`H \Lambda^k(\Omega)` is 

.. math::

  H_0 \Lambda^k(\Omega) = \{ \omega \in H \Lambda^k(\Omega),~~ \trace \omega =0\}.

The :math:`L^2` version of the last complex is 

.. math::

  0 \xrightarrow{\quad} H_0 \Lambda^0(\Omega)  \xrightarrow{~\diff~}  H_0 \Lambda^1(\Omega)  \xrightarrow{~\diff~}   \cdots  \xrightarrow{~\diff~}  H_0 \Lambda^n(\Omega)  \xrightarrow{\quad} 0  


.. topic:: Definition, Harmonic forms:

  The harmonic :math:`k`-forms are the differential :math:`k`-forms that verify the differential equations 

  .. math::

      \left\{
          \begin{aligned}
            \diff \omega &=& 0,\\
            \delta \omega &=& 0,\\
            \trace \star \omega &=& 0.\\
          \end{aligned}
        \right.

  this defines the following space,

  .. math::

    \mathfrak{H}^k (\Omega) = \{ \omega \in H \Lambda^k(\Omega) \cap H_0^* \Lambda^k(\Omega),~~\diff \omega = 0, \delta \omega = 0 \}

We can also define the following space,

.. math::

  \mathfrak{H}_0^k (\Omega) = \{ \omega \in H_0 \Lambda^k(\Omega) \cap H^* \Lambda^k(\Omega),~~\diff \omega = 0, \delta \omega = 0 \}

As we can see, :math:`\star \mathfrak{H}^k (\Omega) = \mathfrak{H}_0^{n-k} (\Omega)`.

.. topic:: Proposition, Poincaré duality:

  There is an isomorphism between the :math:`k` th De Rham cohomology space and the :math:`(n-k)` th cohomology space with boundary conditions.

Homological Algebra and Hilbert complexes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homological Algebra
___________________

*  A cochain complex is a sequence of vector spaces and linear maps

.. TODO
.. .. math::
.. 
..   \cdots \longrightarrow V_{k-1}~\mbox{\shortstack{:math:`\diff_{k-1}` \\ :math:`\longrightarrow`}}~V_{k} \mbox{\shortstack{:math:`\diff_k` \\ :math:`\longrightarrow`}}~V_{k+1}~\longrightarrow~\cdots,~~~~\mbox{with}~ \diff_{k+1} \circ \diff_k = 0.
.. 

*  :math:`k`-cocycles :math:`\mathfrak{Z}^k := \mathcal{N}(d_k)`,

*  :math:`k`-coboundaries :math:`\mathfrak{B}^k := \mathcal{R}(d_{k-1})`,

*  :math:`k`-cohomology :math:`\mathcal{H}^k(V) := \mathfrak{Z}^k / \mathfrak{B}^k`,

*  we say that the sequence is **exact**, if the **cohomology vanishes** (*i.e.* :math:`\forall~k,~~ \mathcal{H}^k(V) = \{0\}`),

*  Given two cochain complexes :math:`V,V^{\prime}`, a **cochain map** :math:`f =(f_k)` (such as :math:`\diff^{\prime}_k f_k = f_{k+1} \diff_k`)

  .. math::

    \begin{array}{ccccccccc}
    \cdots & \longrightarrow & V_{k-1} & \mbox{\shortstack{$\diff_{k-1}$ \\ $\longrightarrow$}} & V_{k} & \mbox{\shortstack{$\diff_k$ \\ $\longrightarrow$}} & V_{k+1} & \longrightarrow~\cdots \\
     & & \downarrow f_{k-1} & & \downarrow f_{k} &  & \downarrow f_{k+1} & & \\
    \cdots & \longrightarrow & V_{k-1}^{\prime} & \mbox{\shortstack{$\diff_{k-1}^{\prime}$ \\ $\longrightarrow$}} & V_{k}^{\prime} & \mbox{\shortstack{$\diff_k^{\prime}$ \\ $\longrightarrow$}} & V_{k+1}^{\prime} & \longrightarrow~\cdots 
    \end{array}
 
*  :math:`f_k` maps :math:`k`-cochains to :math:`k`-cochains and :math:`k`-coboundaries to :math:`k`-coboundaries, thus induces a map :math:`\mathcal{H}^k(f) : \mathcal{H}^k(V) \longrightarrow \mathcal{H}^k(V^{\prime})`.

Let :math:`V^{\prime} \subset V` be two cochain complexes,

* The inclusion :math:`\imath_V` is a cochain map and thus induces a map of cohomology :math:`\mathcal{H}^k(V^{\prime}) \longrightarrow \mathcal{H}^k(V)`,

* If there exists a cochain projection of :math:`V` onto :math:`V^{\prime}`, (this leads to :math:`\pi \circ \imath = \id_{V^{\prime}}`) so :math:`\mathcal{H}^k(\pi) \circ \mathcal{H}^k(\imath) = \id_{\mathcal{H}^k(V^{\prime})}`.

  .. math::

    \begin{array}{ccccccc}
    \cdots & \longrightarrow & V_{k-1} & \mbox{\shortstack{$\diff_{k-1}$ \\ $\longrightarrow$}} & V_{k} & \longrightarrow~\cdots \\
     & & \pi_{k-1} \downarrow \uparrow \imath & & \pi_{k} \downarrow \uparrow \imath & & \\
    \cdots & \longrightarrow & V_{k-1}^{\prime} & \mbox{\shortstack{$\diff_{k-1}$ \\ $\longrightarrow$}} & V_{k}^{\prime} & \longrightarrow~\cdots 
    \end{array}

Thus, :math:`\mathcal{H}^k(\imath)` is **injective** and :math:`\mathcal{H}^k(\pi)` is **surjective**. Hence, if one of the cohomology spaces :math:`\mathcal{H}^k(V)` vanishes, 
then so does :math:`\mathcal{H}^k(V^{\prime})`

Cycles and boundaries of the De Rham complex
____________________________________________

* :math:`k`-cocycles 

.. math::

  \mathfrak{Z}^k = \{ \omega \in H\Lambda^k(\Omega),~~ \diff \omega = 0 \}, ~~~ \mathfrak{Z}^{*k} = \{ \omega \in H^*\Lambda^k(\Omega),~~ \delta \omega = 0 \},

.. math::

  \mathfrak{Z}_0^k = \{ \omega \in H_0\Lambda^k(\Omega),~~ \diff \omega = 0 \}, ~~~ \mathfrak{Z}_0^{*k} = \{ \omega \in H_0^*\Lambda^k(\Omega),~~ \delta \omega = 0 \},

.. math::

* :math:`k`-coboundaries

.. math::

  \mathfrak{B}^k = \diff H\Lambda^{k-1}(\Omega), ~~~ \mathfrak{B}^{* k} = \delta \Lambda^{k+1}(\Omega),

.. math::

  \mathfrak{B}_0^k = \diff H_0\Lambda^{k-1}(\Omega), ~~~ \mathfrak{B}_0^{* k} = \delta \Lambda_0^{k+1}(\Omega),

* each of the spaces of cycles is closed in :math:`\mathcal{H} \Lambda^k(\Omega)` (:math:`\mathcal{H}^* \Lambda^k(\Omega)`), as well in :math:`L^2 \Lambda^k(\Omega)`.

* each of the spaces of boundaries is closed in :math:`L^2 \Lambda^k(\Omega)`.

* let :math:`\perp` denotes the orthogonal complement in :math:`L^2 \Lambda^k(\Omega)`,

.. math::

  \mathfrak{Z}^{k \perp} \subset \mathfrak{B}^{k \perp} = \mathfrak{Z}_0^{* k} , ~~~ \mathfrak{Z}^{* k \perp} \subset \mathfrak{B}^{* k \perp} = \mathfrak{Z}_0^{k}

.. math::

  \mathfrak{Z}_0^{k \perp} \subset \mathfrak{B}_0^{k \perp} = \mathfrak{Z}^{* k} , ~~~ \mathfrak{Z}_0^{* k \perp} \subset \mathfrak{B}_0^{* k \perp} = \mathfrak{Z}^{k}


The Hodge decomposition
_______________________

There are two Hodge decompositions, with different boundary conditions,

1.

  .. math::

    L^2 \Lambda^k(\Omega) 
    = 
    \underbrace{\mathfrak{B}^{k}}_{\mathfrak{Z}_0^{* k\perp}} 
    \oplus 
    \underbrace{\mathfrak{H}^{k} 
    \oplus 
    \mathfrak{B}_0^{* k}}_{\mathfrak{Z}_0^{* k}=\mathfrak{B}^{k\perp}}
    = 
    \overbrace{\mathfrak{B}^{k}
    \oplus 
    \mathfrak{H}^{k}}^{\mathfrak{Z}^{k}=\mathfrak{B}_0^{* k\perp}}  
    \oplus 
    \overbrace{\mathfrak{B}_0^{* k}}^{\mathfrak{Z}^{k\perp}}

2.

  .. math::

    L^2 \Lambda^k(\Omega) 
    = 
    \underbrace{\mathfrak{B}_0^{k}}_{\mathfrak{Z}^{* k\perp}} 
    \oplus 
    \underbrace{\mathfrak{H}_0^{k} 
    \oplus 
    \mathfrak{B}^{* k}}_{\mathfrak{Z}^{* k}=\mathfrak{B}_0^{k\perp}}
    = 
    \overbrace{\mathfrak{B}_0^{k}
    \oplus 
    \mathfrak{H}_0^{k}}^{\mathfrak{Z}_0^{k}=\mathfrak{B}^{* k\perp}}  
    \oplus 
    \overbrace{\mathfrak{B}^{* k}}^{\mathfrak{Z}_0^{k\perp}}

Summary
^^^^^^^

.. math::

  \begin{tabular}{|c||c|c|c|c|}
   \hline
   $\omega^k \in \Lambda^k(\Omega)$             & $k=0$ 
                                                & $k=1$ 
                                                & $k=2$ 
                                                & $k=3$ 
   \\
   \hline
   $\diff \omega^k$                             & $\Grad u$ 
                                                & $\Curl \uu$ 
                                                & $\Div \uu$ 
                                                & $-$
   \\
   $\delta \omega^k$                            & $-$ 
                                                & $-\Div \uu$ 
                                                & $\Curl \uu$ 
                                                & $-\Grad u$
   \\
   $\mathfrak{i}_{\boldsymbol{\beta}} \omega^k$ & $-$ 
                                                & $\boldsymbol{\beta} \cdot \uu$ 
                                                & $\uu \times \boldsymbol{\beta}$ 
                                                & $u \boldsymbol{\beta}$
   \\
   $\mathfrak{j}_{\boldsymbol{\beta}} \omega^k$ & $u \boldsymbol{\beta}$ 
                                                & $-\uu \times \boldsymbol{\beta}$ 
                                                & $\boldsymbol{\beta} \cdot \uu$ 
                                                & $-$
   \\
   $L_{\boldsymbol{\beta}} \omega^k$            & $\boldsymbol{\beta} \cdot \Grad u$ 
                                                & $\Grad \left(\boldsymbol{\beta} \cdot \uu \right)  + \left(\Curl \uu \right) \times \boldsymbol{\beta}$ 
                                                & $\Curl \left(\uu \times \boldsymbol{\beta} \right) + \boldsymbol{\beta} \Div \uu$ 
                                                & $\Div \left( u \boldsymbol{\beta} \right)$
   \\
   $\mathcal{L}_{\boldsymbol{\beta}} \omega^k$  & $-\Div \left( u \boldsymbol{\beta} \right)$ 
                                                & $-\Curl \left(\uu \times \boldsymbol{\beta} \right) - \boldsymbol{\beta} \Div \uu$ 
                                                & $-\Grad \left(\boldsymbol{\beta} \cdot \uu \right)  - \left(\Curl \uu \right) \times \boldsymbol{\beta}$ 
                                                & $-\boldsymbol{\beta} \cdot \Grad u$
   \\
   \hline
   $\tr \omega^k$                               & $u(\xx)$ 
                                                & $\uu(\xx) \times \nn(\xx)$ 
                                                & $\uu(\xx) \cdot  \nn(\xx)$ 
                                                & $-$
   \\
   \hline
   \hline
   $H \Lambda^k(\Omega)$                        & $\Hgrad$ 
                                                & $\Hcurl$ 
                                                & $\Hdiv$ 
                                                & $\Ltwo$
   \\
   $V_k$                                        & $\Vgrad$ 
                                                & $\Vcurl$ 
                                                & $\Vdiv$ 
                                                & $\Vltwo$
   \\
   \hline
  \end{tabular}
..   \caption{Correspondences between differential forms in $3$D, and scalar/vector fields.}



.. rubric:: References

.. bibliography:: refs_feec.bib
   :cited:
