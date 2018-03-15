DeRham sequences
****************

.. sectionauthor:: A. Ratnani

here without boundary conditions

.. math::

  \mathbb{R} \hookrightarrow \Hgrad  \xrightarrow{\quad \Grad \quad}  \Hcurl  \xrightarrow{\quad \Curl \quad}   \Hdiv  \xrightarrow{\quad \Div \quad}  \Ltwo  \xrightarrow{\quad} 0  

Pullbacks
^^^^^^^^^

In the case where the physical domain :math:`\Omega := \mathcal{F}(\hat{\Omega})` is the *image* of a *logical* domain :math:`\hat{\Omega}` by a smooth mapping :math:`\mathcal{F}` (at least :math:`\mathcal{C}^1`), we have the following *parallel* diagrams 

.. math::

  \begin{array}{ccccccc}
  \Hgrad & \xrightarrow{\quad \Grad \quad} & \Hcurl & \xrightarrow{\quad \Curl \quad} &  \Hdiv & \xrightarrow{\quad \Div \quad} & \Ltwo \\  
  \igrad \Bigg\uparrow   &     & \icurl \Bigg\uparrow  &   & \idiv \Bigg\uparrow &  & \iltwo \Bigg\uparrow       \\
  \HgradLogical & \xrightarrow{\quad \Grad \quad} & \HcurlLogical & \xrightarrow{\quad \Curl \quad} &  \HdivLogical & \xrightarrow{\quad \Div \quad} & \LtwoLogical \\  
  %
  \end{array}

Where the *mappings* :math:`\igrad, \icurl, \idiv` and :math:`\iltwo` are called **pullbacks** and are given by

.. math::

  \phi (x) :=& \igrad \hat{\phi} (\hat{x}) = \hat{\phi}(\mathcal{F}^{-1}(x)) 
  \\      
  \Psi (x) :=& \icurl \hat{\Psi} (\hat{x}) = \left( D \mathcal{F} \right)^{-T} \hat{\Psi}(\mathcal{F}^{-1}(x)) 
  \\     
  \Phi (x) :=& \idiv \hat{\Phi} (\hat{x})  = \frac{1}{J} D \mathcal{F} \hat{\Phi}(\mathcal{F}^{-1}(x)) 
  \\ 
  \rho (x) :=& \iltwo \hat{\rho} (\hat{x}) = \hat{\rho}(\mathcal{F}^{-1}(x)) 

where :math:`D \mathcal{F}` is the **jacobian matrix** of the mapping :math:`\mathcal{F}`. 

.. note:: The *pullbacks* :math:`\igrad, \icurl, \idiv` and :math:`\iltwo` are **isomorphisms** between the corresponding spaces. 

Discrete Spaces
^^^^^^^^^^^^^^^

Let us suppose that we have a sequence of finite subspaces for each of the spaces involved in the DeRham sequence. The discrete DeRham sequence stands for the following commutative diagram between continuous and discrete spaces

.. math::

  \begin{array}{ccccccc}
  \Hgrad & \xrightarrow{\quad \Grad \quad} & \Hcurl & \xrightarrow{\quad \Curl \quad} &  \Hdiv & \xrightarrow{\quad \Div \quad} & \Ltwo \\  
  \Pigrad \Bigg\downarrow   &     & \Picurl \Bigg\downarrow  &   & \Pidiv \Bigg\downarrow &  & \Piltwo \Bigg\downarrow       \\
  \Vgrad & \xrightarrow{\quad \Grad \quad} & \Vcurl & \xrightarrow{\quad \Curl \quad}  & \Vdiv & \xrightarrow{\quad \Div \quad} & \Vltwo    \\
  %
  \end{array}

When using a Finite Elements methods, we often deal with a reference element, and thus we need also to apply the *pullbacks* on the discrete spaces. In fact, we have again the following parallel diagram

.. math::

  \begin{array}{ccccccc}
  \Vgrad & \xrightarrow{\quad \Grad \quad} & \Vcurl & \xrightarrow{\quad \Curl \quad} &  \Vdiv & \xrightarrow{\quad \Div \quad} & \Vltwo \\  
  \igrad \Bigg\uparrow   &     & \icurl \Bigg\uparrow  &   & \idiv \Bigg\uparrow &  & \iltwo \Bigg\uparrow       \\
  \VgradLogical & \xrightarrow{\quad \Grad \quad} & \VcurlLogical & \xrightarrow{\quad \Curl \quad} &  \VdivLogical & \xrightarrow{\quad \Div \quad} & \VltwoLogical \\  
  %
  \end{array}

Since, the *pullbacks* are **isomorphisms** in the previous diagram, we can define a **one-to-one** correspondance 

.. math::

  \phi :=& \igrad \hat{\phi}, \quad \phi \in \Vgrad, \hat{\phi} \in \VgradLogical 
  \\
  \Psi :=& \icurl \hat{\Psi}, \quad \Psi \in \Vcurl, \hat{\Psi} \in \VcurlLogical 
  \\
  \Phi :=& \idiv \hat{\Phi}, \quad \Phi \in \Vdiv, \hat{\Phi} \in \VdivLogical 
  \\
  \rho :=& \iltwo \hat{\rho}, \quad \rho \in \Vltwo, \hat{\rho} \in \VltwoLogical 

We have then, the following results

.. math::

  \Grad \phi =& \icurl \Grad \hat{\phi} , \quad \phi \in \Vgrad
  \\
  \Curl \Psi =& \idiv \Curl \hat{\Psi} , \quad \Psi \in \Vcurl
  \\
  \Div \Phi =& \iltwo \Div \hat{\Phi} , \quad \Phi \in \Vdiv



Projectors
^^^^^^^^^^

In some cases, one may need to define projectors on *smooth* functions

.. math::

  \begin{array}{ccccccc}
  \Cinfinity & \xrightarrow{\quad \Grad \quad} & \Cinfinity & \xrightarrow{\quad \Curl \quad}  & \Cinfinity & \xrightarrow{\quad \Div \quad} & \Cinfinity    \\
  \Pigrad \Bigg\downarrow   &     & \Picurl \Bigg\downarrow  &   & \Pidiv \Bigg\downarrow &  & \Piltwo \Bigg\downarrow       \\
  \Vgrad & \xrightarrow{\quad \Grad \quad} & \Vcurl & \xrightarrow{\quad \Curl \quad}  & \Vdiv & \xrightarrow{\quad \Div \quad} & \Vltwo    \\
  \end{array}


Discrete DeRham sequence for B-Splines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Buffa et al :cite:`Buffa2009a` show the construction of a discrete DeRham sequence using B-Splines, (here without boundary conditions)

.. math::

  \begin{array}{ccccccc}
  \Hgrad & \xrightarrow{\quad \Grad \quad} & \Hcurl & \xrightarrow{\quad \Curl \quad}  & \Hdiv & \xrightarrow{\quad \Div \quad} & \Ltwo    \\
  \Pigrad \Bigg\downarrow   &     & \Picurl \Bigg\downarrow  &   & \Pidiv \Bigg\downarrow &  & \Piltwo \Bigg\downarrow       \\
  \Vgradspline &  \xrightarrow{\quad \Grad \quad}&   \Vcurlspline &  \xrightarrow{\quad \Curl \quad}  &  \Vdivspline&   \xrightarrow{\quad \Div \quad} &  \Vltwospline \\ 
  \end{array}

1d case
_______

1. DeRham sequence is reduced to

.. math::

      \mathbb{R} \hookrightarrow 
      \underbrace{\mathcal{S}^{p}}_{\VgradLogical}  \xrightarrow{\quad \Grad \quad}  
      \underbrace{\mathcal{S}^{p-1}}_{\VltwoLogical}  \xrightarrow{\quad} 0  

2. The recursion formula for derivative writes 

.. math::

      {N_i^p}'(t)=D_i^{p}(t)-D_{i+1}^{p}(t)
      \quad \mbox{where} \quad
      D_{i}^{p}(t) = \frac{p}{t_{i+p+1}-t_i}N_i^{p-1}(t) 

3. we have :math:`\mathcal{S}^{p-1} = \mathbf{span}\{ N_i^{p-1}, 1 \leq i \leq n-1 \} = \mathbf{span}\{ D_i^p, 1 \leq i \leq n-1 \}` which is a change of basis as a diagonal matrix

4. Now if :math:`u \in S^p`, with and expansion :math:`u = \sum_i u_i N_i^p`, we have

.. math::

    \begin{align*}
    u^{\prime} = \sum_i u_i \left( N_i^p \right)^{\prime} = \sum_i (-u_{i-1} + u_i) D_i^p 
  %  \label{}
    \end{align*}

5. If we introduce the B-Splines coefficients vector :math:`\mathbf{u} := \left( u_i \right)_{1 \leq i \leq n}` (and :math:`\mathbf{u}^{\star}` for the derivative), we have

.. math::

  \mathbf{u}^{\star} = D \mathbf{u}    

where :math:`D` is the incidence matrix (of entries :math:`-1` and :math:`+1`)

.. topic:: Discrete derivatives:

  .. math::

        \mathcal{G} = D 

.. .. math::
.. 
..   \begin{array}{ccccccc}
..   \Hgrad & \xrightarrow{\quad \Grad \quad} & \Hcurl & \xrightarrow{\quad \Curl \quad} &  \Hdiv & \xrightarrow{\quad \Div \quad} & \Ltwo \\  
..   \Pigrad \Bigg\downarrow   &     & \Picurl \Bigg\downarrow  &   & \Pidiv \Bigg\downarrow &  & \Piltwo \Bigg\downarrow       \\
..     \Vgrad & \xrightleftharpoons[\quad \mathcal{G}^{T} \quad]{\quad \mathcal{G} \quad} & \Vcurl & \xrightleftharpoons[\quad \mathcal{C}^{T} \quad]{\quad \mathcal{C} \quad}  & \Vdiv & \xrightleftharpoons[\quad \mathcal{D}^{T} \quad]{\quad \mathcal{D} \quad} & \Vltwo    \\
..   %
..   \end{array}

2d case
_______

In *2d*, the are two De-Rham complexes:

.. math::

  \begin{array}{ccccc}
  \Hgrad & \xrightarrow{\quad \Grad \quad} & \Hcurl & \xrightarrow{\quad \Rots \quad} & \Ltwo \\
  \Pigrad \Bigg\downarrow   &     & \Picurl \Bigg\downarrow  &   & \Piltwo \Bigg\downarrow   \\
  \Vgrad & \xrightarrow{\quad \Grad \quad} & \Vcurl & \xrightarrow{\quad \Rots \quad} & \Vltwo \\
  \end{array}

and

.. math::

  \begin{array}{ccccc}
  \Hgrad & \xrightarrow{\quad \Curl \quad} & \Hdiv & \xrightarrow{\quad \Div \quad} & \Ltwo \\
  \Pigrad \Bigg\downarrow   &     & \Pidiv \Bigg\downarrow  &   & \Piltwo \Bigg\downarrow   \\
  \Vgrad & \xrightarrow{\quad \Grad \quad} & \Vdiv & \xrightarrow{\quad \Div \quad} & \Vltwo \\
  \end{array}

.. where
.. 
.. .. math::
.. 
..   \Vgrad = \Vgradspline2d, \quad \Vcurl = \Vcurlspline2d, \quad \Vdiv = \Vdivspline2d, \quad  \mbox{and} \quad \Vltwo = \Vltwospline2d.

Let :math:`I` be the identity matrix, we have

.. topic:: Discrete derivatives:

  .. math::

        \mathcal{G} = 
        \begin{pmatrix}
          D \otimes I
          \\
          I \otimes D
        \end{pmatrix}

  .. math::

        \mathcal{C} = 
        \begin{pmatrix}
          I \otimes D
          \\
        - D \otimes I
        \end{pmatrix} 
        \quad \mbox{[scalar curl],} \quad 
        \mathcal{C} = 
        \begin{pmatrix}
        - I \otimes D
         & 
          D \otimes I
        \end{pmatrix} 
        \quad \mbox{[vectorial curl]}

  .. math::

        \mathcal{D} = 
        \begin{pmatrix}
          D \otimes I
         & 
          I \otimes D
        \end{pmatrix}

3d case
_______

.. topic:: Discrete derivatives:

  .. math::

        \mathcal{G} = 
        \begin{pmatrix}
          D \otimes I \otimes I
          \\
          I \otimes D \otimes I 
          \\
          I \otimes I \otimes D
        \end{pmatrix}

  .. math::

        \mathcal{C} = 
        \begin{pmatrix}
          0    &    - I \otimes I \otimes D     &     I \otimes D \otimes I 
          \\
          I \otimes I \otimes D   &    0   &   - D \otimes I \otimes I 
          \\
          - I \otimes D \otimes I  & D \otimes I \otimes I & 0 
        \end{pmatrix} 

  .. math::

        \mathcal{D} = 
        \begin{pmatrix}
          D \otimes I \otimes I
          & 
          I \otimes D \otimes I 
          &
          I \otimes I \otimes D
        \end{pmatrix}

.. note:: From now on, we will denote the discrete derivative by :math:`\mathbb{D}_k` for the one going from :math:`V_k` to :math:`V_{k+1}`.

Algebraic identities
^^^^^^^^^^^^^^^^^^^^

Let us consider the discretization of the exterior derivative

.. math::

  \omega^{k+1} = \diff  \omega^k

multiplying by a test function :math:`\eta^{k+1}` and integrating over the whole computation domain, we get

.. math::

  \left( \eta^{k+1}, \omega^{k+1} \right)_{k+1} = \left( \eta^{k+1}, \diff \omega^{k} \right)_{k+1}

let :math:`E^{k+1}`, :math:`W^{k}` and :math:`W^{k+1}` be the vector representation of :math:`\eta^{k+1}`, :math:`\omega^{k}` and  :math:`\omega^{k+1}`. We get

.. math::

  {E^{k+1}}^T M_{k+1} W^{k+1} = {E^{k+1}}^T D_{k+1,k} W^{k}  

where 

.. math::

  D_{k+1,k} = \left( \left( \eta^{k+1}_i, \diff \omega^{k}_j \right)_{k+1} \right)_{i,j}

On the other hand, using the coderivative, we get 

.. math::

  \left( \eta^{k+1}, \omega^{k+1} \right)_{k+1} = \left( \delta \eta^{k+1}, \omega^{k} \right)_{k} + BC

Let us now introduce the following matrix

.. math::

  D_{k,k+1} = \left( \left( \delta \eta^{k+1}_i, \omega^{k}_j \right)_{k} \right)_{i,j}

hence,

.. math::

  {E^{k+1}}^T D_{k,k+1} W^{k} = \left( \mathbb{D}^{\star}_{k+1} E^{k+1} \right)^T M_{k} W^{k} 


Therefor, we have the following important result

.. topic:: Proposition:

  * :math:`D_{k+1,k} = D_{k,k+1} + BC`

  * :math:`D_{k+1,k} = M_{k+1} \mathbb{D}^T_k`

  * :math:`D_{k,k+1} = {\mathbb{D}^{\star}_{k+1}}^T M_{k}`



.. rubric:: References

.. bibliography:: refs_derham.bib
   :cited:
