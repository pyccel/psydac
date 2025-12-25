New Matrix Assembly for PSYDAC
------------------------------

Here we keep track on the performance of the new assembly algorithm (sum factorization).
We measure both the discretization time of `BilinearForms` as well as the matrix assembly time of a `DiscreteBilinearForm`
and compare it to the old algorithm. Executing `compare_3d_matrix_assembly_speed.py` will add new data to this file.
This allows us to detect whether any future changes have a positive or negative impact.

(Of course, runtime depends on the machine used to execute this file. Hence, an even decrease in runtime
is no reason to celebrate, and an even increase in runtime no reason to worry.)

Test cases
----------

1. Scaling Analysis \ (**REMOVED**)
   We assemble the H1 and Hcurl mass matrices ($\Omega$ is a half hollow torus (analytical mapping), ncells = [32, 32, 32], periodic=[False, False, False]) for varying degrees
   and report the assembly times in Figures.

2. Three specific test cases \
   We report assembly and discretization time for the following test cases \
   2.1 "Q" 
     - $(u, v; F)\mapsto\int_{\Omega}(F\times u)\circ(F\times v)$
     - $u, v, F\in H(curl;\Omega)$, $F$ a "free field"
     - no mapping ($\Omega=(0,1)^3$)
     - ncells = [32, 32, 32], degree = [3, 3, 3], periodic = [False, False, False]
   
   2.2 "weighted $Hdiv$ mass matrix"
     - $(u, v; \gamma)\mapsto\int_{\Omega}u\circ v\ \gamma$
     - $u, v\in H(div;\Omega)$, $\gamma$ an analytic scalar weight function
     - analytical mapping ($\Omega$ a half hollow torus)
     - ncells = [32, 32, 32], degree = [3, 3, 3], periodic = [False, False, False]
      
   2.3 "curl curl"
     - $(u, v)\mapsto\int_{\Omega}(\nabla\times u)\circ(\nabla\times v)$
     - $u, v\in H(curl;\Omega)$
     - Bspline mapping ($\Omega$ a half hollow torus)
     - ncells = [32, 32, 32], degree = [3, 3, 3], periodic = [False, False, False]

3. More variations!
   - $(u, v)\mapsto\int_{\Omega}\nabla u\circ\nabla v$
   - ncells = [16, 8, 32], degree = [2, 4, 3], periodic = [False, True, False]
   - 3.1.1
     - `u, v = elements_of(derham.V0, names='u, v')`
     - no mapping
   - 3.1.2
     - `u, v = elements_of(derham.V0, names='u, v')`
     - analytical mapping
   - 3.1.3
     - `u, v = elements_of(derham.V0, names='u, v')`
     - Bspline mapping
   - 3.2
     - `u, v = elements_of(ScalarFunctionSpace('Vs', domain), names='u, v')`
     - analytical mapping
   - 3.3
     - `u = element_of(derham.V0, name='u')`
     - `v = element_of(ScalarFunctionSpace('Vs', domain), name='v')`
     - analytical mapping
   - `u, v = elements_of(derham.V0, names='u, v')`
   - analytical mapping
   - 3.4
     - additional analytical weight function $\gamma$
   - 3.5
     - additional scalar free field $F\in H^1(\Omega)$
   - 3.6
     - multiplicity vector [1, 3, 2]

Data
----

2025-09-09 17:28:19 (added by Julian O. - ThinkPad T14 on performance mode)
----------

| Test case | old assembly | new assembly | old discretization | new discretization |
| --- | --- | --- | --- | --- |
| 2.1 | 21.358 | 1.676 | 16.568 | 16.589 |
| 2.2 | 2.348 | 0.322 | 4.533 | 3.38 |
| 2.3 | 36.858 | 3.171 | 16.749 | 19.413 |
| 3.1.1 | 0.236 | 0.044 | 1.462 | 1.482 |
| 3.1.2 | 0.54 | 0.051 | 1.722 | 1.52 |
| 3.1.3 | 0.802 | 0.169 | 3.579 | 2.414 |
| 3.2 | 0.517 | 0.045 | 1.027 | 1.425 |
| 3.3 | 0.525 | 0.06 | 1.023 | 1.527 |
| 3.4 | 0.524 | 0.044 | 1.085 | 1.547 |
| 3.5 | 0.577 | 0.057 | 1.51 | 1.784 |
| 3.6 | 0.598 | 0.209 | 2.411 | 1.848 |

