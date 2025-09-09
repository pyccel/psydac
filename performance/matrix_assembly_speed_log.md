New Matrix Assembly for Psydac
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

2025-09-09 16:31:59 (added by Julian O. - ThinkPad T14 on performance mode)
----------

| Test case | old assembly | new assembly | old discretization | new discretization |
| --- | --- | --- | --- | --- |
| 2.1 | 26.580852270126343 | 2.0685055255889893 | 22.13553762435913 | 21.47482991218567 |
| 2.2 | 2.819295883178711 | 0.5051388740539551 | 7.022295236587524 | 4.818620920181274 |
| 2.3 | 45.10732555389404 | 4.260260820388794 | 21.42706608772278 | 24.236170291900635 |
| 3.1.1 | 0.2728254795074463 | 0.0818166732788086 | 1.8811206817626953 | 1.998206377029419 |
| 3.1.2 | 0.7241649627685547 | 0.06369304656982422 | 1.9924912452697754 | 1.871694803237915 |
| 3.1.3 | 0.9324562549591064 | 0.24193167686462402 | 4.724050998687744 | 3.2216196060180664 |
| 3.2 | 0.7062575817108154 | 0.08480167388916016 | 1.30161452293396 | 1.797856092453003 |
| 3.3 | 0.6325840950012207 | 0.04571199417114258 | 1.3283660411834717 | 1.7871713638305664 |
| 3.4 | 1.0434694290161133 | 0.06348180770874023 | 1.2950034141540527 | 1.920619010925293 |
| 3.5 | 0.6724743843078613 | 0.0754859447479248 | 1.6902377605438232 | 2.49491024017334 |
| 3.6 | 0.7051618099212646 | 0.2683985233306885 | 3.0718603134155273 | 2.386533737182617 |

