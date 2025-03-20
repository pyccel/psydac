New Matrix Assembly for Psydac
------------------------------

Here we keep track on the performance of the new assembly algorithm.
We measure both the discretization time of `BilinearForms` as well as the matrix assembly time of a `DiscreteBilinearForm`
and compare it to the old algorithm. Executing `compare_3d_matrix_assembly_speed.py` will add new data to this file.
This allows us to detect whether any future changes have a positive or negative impact.

Test cases
----------

1. Scaling Analysis \
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


2025-03-19 16:19:13
----------

![](tests/figures/H1_2025-03-19_16:19:13.png)
![](tests/figures/Hcurl_2025-03-19_16:19:13.png)

| Test case | old assembly | new assembly | old discretization | new discretization |
| --- | --- | --- | --- | --- |
| 2.1 | 28.224941968917847 | 2.1576409339904785 | 23.585421323776245 | 30.444850206375122 |
| 2.2 | 4.931920051574707 | 0.725954532623291 | 12.519602298736572 | 12.77993392944336 |
| 2.3 | 44.408453702926636 | 4.248292684555054 | 24.583781003952026 | 28.322672605514526 |
| 3.1.1 | 0.23116183280944824 | 0.06337881088256836 | 1.7055208683013916 | 2.4070897102355957 |
| 3.1.2 | 1.0104484558105469 | 0.09787487983703613 | 2.425628185272217 | 3.6106112003326416 |
| 3.1.3 | 0.9545958042144775 | 0.3607370853424072 | 5.0261549949646 | 5.2433154582977295 |
| 3.2 | 1.0606496334075928 | 0.11185073852539062 | 1.6706140041351318 | 3.7393929958343506 |
| 3.3 | 1.0473194122314453 | 0.09792256355285645 | 1.6306214332580566 | 3.327897071838379 |
| 3.4 | 1.0399527549743652 | 0.11473822593688965 | 1.5639660358428955 | 3.353390693664551 |
| 3.5 | 0.9348330497741699 | 0.12152266502380371 | 2.2530579566955566 | 4.756465673446655 |
| 3.6 | 1.013357162475586 | 0.33884382247924805 | 3.090827703475952 | 5.644901752471924 |

2025-03-20 09:57:14
----------

![](tests/figures/H1_2025-03-20_09:57:14.png)
![](tests/figures/Hcurl_2025-03-20_09:57:14.png)

| Test case | old assembly | new assembly | old discretization | new discretization |
| --- | --- | --- | --- | --- |
| 2.1 | 27.988634824752808 | 1.8615539073944092 | 23.975714206695557 | 19.485677242279053 |
| 2.2 | 5.05242133140564 | 0.6173980236053467 | 11.472531080245972 | 7.862813949584961 |
| 2.3 | 44.12760281562805 | 4.010182619094849 | 23.294647455215454 | 18.524545907974243 |
| 3.1.1 | 0.23708844184875488 | 0.07375884056091309 | 1.7830462455749512 | 1.7122766971588135 |
| 3.1.2 | 0.9980983734130859 | 0.0917971134185791 | 2.299697160720825 | 2.204707384109497 |
| 3.1.3 | 1.1703286170959473 | 0.18325209617614746 | 5.208899736404419 | 2.567802906036377 |
| 3.2 | 0.9243614673614502 | 0.0949547290802002 | 1.9204461574554443 | 2.482546806335449 |
| 3.3 | 0.8310558795928955 | 0.1002969741821289 | 1.482534408569336 | 1.9721088409423828 |
| 3.4 | 0.9283547401428223 | 0.097259521484375 | 1.4863159656524658 | 2.015451192855835 |
| 3.5 | 0.9757201671600342 | 0.14624404907226562 | 2.1416451930999756 | 2.58785343170166 |
| 3.6 | 1.259347677230835 | 0.28324031829833984 | 3.146653175354004 | 2.520529270172119 |

