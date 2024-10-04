## Plot convergence rates

using Plots 
order_of(i, err, n) = -log(err[i]/err[i-1])/ log(n[i]/n[i-1])
err = [0.0006093087419533646
8.215587055263705e-05
2.2450946702990757e-05
6.042546774929098e-06]
nps = [[1, 2], [2,4], [3,6], [4,8]]
nbc = [12, 12]
n = prod.(nps) .* prod(nbc)

err = [0.005364339153808812, 0.0015577609777295209, 0.0015776936452793315, 0.0003634617608575696, ]
order = [1.6, 0, 3.6]
nps = [[1, 2], [1,4], [2,4], [3,4]]
nbc = [8, 8]
n = prod.(nps) .* prod(nbc)

err_new = [0.0004456973330175234, 0.00043832888750203517, 0.00018342095777117612]
nps = [[1, 4], [2, 4], [3, 6]]
nbc = [10, 10]
n_new = prod.(nps) .* prod(nbc)

#nbp_arr = [[1,2], [2, 4], [4, 8]]
#kappa = 2
#alpha = 1.25
#epsilon = 0.3
#nb_tau = 1

E_err =  [0.0015014030026049103, 0.0002081294266858147, 2.6905608322145658e-05, ]
B_err =  [0.0004520784305227154, 0.00022817525462152303, 8.865397988279003e-05, ]
D_err =  [0.0012642521309811563, 7.739309358325423e-05, 3.5961833451325855e-05, ]
H_err =  [0.0064305895320782725, 0.0003327376717330393, 0.00010783860709347753, ]

nbc = [10, 10]
deg = [3,3]
nbp = [[1,2], [2, 4], [4, 8]]
n = prod.(nbp) .* prod(nbc)

p = plot(xscale = :log, yscale = :log, legend=true);
plot!(p, n, E_err, label = "E error");
plot!(p, n, D_err, label = "D error");
plot!(p, n, H_err, label = "H error");
plot!(p, n, B_err, label = "B error")

plot!(p ,n, 1 ./ n.^1, label = "order 1", line=:dash)
plot!(p ,n, 100 ./ n.^2, label = "order 2", line=:dash)
plot!(p ,n, 10000 ./ n.^3, label = "order 3", line=:dash)


### test_13_2 nb_tau = 1
E_err = Dict( 
3 => [0.000605132370541113, 0.00013748085731248445, 4.8081108089419376e-05, 1.4523891683067183e-05, ], 
4 => [0.0007427288288723868, 5.497651268123326e-05, 7.5194302362111835e-06, 1.9934731835496476e-06, ], 
5 => [0.0003262877780453646, 3.2899368866255383e-06, 1.424751976358513e-06, 1.0807056581061754e-06, ], 
)

B_err = Dict( 
3 => [0.00018943774707028314, 0.0002163084693398593, 0.0001435976882676298, 5.7887900001611435e-05, ], 
4 => [0.0005017667471051993, 0.0001228246174243614, 3.6652327578967614e-05, 2.6669867459209636e-05, ], 
5 => [0.00035196282983266676, 2.778373085297126e-05, 2.6488675615473815e-05, 2.6313435192486333e-05, ], 
)

D_err = Dict( 
    3 => [0.0006429266528120057, 9.362093073013866e-05, 5.2597448173416474e-05, 1.9691802115460005e-05, ], 
    4 => [0.00030589574854813286, 6.237037089125509e-05, 9.29530076969453e-06, 2.591802009868985e-06, ], 
    5 => [0.0003320077269838461, 5.246281808977938e-06, 2.1568331106309695e-06, 1.9508196041074453e-06, ], 
) 

H_err = Dict(
3 => [0.0028074469442091747, 0.0003264772490686335, 0.0001754843865278241, 6.698266943933871e-05, ], 
4 => [0.0010909236251250083, 0.000172891919551402, 3.903444822152876e-05, 2.645143842874938e-05, ], 
5 => [0.0009801932552943047, 2.7471356178572918e-05, 2.6519622839571894e-05, 2.631522741496249e-05, ], 
)

nbc = [12, 12]
deg = [3, 4, 5]
nbp = [[1,2], [2, 4], [3,6], [4, 8]]
n = prod.(nbp) .* prod(nbc)

dof_1d = Dict(d => [(nbc[1]) * k[1] for k in nbp] for d in deg)

for d in deg
    p = plot(xscale = :log, yscale = :log, legend=true, title="Convergence for degree =" * string(d));
    plot!(p, dof_1d[d], E_err[d], label = "E error");
    plot!(p, dof_1d[d], D_err[d], label = "D error");
    plot!(p, dof_1d[d], H_err[d], label = "H error");
    plot!(p, dof_1d[d], B_err[d], label = "B error")

    plot!(p ,dof_1d[d],  dof_1d[d][1]^(d-2) ./ n.^(d-2), label = "order "*string(d-2), line=:dash)
    plot!(p ,dof_1d[d], dof_1d[d][1]^(d-1) ./ n.^(d-1), label = "order "*string(d-1), line=:dash)
    #plot!(p ,dof_1d[d], 10000 ./ n.^d, label = "order 3", line=:dash)
    savefig(p, "/home/schnack/PhD/pyccel/psydac/02_cond_test/conv_deg="*string(d))
end

#### Hodge Convergence Plots of V^1 in L^2 norm
p = [2, 3, 4] # ( E in (p, p+1) x (p+1, p) )

r = [0.25, 3]
theta = [0, 2*pi]
nbp = [[1,2], [2,4], [4,8]]
nbc = [12, 12]

n_r = Dict(pp => [patch[1] * (nbc[1] + pp) for patch in nbp] for pp in p)
n_theta = Dict(pp => [patch[2] * (nbc[2] + pp) for patch in nbp] for pp in p)

h_r = Dict(pp => (r[end]-r[1]) ./ n_r[pp] for pp in p)
h_theta = Dict(pp => (theta[end]-theta[1]) ./ n_theta[pp] for pp in p)

# dual error
err_dual = Dict( 2 => [0.02506745377598473
0.00421092396837029
0.00030379342426539713],
3 => [0.02955581106303294
0.0008234062442829691
3.101010131021563e-05],
4 => [0.00445028586678166
0.00011129200965657454
2.7426658819828884e-06]
 ) 

plt = plot(dpi=900, title =L"L^2" * " approximation of the Hodge star operator", xscale = :log, yscale = :log, legend=:bottomright, grid=true, xticks = ([10^(-1), 5 * 10^(-2), 10^(-2)],[L"10^{-1}",L"5 \cdot 10^{-2}", L"10^{-2}"]), xlabel = "h", ylabel= L"L^2" *" error");

for pp in p
    plot!(plt, h_r[pp], err_dual[pp], label = "degree $pp", marker=:x)
    plot!(plt, h_r[pp], (1.4 * h_r[pp]).^pp, label = "order $pp", line =:dash)
end
    plot!(plt, h_r[p[end]], (1.6 * h_r[p[end]]).^(p[end]+1), label = "order $(p[end]+1)", line =:dash)

    savefig(plt, "/home/schnack/PhD/pyccel/psydac/02_cond_test_new/hodge_dual")

err_primal = Dict( 2 => [
    0.032714323184759905
    0.004754108871426222
    0.0004388577363741005
],
    3 => [
        0.02518347127207974
        0.000980649962143058
        3.889869086129154e-05
    ],
    4 => [
        0.005866725353064341
        0.00010938122772329016
        5.644777315003557e-06
    ]
     ) 

plt = plot(dpi = 900, title = "Direct approximation of the Hodge star operator", xscale = :log, yscale = :log, legend=:bottomright, grid=true, xticks = ([10^(-1), 5 * 10^(-2), 10^(-2)],[L"10^{-1}",L"5 \cdot 10^{-2}", L"10^{-2}"]), xlabel = "h", ylabel= L"L^2" *" error");

for pp in p
    plot!(plt, h_r[pp], err_primal[pp], label = "degree $pp", marker=:x)
    plot!(plt, h_r[pp], (1.4* h_r[pp]).^pp, label = "order $pp", line =:dash)
end
    plot!(plt, h_r[p[end]], (1.6 * h_r[p[end]]).^(p[end]+1), label = "order $(p[end]+1)", line =:dash)
    savefig(plt, "/home/schnack/PhD/pyccel/psydac/02_cond_test_new/hodge_primal")

### New Convergence test 18.03 r_max = 3

### test_13_2 nb_tau = 1
E_err = Dict( 
2 => [0.005730024322675886, 0.0014963659487157467, 0.0002818115082384959, ], 
3 => [0.0025715551575089053, 0.00029911696996331934, 1.2031153591667473e-05, ],
4 => [0.00040352315508708145, 5.493239874924261e-05, 3.816671177687738e-06, ]
)

B_err = Dict( 
2 => [0.0014597248099960437, 0.0008151507359852176, 0.0004557588041715098, ], 
3 => [0.0012742328639791583, 0.000383743531158304, 0.0003435816962572876, ], 
4  => [0.00039512462411525417, 0.00034866011458763315, 0.00034335092538050743, ]
)

D_err = Dict( 
    2 => [0.00257700981367009, 0.0008654619233305445, 0.0001929288440574433, ], 
    3 => [0.001936178440848999, 0.00012024552128456159, 8.90250249666986e-06, ], 
    4 => [7.063774887400405e-05, 3.315938106909475e-05, 4.827829781167403e-06, ]   
    ) 

H_err = Dict(
2 => [0.030550487110711757, 0.005312688724519439, 0.0007862077022104203, ], 
3 => [0.016776405958435388, 0.0006407448244235532, 0.0003462922540422168, ],
4 => [0.0005145411600820636, 0.00035764716243561303, 0.000343364511177309, ]
)

p = [2,3,4] # ( E in (p, p+1) x (p+1, p) )

r = [0.25, 3]
theta = [0, 2*pi]
nbp = [[1,2], [2,4], [4,8]]
nbc = [12, 12]

n_r = Dict(pp => [patch[1] * (nbc[1] + pp) for patch in nbp] for pp in p)
n_theta = Dict(pp => [patch[2] * (nbc[2] + pp) for patch in nbp] for pp in p)

h_r = Dict(pp => (r[end]-r[1]) ./ n_r[pp] for pp in p)
h_theta = Dict(pp => (theta[end]-theta[1]) ./ n_theta[pp] for pp in p)


for pp in p
    plt = plot(dpi = 900, title = "Convergence for degree = $pp", xscale = :log, yscale = :log, legend=:bottomright, grid=true, xticks = ([10^(-1), 5 * 10^(-2), 10^(-2)],[L"10^{-1}",L"5 \cdot 10^{-2}", L"10^{-2}"]), xlabel = "h", ylabel= L"L^2" *" error");

    plot!(plt, h_r[pp], E_err[pp], label = "E error");
    plot!(plt, h_r[pp], D_err[pp], label = "D error");
    plot!(plt, h_r[pp], H_err[pp], label = "H error");
    plot!(plt, h_r[pp], B_err[pp], label = "B error")

    plot!(plt, h_r[pp],  0.01 * h_r[pp].^(pp-1), label = "order $(pp -1)", line=:dash)
    plot!(plt, h_r[pp],  0.1 * h_r[pp].^(pp), label = "order $(pp)", line=:dash)

   savefig(plt, "/home/schnack/PhD/pyccel/psydac/02_cond_test_new/conv_deg=$pp")

end

#eccomas talk
#== 
errors_poisson_refine handle 
deg 2: 
r 4: 6.716370513834187
r 8: 0.31596651589799896
r 16: 0.015169429285767706

deg 3: 
r 4: 0.8652746143836866
r 8: 0.0460638357318078
r 16: 0.004894048295260852

deg 4: 
r 4: 0.3446834904681976
r 8: 0.08675525054208968
r 16: 0.0057562010063018565

deg 5: 
r 4: 0.18317312138269953
r 8: 0.05573856817542109
r 16: 0.003616293279613309
==#
deg = [2, 3, 4, 5]
r = [4, 8, 16]

err_poisson = Dict(
    2 => [6.716370513834187, 0.31596651589799896, 0.015169429285767706],
    3 => [0.8652746143836866, 0.0460638357318078, 0.004894048295260852],
    4 => [0.3446834904681976, 0.08675525054208968, 0.0057562010063018565],
    5 => [0.18317312138269953, 0.05573856817542109, 0.003616293279613309]
)

order_of(i, err, n) = -log(err[i]/err[i-1])/ log(n[i]/n[i-1])
for (i, errs) in err_poisson
    println("Degree $i")
    for j in 1:length(errs)-1
         println("Order of convergence between r $(r[j]) and r $(r[j+1]) is ", order_of(j+1, errs, r))
    end
end

e = [0.41536, 0.1649, 0.0069]
r = [4, 8, 16]


#==
Time-Harmonic Maxwell non-homogeneous example
    for eccomas talk
==# 
errors_matching = Dict(
    2 => [0.10371345150894275, 0.0072015081284075855, 0.00040849482328333085, 2.6169949551117192e-05],
    3 => [0.01446508035387591, 0.0023264487870962303, 0.00015460555447238713, 1.2077993597503185e-05],
    4 => [0.002864380149151581, 0.0008324706214669557, 1.8540564953024946e-05, 6.116418097565374e-07],
    5 => [0.0013354881139753393, 0.00017353391601301763, 3.891131464720022e-06, 5.634729292836455e-08],
)

N_matching = Dict(
    2 => [432, 1080, 3240, 11016],
    3 => [720, 1512, 3960, 12312],
    4 => [1080, 2016, 4752, 13680],
    5 => [1512, 2592, 5616, 15120],
)

errors_nonmatching = Dict(
    2 => [0.07586741143659759, 0.0049968410988792955, 0.00029225807248635816, 1.8761797756542577e-05],
    3 => [0.011075034432444068, 0.001992064883603627, 0.00013055710285983274, 9.616220561931809e-06],
    4 => [0.0019974896233534694, 0.0005701931050167337, 1.3805467670876804e-05, 4.38188424782895e-07],
    5 => [0.000938573835654647, 0.00012656407527069155, 2.9562921154301244e-06, 4.393519217438543e-08],
)

N_nonmatching = Dict(
    2 => [756, 2160, 7128, 25704],
    3 => [1116, 2736, 8136, 27576],
    4 => [1548, 3384, 9216, 29520],
    5 => [2052, 4104, 10368, 31536],
)

using Plots, LaTeXStrings
#plt = plot(dpi = 900, title = "Relative"*L"L^2"*"-error", xscale = :log, yscale = :log, legend=:bottomright, grid=true, xticks = ([10^(-1), 5 * 10^(-2), 10^(-2)],[L"10^{-1}",L"5 \cdot 10^{-2}", L"10^{-2}"]), xlabel = "h", ylabel= L"L^2" *" error");
plt = plot(dpi = 900, legend_columns=2, scale = :log, yscale = :log, legend=:bottomleft, grid=true, xlabel = "dof", ylabel= L"L^2" *" error");
colors = palette(:tab10)
ref = [2, 4, 8, 16]
for p in [2, 3, 4, 5]

    #plot!(plt, N_matching[p], errors_matching[p], label = "p = $p", color = colors[p-1], markershape = :circle, markerstrokewidth=0)
    #plot!(plt, N_matching[p],  errors_matching[p][2]*(N_matching[p][2]^p)  * (1 ./ N_matching[p]).^(p+1), label = "order $(p+1)", line=:dash, color = colors[p-1])
    plot!(plt, ref, errors_matching[p], label = "p = $p", color = colors[p-1], markershape = :circle, markerstrokewidth=0)
    plot!(plt, ref, errors_matching[p][2]*(ref[2]^(p+1))  * (1 ./ ref).^(p+1), label = "order $(p+1)", line=:dash, color = colors[p-1])
end
savefig(plt, "conv_timeharmonic_matching")

plt = plot(dpi = 900, legend_columns=2, scale = :log, yscale = :log, legend=:bottomleft, grid=true, xlabel = "dof", ylabel= L"L^2" *" error");
colors = palette(:tab10)
for p in [2, 3, 4, 5]

    plot!(plt, N_matching[p], errors_nonmatching[p], label = "p = $p", color = colors[p-1], markershape = :circle, markerstrokewidth=0)
    plot!(plt, N_matching[p], errors_nonmatching[p][2]*(N_matching[p][2]^p)  * (1 ./ N_matching[p]).^p, label = "order $(p)", line=:dash, color = colors[p-1])

end
savefig(plt, "conv_timeharmonic_nonmatching")


plt = plot(dpi = 1200, legend_columns=3, scale = :log, yscale = :log, legend=:bottomleft, grid=true, xlabel = "N p.p.", ylabel= "Relative " * L"L^2" *" error", xticks = ([2, 4, 8, 16],[L"2",L"4", L"8", L"16"]));
colors = palette(:tab10)
ref = [2, 4, 8, 16]
for p in [2, 3, 4, 5]

    #plot!(plt, N_matching[p], errors_matching[p], label = "p = $p", color = colors[p-1], markershape = :circle, markerstrokewidth=0)
    #plot!(plt, N_matching[p],  errors_matching[p][2]*(N_matching[p][2]^p)  * (1 ./ N_matching[p]).^(p+1), label = "order $(p+1)", line=:dash, color = colors[p-1])
    plot!(plt, ref, errors_matching[p], label = "p = $p", color = colors[p-1], markershape = :circle, markerstrokewidth=0)    
    plot!(plt, ref, errors_nonmatching[p], label = "p = $p", color = colors[p-1], markershape = :xcross, markerstrokewidth=0)
    plot!(plt, ref, errors_nonmatching[p][2]*(ref[2]^(p+1))  * (1 ./ ref).^(p+1), label = "order $(p+1)", line=:dash, color = colors[p-1])
    #plot!(plt, ref, errors_nonmatching[p][2]*(ref[2]^(p+1))  * (1 ./ ref).^(p+1), label = "order $(p+1)", line=:dash, color = colors[p-1])

end
savefig(plt, "conv_timeharmonic_both")

#==
    curl-curl Eigenvalue errors for individual ev
==#

# non-matching
ev = [1,2,3,4,5]
dof_nonmatching = [12, 60, 180, 444]
errors_nonmatching = Dict(
    1 => [0.003891224564632534, 8.367391204666674e-05, 0.00010352841041871841, 0.0002707772560572863, 0.007077038912672862],
    2 => [0.000620354219570185, 6.1026260125274234e-06, 6.788895525744465e-07, 5.2585508798941305e-05, 0.001161621840594762],
    3 => [0.00014365918326353366, 1.1254881546740592e-06, 8.09850675409507e-09, 1.2112634941274791e-05, 0.0002701609076432021],
    4 => [3.88741035108886e-05, 2.831232821520757e-07, 4.3511416691899285e-09, 3.264496134747219e-06, 7.323986092444557e-05],
)

dof_matching = [12, 60, 189, 432]
errors_matching = Dict(
    1 => [0.003891224564632534, 8.367391204666674e-05, 0.00010352841041871841, 0.0002707772560572863, 0.007077038912672862],
    2 => [0.0011575861706059065, 1.377665310275944e-05, 5.441107451380049e-06, 0.00010103779993109185, 0.0021643227377712293], 
    3 => [0.0004319560474121964, 3.9260594286716355e-06, 1.0133630112818537e-07, 3.7017093351465746e-05, 0.0008105948271506236],
    4 => [0.0002107510144635505, 1.7169675681216745e-06, 5.103829892050271e-08, 1.7912614749704403e-05, 0.0003961665654372837 ]
)

plt = plot(dpi = 1200, legend_columns=1, yscale = :log, legend=:bottomright, grid=true, xlabel = "eigenvalue number", ylabel="Error");
colors = palette(:tab10)
for n in 1:4

    plot!(plt, ev, errors_nonmatching[n], label = "nc = $(dof_nonmatching[n])", color = colors[n], markershape = :circle, markerstrokewidth=0)    

end
savefig(plt, "cc_ev_nm.png")


plt = plot(dpi = 1200, legend_columns=1, yscale = :log, legend=:bottomright, grid=true, xlabel = "eigenvalue number", ylabel="Error");
colors = palette(:tab10)
for n in 1:4
    plot!(plt, ev, errors_matching[n], label = "nc = $(dof_matching[n])", color = colors[n], markershape = :circle, markerstrokewidth=0)    
end
savefig(plt, "cc_ev_m.png")

plt = plot(dpi = 2400, legend_columns=2, yscale = :log, legend=:bottomright, grid=true, xlabel = "eigenvalue number", ylabel="Relative "*L"L^2"*" error");
colors = palette(:tab10)
for n in 1:4
    plot!(plt, ev, errors_matching[n], label = "nc = $(dof_matching[n])", color = colors[n], markershape = :circle, markerstrokewidth=0)    
    plot!(plt, ev, errors_nonmatching[n], label = "nc = $(dof_nonmatching[n])", color = colors[n], markershape = :xcross, markerstrokewidth=0)    
end
savefig(plt, "cc_ev_both.png")


#### Update 19.09 formp notes

#== 
    Weak divergence approx 
    #capture regex: np\.float64\((\d.\d+e*-*\d*)\)

    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.10210435314326058, 0.01908335135505668, 0.003397637857506263, 0.0006108629798786026], 
    2=> [0.022317881538604513, 0.0020692023781013293, 0.0003012147578547065, 6.531242978686643e-05],
    3=> [0.00581208410840217, 0.00031913440982102407, 1.7930206747935584e-05, 1.4794803301533312e-06],
    4=> [0.0014901587820399847, 4.372974412079378e-05, 3.925927583211601e-06, 4.821736414461238e-07]
    [24, 48, 96, 192]

    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**k]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.11235801760582836, 0.019192071389985212, 0.0032896982346112664, 0.0005711099012227115], 
    2=> [0.02576136791884582, 0.0024808173561727295, 0.00022890366817756745, 2.066214559585022e-05], 
    3=> [0.006636801340509517, 0.0003208307680857298, 1.4310404872785065e-05, 6.272836882172252e-07], 
    4=> [0.001510266533571046, 3.49428308939966e-05, 7.731761529006344e-07, 1.724735595918476e-08]
    [18, 36, 72, 144]

    # works
    # ncells = np.array([[fac * 2**k, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = True, ncells=ncells, deg=deg, mom_pres=True, hom_bc=False, hide_plot=True, verbose=False)
    1=> [0.1670523173023837, 0.016434932822785617, 0.0014768772464594869, 0.0001308163198441592], 
    2=> [0.023073758205141275, 0.004232194164331459, 0.0004311210561410838, 3.9445343620267256e-05], 
    3=> [0.013164887135416513, 0.0003953019430579754, 9.393719437140717e-06, 2.1538587741970027e-07], 
    4=> [0.0003181274815591522, 5.83739590227322e-05, 1.68630480596902e-06, 4.158160343498003e-08]
    [30, 60, 120, 240]

    #fails
    # ncells = np.array([[fac * 2**k, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.09348069093882876, 0.017169958933942332, 0.0031068817774863216, 0.0005669660104213093], 
    2=> [0.025021925119428863, 0.0024156274501849566, 0.0003125380417688864, 6.360780288878652e-05],
    3=> [0.0057741103263283635, 0.0002957307406155766, 1.6981986159583456e-05, 1.429156112313664e-06], 
    4=> [0.001566153282548965, 4.68596021739275e-05, 3.828850269363589e-06, 4.6654204927148265e-07]
    [30, 60, 120, 240]

    # fails
    # ncells = np.array([[fac * 2**k, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    # IDENTITY MAPPING
    1=> [0.0592588645127828, 0.0077439032347807225, 0.001520326355164873, 0.0003642770421844237], 
    2=> [0.0150317947080043, 0.002333539715085711, 0.0004639983187099832, 0.00011101989708406474], 
    3=> [0.002415112716403155, 8.211226232921506e-05, 7.004867464069024e-06, 8.464846774588766e-07], 
    4=> [0.0003239161619176821, 2.249449393254832e-05, 2.462684576233287e-06, 3.033026090377864e-07]
    [30, 60, 120, 240]

    # works:
    # ncells = np.array([[fac * 2**k, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])      
    # h0div = True, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.2017063980702626, 0.03627496378299692, 0.006419631330321424, 0.00113230305706528], 
    2=> [0.05572304436850622, 0.005073532467272198, 0.00045051943205503853, 3.988036709253593e-05], 
    3=> [0.013725192475720262, 0.0006647675705502903, 3.013545246867294e-05, 1.3398245844349905e-06], 
    4=> [0.003454143441071323, 8.155599209686254e-05, 1.8264449710646182e-06, 4.243442666269463e-08]
    [30, 60, 120, 240]

    # works
    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.11235801760582836, 0.019192071389985212, 0.0032896982346112664, 0.0005711099012227115], 
    2=> [0.02576136791884582, 0.0024808173561727295, 0.00022890366817756745, 2.066214559585022e-05], 
    3=> [0.006636801340509517, 0.0003208307680857298, 1.4310404872785065e-05, 6.272836882172252e-07], 
    4=> [0.001510266533571046, 3.49428308939966e-05, 7.731761529006344e-07, 1.724735595918476e-08]
    [18, 36, 72, 144]   
    
    # fails
    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.10210435314326058, 0.01908335135505668, 0.003397637857506263, 0.0006108629798786026], 
    2=> [0.022317881538604513, 0.0020692023781013293, 0.0003012147578547065, 6.531242978686643e-05], 
    3=> [0.00581208410840217, 0.00031913440982102407, 1.7930206747935584e-05, 1.4794803301533312e-06], 
    4=> [0.0014901587820399847, 4.372974412079378e-05, 3.925927583211601e-06, 4.821736414461238e-07]
    [24, 48, 96, 192]

    #
    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.11235801760582836, 0.019192071389985216, 0.0032896982346112794, 0.0005711099012227056], 
    2=> [0.025761367918845826, 0.0024808173561728145, 0.0002289036681774387, 2.0662145595916426e-05], 
    3=> [0.006636801340510308, 0.00032083076808560947, 1.431040487285999e-05, 6.272836883552638e-07],
    4=> [0.0015102665335712348, 3.494283089815165e-05, 7.731761495816128e-07, 1.7247358287910526e-08]
    [8, 36, 72, 144]

    # the later two are used for the paper
    #WORKS!
    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)
    1=> [0.10020966619903729, 0.018667340329043245, 0.0032760114164995424, 0.0005706486826237747], 
    2=> [0.022155902610078892, 0.0017989201429291979, 0.00015698200933989407, 1.4102443951904156e-05], 
    3=> [0.005749269851153656, 0.0003035466818949205, 1.413981893579101e-05, 6.279995627332517e-07], 
    4=> [0.0014812300615435672, 3.173942687780366e-05, 6.18325689219852e-07, 1.2957302860337226e-08]
    [24, 48, 96, 192]

    #WORKS
    ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    N, rel_err_div = solve_weak_div(kind='curl', h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=True)
    1=> [0.07909216599888153, 0.013373946541813548, 0.0030332772941606273, 0.0008426281909444887], 
    2=> [0.024252536584951915, 0.002615128557671429, 0.0002429735301990908, 2.2580347568328123e-05],
    3=> [0.006043519994456351, 0.00022977966810808025, 9.129047623436424e-06, 3.9789419055201394e-07],
    4=> [0.0012043840167700198, 3.5162147891477066e-05, 8.902791639826253e-07, 2.030096118155681e-08]
    [24, 48, 96, 192]
    ==#


dof = [24, 48, 96, 192]

errors = Dict(
    1=> [0.10020966619903729, 0.018667340329043245, 0.0032760114164995424, 0.0005706486826237747], 
    2=> [0.022155902610078892, 0.0017989201429291979, 0.00015698200933989407, 1.4102443951904156e-05], 
    3=> [0.005749269851153656, 0.0003035466818949205, 1.413981893579101e-05, 6.279995627332517e-07], 
    4=> [0.0014812300615435672, 3.173942687780366e-05, 6.18325689219852e-07, 1.2957302860337226e-08]
    )

plt = plot(dpi = 2400, legend_columns=1, yscale = :log, xscale =:log, legend=:bottomleft, grid=true, xlabel = "N", xticks = (dof, latexstring.(dof)), ylabel="Error");
colors = palette(:tab10)
for d in 1:4
    plot!(plt, dof, errors[d], label = "deg = $d", color = colors[d], markershape = :circle, markerstrokewidth=0)    
    plot!(plt, dof, errors[d][2]*(dof[2]^(d+1))  * (1 ./ dof).^(d+1), label = "order $(d+1)", line=:dash, color = colors[d])
end
savefig(plt, "/home/schnack/Downloads/wdiv_conv.png")
