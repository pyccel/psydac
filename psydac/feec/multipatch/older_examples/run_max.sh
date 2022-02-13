#!/bin/bash

# degree
p_min=2
p_max=4
# mesh: K = (2**k)  (= nb cells per dimension, in every patch)
k_min=1
k_max=5

## -------- -------- -------- -------- -------- -------- -------- -------- --------
## problem
#source="ring_J"
#domain="pretzel_f"
#eta="-64"
## ref sol: p=6, k=5 (ncells = 32)

source="manu_J"
domain="curved_L_shape"
#domain="square_8"
#domain="square_9"
eta="-63"
problem=" --problem source_pbm --domain "$domain" --source "$source" --eta "$eta" "
## -------- -------- -------- -------- -------- -------- -------- -------- --------

## -------- -------- -------- -------- -------- -------- -------- -------- --------
## penalization regime
gamma=1
pr="0"
penalization=" --gamma "$gamma" --penal_regime "$pr" "
## -------- -------- -------- -------- -------- -------- -------- -------- --------

plots="--no_plots"

for (( deg=p_min; deg<p_max+1; deg++ ))
do
  for (( k=k_min; k<k_max+1; k++ ))
  do
    nc=$((2**k))

    ## conga scheme
    scheme=" --method conga --proj_sol --geo_cproj "
    cmd="python3 psydac/feec/multipatch/maxwell_pbms.py "$nc" "$deg" "$problem" "$scheme" "$penalization" "$plots
    echo "$ "$cmd
    $cmd

    ## nitsche scheme
    scheme=" --method nitsche "
    cmd="python3 psydac/feec/multipatch/maxwell_pbms.py "$nc" "$deg" "$problem" "$scheme" "$penalization" "$plots
    echo "$ "$cmd
    $cmd

  done
done
printf "\n"
