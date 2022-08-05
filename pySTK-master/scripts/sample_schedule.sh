#!/usr/bin/bash

set -o xtrace

## algadafun = algorithm adaptive function
## algorithm = sampling algorithm = anything supported in pystk. here, testing lcc and hist_grad_rand (Importance)
## adaptive = either empty string '' or --adaptive if adaptive sampling is to be used
## function = block weighting function to use for adaptive sampling. if adaptive = '' function doesn't matter. Currently, max, min, sum and entropy are supported

for algadafun in "lcc,,entropy" "lcc,--adaptive,max" "hist_grad_rand,,entropy" ; do
    IFS=',' read algorithm adaptive function <<< "${algadafun}"
    echo "algorithm = $algorithm adaptive = $adaptive function = $function"
    for percentage in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 ; do
        for data in ./NVB*.h5 ; do
            echo $data
            sbatch --export=ALL,percentage=${percentage},algorithm=${algorithm},adaptive=${adaptive},function=${function},data=${data} "./sample.sbatch"
        done
    done
done


