#! /usr/bin/bash

git clone https://github.com/AMReX-codes/amrex.git
git clone https://github.com/mrzv/reeber.git
git clone https://github.com/diatomic/diy.git

cd amrex
mkdir build
mkdir install
cd build
cmake .. -DENABLE_MPI=ON -DENABLE_AMRDATA=ON -DCMAKE_BUILD_TYPE=Release -DENABLE_DP=OFF -DENABLE_LINEAR_SOLVERS=OFF -DCMAKE_INSTALL_PREFIX=../install
make -j12
make install

cd ../../reeber
mkdir build
cd build/
cmake -DDIY_INCLUDE_DIR=../../diy/include -DAMReX_DIR=/home/biswas/work/halo_finder/hdf5_halo/amrex/install/lib/cmake/AMReX -Damrex=ON  ../
cd examples/amr-connected-components
make -j12

echo "Reeber installed! To run reeber on a dataset, use: "
echo "/reeber/build/examples/amr-connected-components/amr_connected_components_float -b 64 -n -w -f native_fields/baryon_density ${dataset} none none \"halos_$(basename ${dataset})\"" 
echo "This will output the halo information to ./halos_${dataset}"

# Copy NYX hdf5 output files to this directory
cp /projects/exasky/data/NYX/highz/512/NVB*.hdf5 ./