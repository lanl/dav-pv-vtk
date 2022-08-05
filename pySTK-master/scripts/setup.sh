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
# Extract just the baryon_density field from said file, save as .h5 file
for f in NVB_C009_l10n512_S12345T692_z54.hdf5 NVB_C009_l10n512_S12345T692_z5.hdf5 NVB_C009_l10n512_S12345T692_z42.hdf5 ; do
    python ../code/dds/h5tovti.py --input $f
for f in NVB_C009_l10n512_S12345T692_z54.vti NVB_C009_l10n512_S12345T692_z5.vti NVB_C009_l10n512_S12345T692_z42.vti ; do
    python ../code/dds/vtitoh5.py --infile $f --noexp
rm ./NVB*log*  # remove log-scaled files

echo "Ready for sampling!"

# Try the github first, if not copy the files from my directory. It's on darwin anyway, should work
function cbench_github_install() {
    module load gcc openmpi
    git clone https://github.com/lanl/VizAly-Foresight.git
    cd VizAly-Foresight
    source evn_scripts/VizAly-CBench.bash.darwin
    source buildDependencies.sh
    source build.sh
}

function cbench_copy_install() {
    cp /projects/exasky/wanyef/compression/VizAly-Foresight.tar.gz ./
    tar -xvf VizAly-Foresight.tar.gz
}

cbench_github_install || cbench_copy_install

echo "Fixing CBench compression input file template"
for f in ./compression*.json ; do
    sed -i -e "s#/_template_#$(pwd)#" $f
done
