#!/usr/bin/bash

args="--nthreads 40"

module load gcc openmpi miniconda3
source activate intellSampler
which python
conda activate intellSampler
which python
EXEDIR="../code"
EXE="${HOME}/.conda/envs/intellSampler/bin/python"

adaptive=$adaptive
percentage=$percentage
algorithm=$algorithm
function=$function
if [ -z "$adaptive" ]
then
    adaptivebool="False"
else
    adaptivebool="True"
fi

DATAFILE=$(basename ${data})
DATA=$data
BASENAME="${DATAFILE}_${algorithm}_${adaptivebool}_${function}_${percentage}"
SAMPLEDIR="sampled_output_${BASENAME}"
echo "Sampling ====="
$EXE ${EXEDIR}/sampling_pymp.py --blk_dims 16 --percentage ${percentage} --input ${DATA} --outpath ${SAMPLEDIR} --method ${algorithm} ${args} ${adaptive} --function ${function}
echo "Reconstruction ====="
$EXE ${EXEDIR}/reconstruction_pymp.py --sampledir ${SAMPLEDIR} --recontype 4 --vtiout
f="${SAMPLEDIR}/reconstructed_${BASENAME}_4.vti"
echo "Converting $f to h5 ====="
$EXE ${EXEDIR}/dds/vtitoh5.py --infile $f --noexp
f="${SAMPLEDIR}/reconstructed_${BASENAME}_4.h5"
echo "Running reeber ====="
./reeber/build/examples/amr-connected-components/amr_connected_components_float -b 64 -n -w -f native_fields/baryon_density ${f} none none "halos_${BASENAME}" 

