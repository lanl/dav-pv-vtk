#! /usr/bin/bash

echo "===== Scheduling sampling scripts on darwin ====="
bash sample_schedule.sh

echo "===== Performing compression runs ====="
module load gcc openmpi
for f in ./compression_test* ; do 
    mpirun -np 4 ./VizAly-Foresight/build/CBench $f
done

for f in ./SZ*.h5 ; do
   ./reeber/build/examples/amr-connected-components/amr_connected_components_float -b 64 -n -w -f native_fields/baryon_density $f none none "halos_$(basename ${f})" 
done

for f in ./zfp*.h5 ; do
   ./reeber/build/examples/amr-connected-components/amr_connected_components_float -b 64 -n -w -f native_fields/baryon_density $f none none "halos_$(basename ${f})" 
done

echo "Done with compression! Wait until sampling scripts are done, then run `conda activate intellSampler && python HaloMatcher.py`"
echo "Use `squeue -u ${USER}` to find out if any sampling scripts are still running."