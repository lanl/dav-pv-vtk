# pySTK

This is python-based sampling toolkit. Uses pymp to perform sampling and reconstruction in parallel on data blocks.

## Installation and Running

To install pystk, first clone the repo

```
git clone --recursive https://gitlab.lanl.gov/wanyef/pystkgraph.git
```

Then, download the dependencies using ENV.yml (or ENV_SPECIFIC.yml if the first option fails).

```
conda env create -f ENV.yml
```

This will create an `intellSampler` anaconda environment, which will be used to run the sampling code.

If on darwin, use the installation script provided to download additional dependencies (namely reeber and 
VizAly-Foresight without Isabela support) for benchmarking and evaluating the sampling code, as well as to copy a set
of NYX datasets for the same purpose.

```
cd code/scripts
bash setup.sh
```

To run a series of experiments on darwin, use 
```
bash run_darwin.sh && evaluate.py
```

This will output a csv file named 
`compression_comparison_results`, which can be used to compare sampling and compression results.

## Notes

`reconstruction_serial.py` is not fully implemented, and doesn't work well at small sample sizes. Use the `_pymp.py` script versions instead.

Reconstruction methods 5 and 6 use the `scipy.interpolate.griddata` method with the `nearest` and `linear` parameters, respectively. For these methods, include the `--store_corners` options to the sampling command.

The `--blk_dims` parameter passed to `sampling_pymp.py` needs to perfectly divide the input grid dimensions to avoid dead space on some edges of the grid.

The `--sparse` command does not need to be passed in to the `lcc` sampling method even though it is graph-based. The 
sparsity of the graph is automatically determined for `lcc` using the `percentage` parameter.

## Usage

To perform sampling, run `python sampling_pymp.py --method <method> --percentage <[0.0..1.0]> --input <path_to_vti/h5_file> <...args>`

If using adaptive sampling, add `--adaptive --function <min/max/entropy>` to the end of the previous command.

To perform reconstruction, run `python reconstruction_pymp.py --sampledir _sampled_output_<method>_ --vtiout True --recontype <[1..7]>`

The output will be saved in `_sampled_output_<method>__reconstructed`.

For more usage examples, refer to `scripts/run_darwin.sh`
