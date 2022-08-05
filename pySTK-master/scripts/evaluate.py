import argparse
import glob
import pandas as pd 
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import os


columns = ["Filename", "Dataset", "Algorithm", "Args", "Compression Ratio", "Time", "Num. Halos", "Total Halo Mass", "Top Halo Mass", "Top 5 Halo Mass"]
header_list = ["ID", "x", "y", "z", "No_0levelCells", "No_Vertices", "HaloMass", "IntegralAllFields"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store", required=False, default="halos_rho_1.h5", help="The original, unsampled halo data")
    # parser.add_argument("--sampled", action="store", required=True, help="The sampled halo data")
    parser.add_argument("-n", "--num_halos", action="store", required=False, type=int, default=10, help="The number of halos to compare")
    args = parser.parse_args()
    return args
# End of parse_arguments()


def split_sample_filename(filename):
    fname = filename.replace("hist_grad_rand", "Importance")
    fname = fname.replace("lcc_rand", "lccrand")
    parts = fname.split("_")
    print("filename: ", fname, " parts: ", parts)
    # halos NVB C009 l10n512 S12345T692 z5.hdf5 lcc True max 0.01
    arguments = "_".join(parts[-4:])
    algorithm = parts[-5]
    dataset = parts[-6]
    return arguments, algorithm, dataset
# End of split_sample_filename()


#### Print attributes and rank and store the top N halos by halo mass in a dataframe ####
def topNHalosByMass(df, N, filename):
    # N = int(input("Value of N for top N halos: "))
    # print("Top " + str(N) + " Halos based on mass:")
    # print("-----------------------")
    # print(df.nlargest(N,('HaloMass')))
    # print("Total Halo Mass: ", df['HaloMass'].sum(), " from ", df['HaloMass'].values.size, " halos")
    resdf = df.nlargest(N,('HaloMass'))
    # print("Top 5 Halo Mass: ", resdf['HaloMass'].sum())
    resdf["HaloRankID"] = range(1, len(resdf) + 1)
    resdf["StartingHaloRankID"] = " "
    arguments, algorithm, dataset = split_sample_filename(filename)
    return (arguments, algorithm, dataset, df['HaloMass'].values.size, df['HaloMass'].sum(), df.nlargest(1, ('HaloMass'))['HaloMass'].sum(), resdf['HaloMass'].sum())
# End of topNHalosByMass()


if __name__ == "__main__":
    # TODO:
    # - store compression results
    args = parse_arguments()
    data = list()
    #### Assign a header ####
    #### Read all the files accross timesteps with header ####
    for original in ["NVB_C009_l10n512_S12345T692_z42.h5", "NVB_C009_l10n512_S12345T692_z54.h5", "NVB_C009_l10n512_S12345T692_z5.h5"]:
        print("Processing results on: ", original)
        orig_df = pd.read_csv("halos_{0}".format(original), names=header_list, delim_whitespace=True)
        orig_size = os.path.getsize(original)
        # arguments, algorithm, dataset, num_halos, total_mass, largest_mass, top_5_mass = topNHalosByMass(orig_df, 5, original)
        _, _, _, num_halos, total_mass, largest_mass, top_5_mass = topNHalosByMass(orig_df, 5, original)
        dataset = original.split("_")[-1]
        algorithm = "None"
        arguments = "None"
        data.append([original, dataset, algorithm, arguments, 1.0, 0.0, num_halos, total_mass, largest_mass, top_5_mass])
        for alg in ["lcc", "hist_grad_rand"]:  # , "lcc_rand"]:
            for adaptive, function in [("True", "max"), ("False", "entropy")]:
                for samplesize in ["0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4"]:
                    print("algorithm: ", alg)
                    filename = "halos_{0}_{1}_{2}_{3}_{4}".format(original, alg, adaptive, function, samplesize)
                    try:
                        sample_df = pd.read_csv(filename, names=header_list, delim_whitespace=True)
                    except FileNotFoundError as e:
                        print("ERROR: Could not find file {0}. Skipping".format(filename))
                        continue
                    samplesize2 = samplesize
                    if samplesize == "0.00001":
                        samplesize2 = "1e-05"
                    elif samplesize == "0.00005":
                        samplesize2 = "5e-05"
                    elif samplesize == "0.0001":
                        samplesize2 = "1e-04"
                    # elif samplesize == "0.0005":
                    #     samplesize2 = "5e-04"
                    sample_size = os.path.getsize("sampled_output_{0}_{1}_{2}_{3}_{4}_archive{5}_{1}.zip".format(original, alg, adaptive, function, samplesize, samplesize2))
                    compression_ratio = orig_size / sample_size
                    print("filename: ", filename, " compression ratio: ", compression_ratio)
                    arguments, algorithm, dataset, num_halos, total_mass, largest_mass, top_5_mass = topNHalosByMass(sample_df, 5, filename)
                    data.append([filename, dataset, algorithm, arguments, compression_ratio, 0.0, num_halos, total_mass, largest_mass, top_5_mass]) 
    compression_df = pd.DataFrame()
    for original in ["NVB_C009_l10n512_S12345T692_z42.h5", "NVB_C009_l10n512_S12345T692_z54.h5", "NVB_C009_l10n512_S12345T692_z5.h5"]:
        compression_filename = original.split("_")[-1].replace("h5", "hdf5.csv")
        compression_part = pd.read_csv("metrics_{0}".format(compression_filename), skipinitialspace=True)
        compression_part["Filename"] = original
        compression_df = compression_df.append(compression_part)
    compression_df.columns = compression_df.columns.str.strip()
    compression_df["name"] = compression_df["name"].str.replace("__", "_")
    datafiles = glob.glob("halos_SZ*.h5")
    datafiles.extend(glob.glob("halos_zfp*.h5"))
    print(datafiles)
    for datafile in datafiles:
        # print("datafile: ", datafile)
        compression_part = pd.read_csv(datafile, names=header_list, delim_whitespace=True)
        _, _, _, num_halos, total_mass, largest_mass, top_5_mass = topNHalosByMass(compression_part, 5, datafile)
        arguments = "_".join(datafile.split("_")[2:-5])
        algorithm = datafile.split("_")[1]
        dataset = datafile.split("_")[-1].replace("hdf5", "h5")
        filename = "_".join(datafile.split("_")[-5:]).replace("hdf5", "h5")
        # print("filenames: ", compression_df["Filename"])
        # print("filename: ", filename)
        _compdf = compression_df.loc[compression_df["Filename"] == filename]
        # print("compdf: ", _compdf["name"])
        rowname = "{0}_{1}".format(algorithm, arguments)[:-1]
        # print(rowname)
        _compdf = _compdf.loc[_compdf["name"] == rowname]
        # print("compdf: ", _compdf)
        row = _compdf.iloc[0]
        data.append([filename, dataset, algorithm, arguments, row["Compression Ratio"], 0.0, num_halos, total_mass, largest_mass, top_5_mass])
    
    results = pd.DataFrame(data, columns=columns)
    # results.append(compression_df)
    print("writing results to compression_comparison_results.csv")
    results.to_csv("compression_comparison_results.csv")

