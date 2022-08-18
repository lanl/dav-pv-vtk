import argparse
import os
import pandas as pd

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--Reeberout_originaldata', action="store", required=True, help="reeber output of original Nyx input data")
parser.add_argument('--Reeberout_recondata', action="store", required=True, help="reeber output of reconstructed data")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #### Assign a header for the Halo output files ####
    header_list = ["ID", "x", "y", "z", "No_0levelCells", "No_Vertices", "HaloMass", "IntegralAllFields"]
    #### Read the halo output for the original input file with a header ####
    df = pd.read_csv(args.Reeberout_originaldata, names=header_list, delim_whitespace=True)
    #### Read the halo output for the reconstructed output with a header ####
    df_recon = pd.read_csv(args.Reeberout_recondata, names=header_list, delim_whitespace=True)

    #### Sort both the files ####
    df = df.sort_values(by=['HaloMass'], ascending=False)
    df_recon = df_recon.sort_values(by=['HaloMass'], ascending=False)

    #### Print percentage of difference of mass of the largest Halo
    print("% difference in mass of largest Halo",(df['HaloMass'].head(1).sum() - df_recon['HaloMass'].head(1).sum())/df['HaloMass'].head(1).sum()*100)


    #### Print percentage of difference of mass of largest 5 Halos 
    print("% difference in mass of largest 5 Halos",(df['HaloMass'].head(5).sum() - df_recon['HaloMass'].head(5).sum())/df['HaloMass'].head(5).sum()*100)

    #### Print percentage of difference in the total Mass of Halos

    print("% difference in total mass of Halos",(df['HaloMass'].sum() - df_recon['HaloMass'].sum())/df['HaloMass'].sum()*100)

    #### Print percentage of difference in the total number of Halos

    print("% difference in total number of Halos", (len(df['HaloMass']) - len(df_recon['HaloMass']))/len(df['HaloMass'])*100)
