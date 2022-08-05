import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5)
pd.options.mode.chained_assignment = None  # default='warn'

ARG_REGEX = {
    r"abs.*": "abs",
    r"pwrel.*": "pwrel",
    r"True_max.*": "adaptive (max)",
    r"False.*": "",
    r"prec.*": "prec"
}

SELECTED = ["LCC Adaptive", "SZ-abs", "zfp-abs", "LCC 1.0 Adaptive", "LCC 5.0 Adaptive"]

ALG_MAP = {
    "lcc-0.0-adaptive (max)": "LCC 0.0 Adaptive",
    "lcc-0.25-adaptive (max)": "LCC 0.25 Adaptive",
    "lcc-1.0-adaptive (max)": "LCC 1.0 Adaptive",
    "lcc-5.0-adaptive (max)": "LCC 5.0 Adaptive",
    "lcc-": "LCC",
    "lcc-adaptive (max)": "LCC Adaptive",
    "Importance-": "Importance",
    "lccrand-": "LCC-Rand",
    "lccrand-adaptive (max)": "LCC-Rand Adaptive",
    "SZ-abs": "SZ-abs",
    "SZ-pwrel": "SZ-pwrel",
    "zfp-abs": "zfp-abs",
    "zfp-prec": "zfp-prec",
    "zfp-pwrel": "zfp-pwrel"
}


SELECTED_SAMPLING = ["LCC Adaptive", "Importance", "LCC"]
# SELECTED_SAMPLING = ["LCC 0.0 Adaptive", "LCC 0.25 Adaptive", "LCC 1.0 Adaptive", "LCC 5.0 Adaptive"]

if __name__ == "__main__":
    csv_filename = "compression_results7.csv"
    df = pd.read_csv(csv_filename, skipinitialspace=True)
    df["Algorithm"] = df["Algorithm"].str.replace("histgradrand", "Importance")
    # lccdf = df.loc[df["Algorithm"] == "lcc"]
    # lccdf["Tolerance"] = lccdf["Args"].map(lambda x: x.split("_")[-1])
    # lccdf["Algorithm"] = lccdf[["Algorithm", "Tolerance"]].agg('-'.join, axis=1)
    # df.loc[df["Algorithm"] == "lcc", "Algorithm"] = lccdf["Algorithm"]
    df["Args"] = df["Args"].replace(ARG_REGEX, regex=True)
    datasets = df["Dataset"].unique()
    # SAMPLE RATE RESULTS
    for dataset in datasets:
        print(dataset)
        subdf = df.loc[df["Dataset"] == dataset]
        baseline_row = subdf.loc[df["Algorithm"] == "None"].iloc[0]
        subdf["Algorithm"] = subdf[["Algorithm", "Args"]].agg('-'.join, axis=1)
        subdf["Sampling Rate"] = subdf["Filename"].map(lambda x: x.split("_")[-1])
        subdf = subdf.loc[subdf["Sampling Rate"].str.len() < 6]
        subdf["Algorithm"] = subdf["Algorithm"].map(ALG_MAP)
        subdf = subdf[subdf["Algorithm"].isin(SELECTED_SAMPLING)]
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top Halo Mass"] - baseline_row["Top Halo Mass"]) / baseline_row["Top Halo Mass"]
        ax = sns.lineplot(x="Sampling Rate", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Mass of Largest Halo {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_sampling_top_halo.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top 5 Halo Mass"] - baseline_row["Top 5 Halo Mass"]) / baseline_row["Top 5 Halo Mass"]
        ax = sns.lineplot(x="Sampling Rate", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Mass of Largest 5 Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_sampling_top_5_halos.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Total Halo Mass"] - baseline_row["Total Halo Mass"]) / baseline_row["Total Halo Mass"]
        ax = sns.lineplot(x="Sampling Rate", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Total Mass of All Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_sampling_all_halos.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Num. Halos"] - baseline_row["Num. Halos"]) / baseline_row["Num. Halos"]
        ax = sns.lineplot(x="Sampling Rate", y="% Difference", data=subdf, hue="Algorithm")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Number of Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_sampling_num_halos.png".format(dataset))
        plt.cla()
    # COMPRESSION RATE RESULTS
    for dataset in datasets:
        subdf = df.loc[df["Dataset"] == dataset]
        baseline_row = subdf.loc[subdf["Algorithm"] == "None"].iloc[0]
        subdf["Algorithm"] = subdf[["Algorithm", "Args"]].agg('-'.join, axis=1)
        subdf["Sampling Rate"] = subdf["Filename"].map(lambda x: x.split("_")[-1])
        subdf = subdf.loc[subdf["Sampling Rate"].str.len() <= 6]
        print(subdf["Algorithm"].unique())
        subdf["Algorithm"] = subdf["Algorithm"].map(ALG_MAP)
        subdf = subdf[subdf["Algorithm"].isin(SELECTED_SAMPLING)]
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top Halo Mass"] - baseline_row["Top Halo Mass"]) / baseline_row["Top Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-5, 100])
        ax.set_title("Mass of Largest Halo {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_cr_sampling_top_halo.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top 5 Halo Mass"] - baseline_row["Top 5 Halo Mass"]) / baseline_row["Top 5 Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-5, 100])
        ax.set_title("Mass of Largest 5 Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_cr_sampling_top_5_halos.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Total Halo Mass"] - baseline_row["Total Halo Mass"]) / baseline_row["Total Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        # ax.set_xscale("log")
        ax.set_ylim([-5, 100])
        ax.set_title("Total Mass of All Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_cr_sampling_all_halos.png".format(dataset))
        plt.cla()
    # COMPARISON WITH COMPRESSION
    for dataset in datasets:
        subdf = df.loc[df["Dataset"] == dataset]
        # print(subdf["Compression Ratio"])
        baseline_row = subdf.loc[df["Algorithm"] == "None"].iloc[0]
        subdf = subdf.loc[subdf["Compression Ratio"] > 10]
        # print(subdf)
        subdf["Algorithm"] = subdf[["Algorithm", "Args"]].agg('-'.join, axis=1)
        subdf["Algorithm"] = subdf["Algorithm"].map(ALG_MAP)
        subdf = subdf[subdf["Algorithm"].isin(SELECTED)]
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top Halo Mass"] - baseline_row["Top Halo Mass"]) / baseline_row["Top Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Mass of Top Halo {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_top_halo.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Top 5 Halo Mass"] - baseline_row["Top 5 Halo Mass"]) / baseline_row["Top 5 Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Mass of Top 5 Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_top_5_halos.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Total Halo Mass"] - baseline_row["Total Halo Mass"]) / baseline_row["Total Halo Mass"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Total Mass of All Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_all_halos.png".format(dataset))
        plt.cla()
        #########
        subdf["% Difference"] = 100 * np.abs(subdf["Num. Halos"] - baseline_row["Num. Halos"]) / baseline_row["Num. Halos"]
        ax = sns.lineplot(x="Compression Ratio", y="% Difference", data=subdf, hue="Algorithm")
        ax.set_xscale("log")
        ax.set_ylim([-0.1, 1])
        ax.set_title("Number of Halos {0}".format(dataset))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{0}_num_halos.png".format(dataset))
        plt.cla()