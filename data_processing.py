#---------------------#
### DATA PROCESSING ###
#---------------------#
import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

def process_data(data_file):
    ### Read .tar file ###
    pd.set_option('display.max_columns', None)
    tar = tarfile.open(data_file,"r:gz")
    tar.extractall()
    tar = tarfile.open(data_file,"r:gz")
    tn = tar.next()
    tn = tar.next()
    pq.read_schema(tn.name)
    df = pd.read_parquet(tn.name)

    ### Create data frame with subset of variables ###
    dfs = []
    for member in tar:
        if member.isreg():
            df_temp = pd.read_parquet(member.name,columns = features)
            dfs.append(df_temp)

    df_all = pd.concat(dfs)
    print('Converted ' + str(data_file))
    tar = None

    ### Subset data frames
    df_subset = df_all.iloc[::-1]
    df_subset = df_subset.iloc[1: , :]
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset['evar83'] != '']
    df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
    df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
    df_subset_1 = df_subset_1[~df_subset_1['visitoridlow'].isin(df_subset_2['visitoridlow'].unique())]
    df_subset_1 = df_subset_1.drop_duplicates(subset=['visitoridlow'])
    df_subset_2 = df_subset_2.drop_duplicates(subset=['visitoridlow'])
    df_subset_1['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
    df_subset_2['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
    print('Subsetted ' + str(data_file))
    df_subset = None

    return df_subset_1, df_subset_2
