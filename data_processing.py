#---------------------#
### IMPORT PACKAGES ###
#---------------------#
import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

#------------------------------#
### DATA PROCESSING PIPELINE ###
#------------------------------#
def read_data(data_file):
    pd.set_option('display.max_columns', None)
    tar = tarfile.open(data_file, 'r:gz')
    tar.next()
    tn = tar.next()
    tn1 = tar.next()
    tn2 = tar.next()
    tn3 = tar.next()
    tn4 = tar.next()
    tn5 = tar.next()
    tn6 = tar.next()
    tn7 = tar.next()
    tn8 = tar.next()

    df = pd.read_parquet(tn.name)
    df1 = pd.read_parquet(tn1.name)
    df2 = pd.read_parquet(tn2.name)
    df3 = pd.read_parquet(tn3.name)
    df4 = pd.read_parquet(tn4.name)
    df5 = pd.read_parquet(tn5.name)
    df6 = pd.read_parquet(tn6.name)
    df7 = pd.read_parquet(tn7.name)
    df8 = pd.read_parquet(tn8.name)

    df = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8])
    print('Converted ' + str(data_file))

    return df

def see_values(data, category):
    counts = pd.DataFrame(data.groupby([category])[category].count())
    counts.rename(columns={category: 'counts'}, inplace=True,)
    counts = counts.sort_values(['counts'], ascending=[0])
    graph = counts.reset_index()
    return graph

def feature_importance(df):
    ### 1st subset: first click ###
    ids_purchase = (df[df['checkoutthankyouflag'] == 1])['visitid'].drop_duplicates()
    ids_purchase = ids_purchase.tolist()
    df_first_visit = df[df['hit_time_gmt']==df['visitstarttimegmt']]
    df_first_visit = df_first_visit.sort_values(by='hit_time_gmt', inplace = False)
    df_first_visit = df_first_visit.drop_duplicates(subset=['visitid'], keep = "first", inplace = False)

    target = list()
    for ids in df_first_visit['visitid']:
        if ids in ids_purchase:
            target.append("1")
        else:
            target.append("0")
            
    df_first_visit.insert(0, "target", target)
    df_first_visit['post_evar23'].head().to_list()
    purchase_percent = pd.DataFrame()
    df_first_visit_purchases = df_first_visit[df_first_visit['visitid'].isin(ids_purchase)]

    for names in df_first_visit_purchases.columns:
        purchase_percent[names] = [100 * (sum(see_values(df_first_visit_purchases, names)['counts'])) / len(df_first_visit_purchases)]
    
    print('Finished 1st subset')

    ### 2nd subset: remove extraneous features ###
    features = list()

    for names in purchase_percent.columns:
        if (purchase_percent[names][0]) > 5:
            features.append(names)

    df_first_subset = df_first_visit[features]

    removed = ['hitdatahistorymkey', 'filename', 'linenumber', 'visitoridhigh', 'visitoridlow',
               'visitdatetime', 'visitdate', 'visitmonth', 'visitid', 'fiscalyear', 'fiscalweeknumber', 
               'fiscalmonthnumber', 'visitstarttimegmt', 'firsthittimegmt', 'lasthittimegmt', 
               'initialloaddate', 'updatedloaddate', 'checkoutthankyouflag', 'cookieid']

    for variables in df_first_subset.columns:
        if variables in removed:
            features.remove(variables)

    df_second_subset = df_first_subset[features]
    df_second_subset = df_second_subset.fillna("None")

    print('Finished 2nd subset')

    ### 3rd subset: variance threshold ###
    ord_enc = OrdinalEncoder()
    variance_train = df_second_subset.loc[:, df_second_subset.columns != 'target']
    variance_transformed = ord_enc.fit_transform(variance_train)

    var_thr = VarianceThreshold(threshold = 0.01) 
    var_thr.fit(variance_transformed)
    df_third_subset = (variance_train.loc[:,var_thr.get_support()])
    df_third_subset.insert(0, "target", target)
    features = df_third_subset.columns.to_list()

    print('Finished 3rd subset')

    ### 4th subset: random forest ###
    clf = RandomForestClassifier(class_weight='balanced')

    x_train, x_test, y_train, y_test = train_test_split(df_third_subset.iloc[:,:], 
                                                        df_third_subset['target'], 
                                                        test_size=0.20)

    x_train_1 = x_train[x_train['target'] == "1"]
    y_train_1 = x_train_1['target']
    x_train_2 = x_train[x_train['target'] == "0"].sample(n=len(x_train_1))
    y_train_2 = x_train_2['target']
    x_train = (pd.concat([x_train_1, x_train_2])).iloc[:, 1:]
    y_train = pd.concat([y_train_1, y_train_2])
    x_test = x_test.iloc[:, 1:]

    ord_enc = OrdinalEncoder()
    x_train = ord_enc.fit_transform(x_train)
    y_train = ord_enc.fit_transform(y_train.to_numpy().reshape(-1,1))
    x_test = ord_enc.fit_transform(x_test)
    y_test = ord_enc.fit_transform(y_test.to_numpy().reshape(-1,1))

    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    labels = ['0', '1']

    feat_list = []
    total_importance = 0

    for feature in zip(df_third_subset.iloc[:,1:].columns, clf.feature_importances_):
        feat_list.append(feature)
        total_importance += feature[1]

    included_feats = []

    for feature in zip(df_third_subset.iloc[:,1:].columns, clf.feature_importances_):
        if feature[1] > .05:
            included_feats.append(feature[0])

    df_imp = pd.DataFrame(feat_list, columns =['FEATURE', 'IMPORTANCE']).sort_values(by='IMPORTANCE', ascending=False)
    df_imp['CUMSUM'] = df_imp['IMPORTANCE'].cumsum()

    return df_third_subset, target, df_imp

def data_n_features(subsetted_data, df_imp, n, target):
    n_features = df_imp.iloc[0:n, 0]
    df_fourth_subset = subsetted_data[n_features]
    df_fourth_subset.insert(0, "target", target)
    print('Finished 4th subset')
    return df_fourth_subset, n_features