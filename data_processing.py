#---------------------#
### IMPORT PACKAGES ###
#---------------------#
import sys
#!conda install --yes --prefix {sys.prefix} category_encoders

import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from category_encoders import BinaryEncoder
from category_encoders import WOEEncoder
from category_encoders import LeaveOneOutEncoder

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
    tn9 = tar.next()
    tn10 = tar.next()
    tn11 = tar.next()
    tn12 = tar.next()

    df = pd.read_parquet(tn.name)
    df1 = pd.read_parquet(tn1.name)
    df2 = pd.read_parquet(tn2.name)
    df3 = pd.read_parquet(tn3.name)
    df4 = pd.read_parquet(tn4.name)
    df5 = pd.read_parquet(tn5.name)
    df6 = pd.read_parquet(tn6.name)
    df7 = pd.read_parquet(tn7.name)
    df8 = pd.read_parquet(tn8.name)
    df9 = pd.read_parquet(tn9.name)
    df10 = pd.read_parquet(tn10.name)
    df11 = pd.read_parquet(tn11.name)
    df12 = pd.read_parquet(tn12.name)

    df = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12])

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
    df_first_visit['eventlistlength'] = df_first_visit.eventlist.str.count(',')
    df_first_visit['productlistlength'] = df_first_visit.productlist.str.count(',')
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

    removed = ['eventlist', 'hitdatahistorymkey', 'filename', 'linenumber', 'visitoridhigh', 'visitoridlow', 'hit_time_gmt',
               'visitdatetime', 'postttimeinfo', 'visitdate', 'visitmonth', 'visitid', 'fiscalyear', 'fiscalweeknumber', 
               'fiscalmonthnumber', 'visitstarttimegmt', 'checkoutthankyouflag', 'cookieid', 'lasthittimegmt', 'firsthittimegmt', 
               'lastpurchasetimegmt', 'updatedloaddate', 'initialloaddate', 'productlist']

    for variables in df_first_subset.columns:
        if variables in removed:
            features.remove(variables)

    df_second_subset = df_first_subset[features]
    
    numeric = ['visitnumber', 'postbrowserheight', 'postbrowserwidth', 'lastpurchasenum', 'addonsymal', 'cdedspomodel',
               'myaccountengagement', 'post_evar46', 'evar83', 'post_evar30', 'eventlistlength', 'productlistlength']
    
    for names in df_second_subset.columns:
        if names in numeric:
            df_second_subset[names] = pd.to_numeric(df_second_subset[names]).convert_dtypes()
    
    for names in df_second_subset.columns:
        if (pd.api.types.is_numeric_dtype(df_second_subset[names])) != True:
            df_second_subset[names] = df_second_subset[names].fillna("None")

    print('Finished 2nd subset')

    ### 3rd subset: variance threshold ###
    ord_enc = OrdinalEncoder()
    variance_train_categorical = df_second_subset.loc[:, ~df_second_subset.columns.isin(numeric)]
    variance_transformed_categorical = ord_enc.fit_transform(variance_train_categorical)

    var_thr = VarianceThreshold(threshold = 0.01) 
    var_thr.fit(variance_transformed_categorical)

    temp_features = (variance_train_categorical.loc[:,var_thr.get_support()]).columns.to_list()
    df_third_subset = df_second_subset[temp_features]
    df_third_subset = df_third_subset.join(df_second_subset[numeric])

    processed_df = df_third_subset

    features = df_third_subset.columns.to_list()

    print('Finished 3rd subset')

    ### 4th subset: random forest ###
    clf = RandomForestClassifier(class_weight='balanced', random_state=148)

    x_train, x_test, y_train, y_test = train_test_split(df_third_subset.iloc[:,:], 
                                                        df_third_subset['target'], 
                                                        test_size=0.10)

    x_train_1 = x_train[x_train['target'] == "1"]
    y_train_1 = x_train_1['target']
    x_train_2 = x_train[x_train['target'] == "0"].sample(n=len(x_train_1), random_state=148)
    y_train_2 = x_train_2['target']
    x_train = (pd.concat([x_train_1, x_train_2])).iloc[:, 1:]
    y_train = pd.concat([y_train_1, y_train_2])
    x_test = x_test.iloc[:, 1:]

    x_train_categorical = x_train.loc[:, ~x_train.columns.isin(numeric)]
    x_train_numerical = x_train.loc[:, x_train.columns.isin(numeric)]
    x_test_categorical = x_test.loc[:, ~x_test.columns.isin(numeric)]
    x_test_numerical = x_test.loc[:, x_test.columns.isin(numeric)]

    numerical_columns = x_train_numerical.columns.to_list()

    # Ordinal encoding
    ord_enc = OrdinalEncoder()
    y_train_transformed = ord_enc.fit_transform(y_train.to_numpy().reshape(-1,1))
    y_test_transformed = ord_enc.fit_transform(y_test.to_numpy().reshape(-1,1))
    
    # Leave-one-out encoding
    loo_enc = LeaveOneOutEncoder()
    loo_enc.fit(x_train_categorical, y_train_transformed)
    x_train_categorical_transformed = loo_enc.transform(x_train_categorical)
    x_test_categorical_transformed = loo_enc.transform(x_test_categorical)
    
    # Min-max scaling
    minmax = MinMaxScaler()
    x_train_numerical_transformed = minmax.fit_transform(x_train_numerical)
    x_test_numerical_transformed = minmax.transform(x_test_numerical)

    # Create data frames of numerical and categorical variables
    x_train_numerical_transformed = pd.DataFrame(x_train_numerical_transformed, columns = numerical_columns)
    x_train_numerical_transformed = x_train_numerical_transformed.fillna(-999)
    x_test_numerical_transformed = pd.DataFrame(x_test_numerical_transformed)
    x_test_numerical_transformed = x_test_numerical_transformed.fillna(-999)

    x_train_numerical_transformed = x_train_numerical_transformed.astype('float32')
    x_train_categorical_transformed = x_train_categorical_transformed.astype('float32')
    x_test_numerical_transformed = x_test_numerical_transformed.astype('float32')
    x_test_categorical_transformed = x_test_categorical_transformed.astype('float32')

    x_train_total_transformed = np.concatenate([x_train_categorical_transformed.to_numpy(), x_train_numerical_transformed.to_numpy()], axis=1)
    x_test_total_transformed = np.concatenate([x_test_categorical_transformed.to_numpy(), x_test_numerical_transformed.to_numpy()], axis=1)

    # Train the random forest classifier
    clf.fit(x_train_total_transformed, y_train_transformed)
    y_predicted = clf.predict(x_test_total_transformed)
    labels = ['0', '1']

    print('Trained random forest')

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

    n_features = df_imp.iloc[0:60, 0]

    return processed_df, n_features, target