import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def data_preprocessing(path, data_name, is_mat=False, need_transform=False):
    if is_mat:
        data = loadmat(r"C:\Users\User\Downloads\charliec443 scikit-feature master skfeature-data\GLIOMA.mat")
        X = data['X']
    else:
        data = pd.read_csv(r"C:\Users\User\Downloads\CLL.csv")
    if need_transform:
        data = data.transpose()
        all_columns = data.columns.tolist()
        all_columns[-1] = 'y'
        data.columns = all_columns
        print(data.y.unique())
        print(data.shape)
        data = data.astype({"y": int})
        data.reset_index(drop=True, inplace=True)
        
    # nan value complete
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(data['X'])

    # remove variance 0 features
    thresholder = VarianceThreshold(threshold=0.0)
    X = thresholder.fit_transform(X)

    # normalize the data
    pt = PowerTransformer()
    X = pt.fit_transform(X)

    # build data frame
    df = pd.DataFrame(X)
    df = df.assign(y=data['Y'])

    # if there is nan value in y create it as new class
    if df.isnull().values.any():
        if df['y'].isnull().values.any():
            print("inside if")

    # if there is categorical feature incode them.
    num_cols = df._get_numeric_data().columns
    if len(num_cols) < len(df.columns):
        df.apply(LabelEncoder().fit_transform)

    # save the new dataframe to csv file
    df.to_csv('{0}.csv'.format(data_name), index=False)
