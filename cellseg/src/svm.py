import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


def get_data(X_list, y_list, indexes):
    """
    Concatenate X and y dataframes based on given indexes.

    Parameters:
    - X_list (list): List of X dataframes.
    - y_list (list): List of y dataframes.
    - indexes (list): List of indexes specifying which dataframes to concatenate.

    Returns:
    - pd.DataFrame: Concatenated X dataframe.
    - pd.DataFrame: Concatenated y dataframe.
    """

    X = pd.DataFrame()
    y = pd.DataFrame()
    
    for index in indexes:
        X = pd.concat([X, X_list[index]], axis=0)
        y = pd.concat([y, y_list[index]], axis=0)

    return X, y


def cross_validation_LinearSVC_cytoplasm(Xy_full_paths, X_clustered, y_clustered, output_path):
    """
    Perform cross-validation with LinearSVC model.

    Parameters:
    - Xy_full_paths (list): List of paths to CSV files containing full data.
    - X_clustered (list): List of X dataframes.
    - y_clustered (list): List of y dataframes.
    - output_path (str): Path to save the results.

    Returns:
    - None
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    score = list()
    names = list()

    # For loop over folds
    for i, (train_index, test_index) in enumerate(k_fold.split(Xy_full_paths)):
        X_train, y_train = get_data(X_clustered, y_clustered, train_index)

        # Fitting part
        # Scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # PCA
        reducer = PCA(n_components=11)
        reducer.fit(X_train)
        X_train = reducer.transform(X_train)

        # LinearSVC - fit
        linear_svc = LinearSVC(dual=False, C=8.0, multi_class='crammer_singer') # penalty='l1', loss='hinge'
        
        linear_svc.fit(X_train, y_train.values.ravel())

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):

            # Loading X_test & y_test
            X_test = pd.read_csv(f'{Xy_full_paths[index]}')

            y_test = X_test['target'].values

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = reducer.transform(X_test)

            # LinearSVC - Predicting
            y_hat = linear_svc.predict(X_test)
            
            # Metrics
            score.append(f1_score(y_test, y_hat, average='macro'))

            # Output_name
            # output_name = Xy_full_paths[index].split('\\')[-1][:-4] # this is for Windows
            output_name = Xy_full_paths[index].split('/')[-1][:-4]

            # Confusion matrix
            # cm_save(y_test, y_hat, output_path, output_name)

            # Reshape to img 500x500
            # y_hat = y_hat.reshape(500, 500)
            # y_test = y_test.reshape(500, 500)

            # # Saving images
            # plt.imsave(f'{output_path}{output_name}_prediction.png', y_hat)
            # plt.imsave(f'{output_path}{output_name}_target.png', y_test)

            names.append(output_name)

    data = {'names': names, 'f1': score}
    df = pd.DataFrame(data)
    df.to_csv(f'{output_path}results.csv', index=False)

    return None