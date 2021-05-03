import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

AVERAGE_DAYS_IN_YEAR = 365.25
AVERAGE_DAYS_IN_MONTH = 30.4375
NON_POSITIVE_NO_GOOD = ['price', 'sqft_living', 'sqft_lot','floors', 'yr_built', 'sqft_living15','sqft_lot15']
NEGATIVE_NO_GOOD = ['bedrooms', 'bathrooms', 'sqft_above' , 'sqft_basement', 'yr_renovated']
ONE_HOT_ENC = ['zipcode']


def fit_linear_regression(x, y):
    x_to_fit = x
    if (len(x_to_fit.shape) == 1):
        x_to_fit = x_to_fit.reshape((x_to_fit.shape[0], 1))
    _, s, _ = np.linalg.svd(x_to_fit)
    x_pinv = np.linalg.pinv(x_to_fit)
    w = np.dot(x_pinv, y)
    return w, s


def predict(x, w):
    return (np.dot(x, w))


def mse(y_label, y_predict):
    diff = y_label - y_predict
    return (1 / len(diff)) * np.dot(diff, diff)


def load_data(kc_house_data_path):
    df = pd.read_csv(kc_house_data_path)
    df = df.dropna()

    new_date_coulomn = []
    for date in df['date']:  ## get some number for the date
        new_date_coulomn.append(date_str_to_int(date))
    df['date'] = new_date_coulomn
    for element in NON_POSITIVE_NO_GOOD:
        df = df[df[element] > 0]
    for element in NEGATIVE_NO_GOOD:
        df = df[df[element] >= 0]
    for element in ONE_HOT_ENC:
        oh_df = pd.get_dummies(df[element])
        df = result = pd.concat([df, oh_df], axis=1)
    df = df.drop(ONE_HOT_ENC, axis=1)
    y = df['price']
    df = df.drop(['price', 'id'], axis=1)
    df['ones'] = 1
    return df, y  ## get rid of ID as it is not relevant


def date_str_to_int(str_date):
    try:
        year = int(str_date[0:4])
        month = int(str_date[4:6])
        day = int(str_date[6:8])
        return ((year * AVERAGE_DAYS_IN_YEAR) + (month * AVERAGE_DAYS_IN_MONTH) + day)
    except:
        return np.nan


def plot_singular_values(s):
    plt.figure(figsize=(15, 10))
    plt.title('Singular Value in Order of Size')
    plt.xlabel('Index')
    plt.ylabel('Singular Values')
    plt.scatter(np.arange(len(s)), s)
    plt.show()


def divide_data(x, y, train_prop=0.75):
    train_index = np.random.choice(a=[False, True], size=y.shape, p=[1 - train_prop, train_prop])
    train_x, train_y = (x[train_index]), y[train_index]
    test_x, test_y = x[np.logical_not(train_index)], y[np.logical_not(train_index)]

    return train_x, train_y, test_x, test_y


def train_results(x, y):
    train_x, train_y, test_x, test_y = divide_data(x, y)
    mse_for_percent = []
    for i in range(1, 100):
        print(i)
        max_index = int((i * len(train_y)) / 100)
        w, s = fit_linear_regression((train_x[:max_index]), train_y[:max_index])
        predict_y = predict(test_x, w)
        mse_for_percent.append(mse(test_y, predict_y))
    return mse_for_percent


def feature_evaluation(x, y):
    y_std = np.std(y)
    for i, j in x.iteritems():
        feature = x[i]
        pearson_cor = np.cov(feature, y) / (np.std(feature) * y_std)
        plt.figure(figsize=(10, 5))
        plt.title("Scatter of The price of  House vs. {}\nPearson Correlation =\n{}".format(i, pearson_cor))
        plt.scatter(feature, y)
        plt.xlabel(i)
        plt.ylabel('Price')
        plt.show()

if __name__ == '__main__':
    features,prices = load_data(r"kc_house_data.csv")
    ## Question 15
    house_w,house_s = fit_linear_regression(features , prices)
    plot_singular_values(house_s)
    ## Question 16
    mses = train_results(features,prices)
    plt.figure(figsize=(20, 10))
    plt.title("Scatter of The MSE vs. The percentage of Train batch used to fit")
    plt.scatter(np.arange(len(mses)), np.log10(mses))
    plt.xlabel('Percenage')
    plt.ylabel('MSE value')
    plt.show()
    ## Question 17
    feature_evaluation(features , prices)
