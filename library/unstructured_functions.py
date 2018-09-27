def mean(data):
    length = len(data)
    Sum = 0
    for i in range(length):
        Sum += data[i]

    result = Sum/(length + 10**(-8))
    return result



def Sxx(X, y):
    X_mean = mean(X)
    X_ = []

    for i in range(len(X)):
        X_.append((X[i]-X_mean)**2)

    return sum(X_)

def Sxy(X, y):
    X_mean = mean(X)
    y_mean = mean(y)
    Xy_ = []

    for i in range(len(X)):
        Xy_.append((X[i] - X_mean)*(y[i] - y_mean))

    return sum(Xy_)

def beta_1(X, y):
    return Sxy(X, y)/Sxx(X, y)

def beta_0(X, y):
    X_mean = mean(X)
    y_mean = mean(y)
    return y_mean - beta_1(X, y) * X_mean


def linear_regression(X, y):
    return beta_1(X, y), beta_0(X, y)

def plotter(X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    beta_0, beta_1 = linear_regression(X, y)
    X_ = np.linspace(min(X), max(X), 200)
    y_ = beta_1 * X_ + beta_0


    plt.figure(figsize = (10, 6))
    plt.plot(X_, y_, color = 'red')
    plt.scatter(X, y)
    plt.annotate(r'a = %f'%(beta_1), xy = (max(X_)-1, min(y_)+0.5))
    plt.annotate(r'b = %f'%(beta_0), xy = (max(X_)-1, min(y_)))
    plt.title('Sample Result: linear regression of time series dataset.')
    plt.show()

    return beta_1, beta_0





def rolling_splitter(x, y, cut = 8, delete = None):
    '''
    40320个而不是40000个的splitter
    '''
    # print(x.shape)
    x = np.squeeze(x)
    if delete != None:
        x = np.delete(x, delete, axis = 1)
    # x = x.T
    # define hyper-parameter and return paras:
    x_ = []
    y_ = []

    length = 40320 * 1082 / cut
    x = np.asarray(x)
    # x = np.delete(x, [3, 8, 16], axis = 0)
    y = np.asarray(y)
    y = y.reshape(-1,1)

    print(np.shape(x))
    print(y.shape)

    # cutting first
    for i in range(cut):
        if i != cut-1:
            x_.append(x[int(i*length):int(i*length + length),:])
            y_.append(y[int(i*length):int(i*length + length),:])
        elif i == cut-1:
            x_.append(x[int(i*length):int(i*length + length - 5394330 + 5361870),:])
            y_.append(y[int(i*length):int(i*length + length),:])

    # deleting data if it have nan value inside of it.
    for i in range(cut):
        print('in %d th iteration:'%i)
    # deleting rows form train dataset
        train_index = []
        train_X = x_[i]
        train_y = y_[i]
        print(train_y.shape)
        for j in range(train_y.shape[0]):

            if np.isnan(train_y[j]):
                train_index.append(j)
        train_X = np.delete(train_X, train_index, axis = 0)
        train_y = np.delete(train_y, train_index, axis = 0)
        x_[i] = train_X
        y_[i] = train_y
        print(train_y.shape)
        print(train_X.shape)
        assert np.shape(train_X)[0] == np.shape(train_y)[0]

    print("The shape of total X is,",np.shape(x_))
    print("The shape of total y is,",np.shape(y_))


    return x_, y_, x
