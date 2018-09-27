
def auto_trader(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyTrader3 import PyTrader


    sns.set()

    print('start importing:')
    # import benchmark data and back-up data

    # Your alpha goes here
    yxCombo = np.memmap("/home/hanhy/data/evaluation/tdcombo32.dat", dtype='float32', mode='r', shape=(40320,1829))

    # Benchmark Data goes here
    benchmark = np.memmap("/home/hanhy/data/evaluation/xs130combo.dat", dtype='float32', mode='r', shape=(40320,1829))


    # Evaluation datas go here
    pClose = np.memmap("/home/hanhy/data/evaluation/stk_clsadj", dtype='float32', mode='r', shape=(40320,1829))
    bidAdj = np.memmap("/home/hanhy/data/evaluation/stk_bidadj", dtype='float32', mode='r', shape=(40320,1829))
    askAdj = np.memmap("/home/hanhy/data/evaluation/stk_askadj", dtype='float32', mode='r', shape=(40320,1829))

    # needed data
    if type(data) == str:
        data = np.memmap(data, dtype = 'float32', mode = 'r', shape=(40320, 1829))
    else:
        data = data


    # start trading:
    print('start calculating:')

    alpha = yxCombo


    # config = {"nBars": 40320, "iStkMax": 1082}
    config = {}

    # Benchmark alpha combo
    myJudger = PyTrader(benchmark, pClose, bidAdj, askAdj, config)
    myTrader = PyTrader(alpha, pClose, bidAdj, askAdj, config)
    dataTrader = PyTrader(data, pClose, bidAdj, askAdj, config)

    BenchStats = np.array(myJudger.trade(), copy=False)
    dailyStats = np.array(myTrader.trade(), copy=False)
    dataStats = np.array(dataTrader.trade(), copy = False)

    cumBen = np.nancumsum(BenchStats, axis=0)
    cumPnl = np.nancumsum(dailyStats, axis=0)
    cumData = np.nancumsum(dataStats, axis=0)

    print('Plotting:')

    plt.figure(figsize = (12, 8))
    plt.title('Mid-price Comparison Between Target, Benchmark Combo and Ideal Combo.')
    plt.plot(cumPnl[:, 1] /1e7, label = 'IdealCombo')
    plt.plot(cumData[:, 1]/1e7, label = 'Our Model')
    plt.plot(cumBen[:, 1] /1e7, ':', label = 'Benchmark')
    plt.legend()
    plt.show()



    return None
