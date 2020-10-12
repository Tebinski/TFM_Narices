from python.LSTMmodel import LSTMmodel2
from python.LoadUciData import load_data
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df_gas = load_data()

    ## GAS CLASIFICATION
    ## First, we will NOT use concentration data
    gas_X = df_gas.drop(columns=['Batch ID', 'GAS', 'CONCENTRATION']).to_numpy()
    gas_y = df_gas['GAS'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(gas_X, gas_y,
                                                        test_size=0.33,
                                                        random_state=42)
    seq = LSTMmodel2()
    model = seq.model_train(X_train, y_train)
    test_loss0, test_acc0 = seq.model_evaluate(model, X_test, y_test)
