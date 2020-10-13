"""
Comprobarmos el rendimiento de las redes neuronales para realizar la tarea de clasificacion utilizando la informacion
de la concentracion dentro del dataset.
"""


from sklearn.model_selection import train_test_split

from python.LoadUciData import load_data

if __name__ == '__main__':

    # Load data
    df_gas = load_data()

    df_train_reg = df_gas[df_gas['GAS'] == 1]

    gas_X = df_train_reg.drop(columns=['Batch ID', 'GAS', 'CONCENTRATION']).to_numpy()
    gas_y = df_train_reg['CONCENTRATION'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(gas_X, gas_y,
                                                        test_size=0.33,
                                                        random_state=42)
