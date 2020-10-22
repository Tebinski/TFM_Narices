import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from python.LoadUciData import load_data
from python.LoadSensorData import get_sensors_list, get_sensor
from python.StandardFigure import save_figure
from python.SeqModel import SeqModel, SeqModelSimple



def create_and_save_model(sensor, name):
    """ Creates Sequential model using as objetive only the GAS columnn."""
    df_sensor = get_sensor(sensor)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model(name)


def create_and_save_model_pack(sensor_pack, name):
    """ Creates Sequential model using as objetive only the GAS columnn."""
    df_sensor = get_sensors_list(sensor_pack)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model(name)


def load_and_test_model_pack(modname, sensorpack):
    """ Load Sequential model and calculates confusion matrix """
    print(modname)
    mod = SeqModel()
    mod.load_model(modname)
    df_sensor = get_sensors_list(sensorpack)
    mod1 = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod1.split_data()
    mod.model_evaluate(X_test, y_test)

    y_pred = mod.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf = confusion_matrix(y_pred, y_test)
    conf = conf.T / y_test.value_counts().sort_index().values
    return mod, conf


def load_and_test_model(modname, sensor):
    """ Load Sequential model and calculates confusion matrix """
    print(modname)
    mod = SeqModel()
    mod.load_model(modname)
    df_sensor = get_sensor(sensor)
    mod1 = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod1.split_data()
    mod.model_evaluate(X_test, y_test)

    y_pred = mod.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf = confusion_matrix(y_pred, y_test)
    conf = conf.T / y_test.value_counts().sort_index().values
    return mod, conf

def create_model_sensors():

    for s in range(0,16):
        create_and_save_model(s, f'Sensor{s}')


def create_models_packs(dict_sensors):
    # Create models one of each type
    sensor_pack_1 = dict_sensors['SensorP1']
    sensor_pack_2 = dict_sensors['SensorP2']
    sensor_pack_3 = dict_sensors['SensorP3']
    sensor_pack_4 = dict_sensors['SensorP4']

    create_and_save_model_pack(sensor_pack_1, 'Seq_SensorP1')
    create_and_save_model_pack(sensor_pack_2, 'Seq_SensorP2')
    create_and_save_model_pack(sensor_pack_3, 'Seq_SensorP3')
    create_and_save_model_pack(sensor_pack_4, 'Seq_SensorP4')

    # Create models same type (this is a redundant model)
    sensor_pack_A = dict_sensors['TypeA']
    sensor_pack_B = dict_sensors['TypeB']
    sensor_pack_C = dict_sensors['TypeC']
    sensor_pack_D = dict_sensors['TypeD']

    create_and_save_model_pack(sensor_pack_A, 'Seq_TypeA')
    create_and_save_model_pack(sensor_pack_B, 'Seq_TypeB')
    create_and_save_model_pack(sensor_pack_C, 'Seq_TypeC')
    create_and_save_model_pack(sensor_pack_D, 'Seq_TypeD')


if __name__ == '__main__':

    dict_sensors = {'SensorP1': np.array([0, 2, 4, 6]),
                    'SensorP2': np.array([1, 3, 5, 7]),
                    'SensorP3': np.array([8, 10, 12, 14]),
                    'SensorP4': np.array([9, 11, 13, 15]),
                    'TypeA': np.array([0, 1, 8, 9]),
                    'TypeB': np.array([2, 3, 10, 11]),
                    'TypeC': np.array([4, 5, 12, 13]),
                    'TypeD': np.array([6, 7, 14, 15])}

    # Train a neural net model for each sensor pack (type I, II, III and IV)
    create_models_packs(dict_sensors)

    #Train  neural net for each infividual sensor
    create_model_sensors()

    # load models
    modP1, confP1 = load_and_test_model_pack('Seq_SensorP1', dict_sensors['SensorP1'])
    modP2, confP2 = load_and_test_model_pack('Seq_SensorP2', dict_sensors['SensorP2'])
    modP3, confP3 = load_and_test_model_pack('Seq_SensorP3', dict_sensors['SensorP3'])
    modP4, confP4 = load_and_test_model_pack('Seq_SensorP4', dict_sensors['SensorP4'])

    modA, confA = load_and_test_model_pack('Seq_TypeA', dict_sensors['TypeA'])
    modB, confB = load_and_test_model_pack('Seq_TypeB', dict_sensors['TypeB'])
    modC, confC = load_and_test_model_pack('Seq_TypeC', dict_sensors['TypeC'])
    modD, confD = load_and_test_model_pack('Seq_TypeD', dict_sensors['TypeD'])

    # Los 4 modelos tienen la misma accuracy, pero, ¿y la matrix de confusion?
    fig = plt.figure(); ax = sns.heatmap(confP1, annot=True); plt.title('Seq_SensorP1'); plt.show(); save_figure(fig,'Conf_Seq_SensorP1' )
    fig = plt.figure(); ax = sns.heatmap(confP2, annot=True); plt.title('Seq_SensorP2'); plt.show(); save_figure(fig,'Conf_Seq_SensorP2' )
    fig = plt.figure(); ax = sns.heatmap(confP3, annot=True); plt.title('Seq_SensorP3'); plt.show(); save_figure(fig,'Conf_Seq_SensorP3' )
    fig = plt.figure(); ax = sns.heatmap(confP4, annot=True); plt.title('Seq_SensorP4'); plt.show(); save_figure(fig,'Conf_Seq_SensorP4' )

    # let's see the result for each sensor type
    fig = plt.figure(); ax = sns.heatmap(confA, annot=True); plt.title('Seq_TypeA'); plt.show(); save_figure(fig,'Conf_Seq_TypeA')
    fig = plt.figure(); ax = sns.heatmap(confB, annot=True); plt.title('Seq_TypeB'); plt.show(); save_figure(fig,'Conf_Seq_TypeB')
    fig = plt.figure(); ax = sns.heatmap(confC, annot=True); plt.title('Seq_TypeC'); plt.show(); save_figure(fig,'Conf_Seq_TypeC')
    fig = plt.figure(); ax = sns.heatmap(confD, annot=True); plt.title('Seq_TypeD'); plt.show(); save_figure(fig,'Conf_Seq_TypeD')

    # Nos quedamos con el numero de aciertos (la diagonal)
    diagA = np.diag(confA)
    diagB = np.diag(confB)
    diagC = np.diag(confC)
    diagD = np.diag(confD)

    matches = np.stack([diagA, diagB, diagC, diagD])
    df_match = pd.DataFrame(data=matches, index=['TypeA',
                                                 'TypeB',
                                                 'TypeC',
                                                 'TypeD'])
    fig = plt.figure()
    ax = sns.heatmap(df_match, annot=True, fmt=".02f", vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.show()


    # Voy a separar los sensores
    diag = []
    sens_ix = np.concatenate([dict_sensors['TypeA'],
                              dict_sensors['TypeB'],
                              dict_sensors['TypeC'],
                              dict_sensors['TypeD']])
    for s in range(0,16):
        mod, conf = load_and_test_model(f'Sensor{s}', s)
        d = np.diag(conf)
        diag.append(d)
    df_match = pd.DataFrame(data=diag, index=[f'Sensor{s}' for s in sens_ix])
    fig = plt.figure(figsize=(8,8))
    ax = sns.heatmap(df_match, annot=True, fmt=".02f", vmin=0, vmax=1, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.show()

    # La precision para cada gas, para cada tipo de sensor, no tiene sentido.
    # Pueden ser muchas cosas.Redactar.
    #parece que solo los s
    # sensores 9,2,4,11 son capaces de catalogar correctamente el gas2!
    # Parece que solo algunos deentre todos los sensores son capaces de detectar un gas determinado.
    # ramdomforest all rescate?
