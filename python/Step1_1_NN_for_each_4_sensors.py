import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from python.LoadUciData import load_data
from python.LoadSensorData import get_sensors
from python.StandardFigure import save_figure
from python.SeqModel import SeqModel, SeqModelSimple



def create_4_models(sensor_packs):

    sensor_pack_1, sensor_pack_2, sensor_pack_3, sensor_pack_4 = sensor_packs

    # Sensor pack 1
    df_sensor = get_sensors(sensor_pack_1)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model('Seq_SensorP1')

    # Sensor pack 2
    df_sensor = get_sensors(sensor_pack_2)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model('Seq_SensorP2')

    # Sensor pack 3
    df_sensor = get_sensors(sensor_pack_3)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model('Seq_SensorP3')

    # Sensor pack 4
    df_sensor = get_sensors(sensor_pack_4)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod.train_and_save_model('Seq_SensorP4')


if __name__ == '__main__':

    #Train a neural net model for each sensor pack (type I, II, III and IV)
    sensor_pack_1 = np.array([0, 2, 4, 6])
    sensor_pack_2 = sensor_pack_1 + 1
    sensor_pack_3 = np.array([8, 10, 12, 14])
    sensor_pack_4 = sensor_pack_3 + 1

    sensor_packs = [sensor_pack_1,
                    sensor_pack_2,
                    sensor_pack_3,
                    sensor_pack_4]

    #create_4_models(sensor_packs)

    #load models
    mod1 = SeqModel()
    mod1.load_model('Seq_SensorP1')
    df_sensor = get_sensors(sensor_pack_1)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod1.model_evaluate(X_test, y_test)

    y_pred = mod1.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf1 = confusion_matrix(y_pred, y_test)

    mod2 = SeqModel()
    mod2.load_model('Seq_SensorP2')
    df_sensor = get_sensors(sensor_pack_2)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod2.model_evaluate(X_test, y_test)

    y_pred = mod1.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf2 = confusion_matrix(y_test, y_pred)

    mod3 = SeqModel()
    mod3.load_model('Seq_SensorP3')
    df_sensor = get_sensors(sensor_pack_3)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod3.model_evaluate(X_test, y_test)

    y_pred = mod1.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf3 = confusion_matrix(y_test, y_pred)

    mod4 = SeqModel()
    mod4.load_model('Seq_SensorP4')
    df_sensor = get_sensors(sensor_pack_4)
    mod = SeqModelSimple(df_sensor)
    X_train, X_test, y_train, y_test = mod.split_data()
    mod4.model_evaluate(X_test, y_test)

    y_pred = mod1.get_model().predict(X_test)
    y_pred = y_pred.argmax(axis=1) + 1
    y_test = y_test.idxmax(axis=1)
    conf4 = confusion_matrix(y_test, y_pred)

    # Los 4 modelos tienen la misma accuracy, pero, ¿y la matrix de confusion?
    sns.heatmap(conf1); plt.show()
    sns.heatmap(conf2); plt.show()
    sns.heatmap(conf3); plt.show()
    sns.heatmap(conf4); plt.show()

    # Si hubiesemos usado todos los sensores 1, segun el paper, hubieramos detectado mejor
    # los gases que dice el paper

