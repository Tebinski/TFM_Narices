
def lstm_data_transform(x_data, y_data, num_steps=5):
    """ Changes data to the format for LSTM training
        for sliding window approach
        https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00
    """
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

class LSTMmodel1:

    def __init__(self):
        pass

    def _structure(self):
        model = Sequential()
        model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(LSTM(hidden_size, return_sequences=True))
        if use_dropout:
            model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(vocabulary)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])

        model.fit_generator(train_data, len(train_data) // (batch_size * num_steps), num_epochs,
                            validation_data=valid_data,
                            validation_steps=len(valid_data) // (batch_size * num_steps))
