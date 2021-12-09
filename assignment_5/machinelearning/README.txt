Joseph Edradan
12/8/2021
920783419

All I did was problem 1 because I don't have time to do the rest. The implementation should be simple to a
Keras sequential model similar to the below, though I haven't tested it:

    # New model
    model = tf.keras.models.Sequential()

    # Add 1 layer with 1 node
    model.add(layers.Dense(1, input_dim=INPUT_DIMENSION, activation="relu"))

    # Classification model using Stochastic Gradient Descent as the optimizer
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # Train the model
    model.fit(
        x=x,
        y=y,
        batch_size=BATCH_SIZE,  # Should be 1
        epochs=EPOCHS,  # Should be 10000
        # verbose="auto",
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss')],
    )

I spend 2 hours on the assigment because I used Dataset.iterate_forever(BATCH_SIZE) instead.