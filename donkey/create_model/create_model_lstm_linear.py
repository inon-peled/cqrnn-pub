import tensorflow as tf

def create_model(
        dense_kernel_initializer,
        loss_function_maker, 
        optimizer, 
        **loss_function_kwargs):
    model = tf.keras.models.Sequential(
        [
#            tf.keras.layers.Dropout(0.2), # <-------------- THIS GREATLY REDUCES PERFORMANCE WITH LSTM!!!!!!!!!!!
#          tf.keras.layers.Dense(
#             kernel_regularizer=tf.keras.regularizers.L2(0.01),
#             units=1,
#             use_bias=True,
#             activation=activation,
#             kernel_initializer=dense_kernel_initializer
#         ),
         tf.keras.layers.LSTM(
            kernel_initializer=dense_kernel_initializer,
             activation='linear',  # <------------- THIS MADE ALL THE DIFFERENCE FOR BOTH LOSS=MSE AND LOSS=YS!!!!!!!!!!!
#             recurrent_activation=None,
            units=1,
        ),
#        tf.keras.layers.Dense(
#            kernel_regularizer=tf.keras.regularizers.L2(0.01),
#            units=50,
#            use_bias=True,
#            activation='relu',
#            kernel_initializer=dense_kernel_initializer
#        ),
#        tf.keras.layers.Dense(
#            kernel_regularizer=tf.keras.regularizers.L2(0.01),
#            units=50,
#            use_bias=True,
#            activation='relu',
#            kernel_initializer=dense_kernel_initializer
#        ),
         tf.keras.layers.Dense(
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
            units=1,
            use_bias=True,
            activation='linear',
            kernel_initializer=dense_kernel_initializer
        )
        ]
    )
    model.compile(loss=loss_function_maker(**loss_function_kwargs), # 'mse'
                  optimizer=optimizer)
    return model
