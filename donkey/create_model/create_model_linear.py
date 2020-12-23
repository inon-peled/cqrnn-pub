import tensorflow as tf

def create_model(
        dense_kernel_initializer,
        loss_function_maker, 
        optimizer, 
        **loss_function_kwargs):
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(
            units=1,
            use_bias=True,
            activation='linear',
            kernel_initializer=dense_kernel_initializer
        )]
    )
    model.compile(loss=loss_function_maker(**loss_function_kwargs), 
                  optimizer=optimizer)
    return model
