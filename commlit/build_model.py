
# -------------------------- # -------------------------- #
# Modele building module --- # -------------------------- #
# -------------------------- # -------------------------- #

import tensorflow as tf
import keras

def build_model(cnn, att, agg, Q, extra, out):
    """
    """
    
    # all inputs and layers
    all_layers = []
    all_inputs = []
    
    # CNN layer
    if cnn is not None:
        in_cnn = keras.layers.Input(shape=cnn["shape"][1:])
        all_inputs.append(in_cnn)
        cnn_x = keras.layers.Conv2D(
            filters=cnn["filters"], 
            kernel_size=(1, cnn["shape"][2]),
            activation=cnn["acti"],
            kernel_regularizer=keras.regularizers.l2(cnn["l2_reg"])
        )(in_cnn)
        
        # "Attention" layer
        if att is not None:
            in_att = keras.layers.Input(shape=att["shape"][1:])
            all_inputs.append(in_att)
            att_x = keras.layers.Conv2D(
                filters=att["filters"], 
                kernel_size=(1, att["shape"][2]),
                activation=att["acti"],
                kernel_regularizer=keras.regularizers.l2(att["l2_reg"])
            )(in_att)
            cnn_x = keras.layers.Multiply()([cnn_x, att_x])
        
        cnn_x = tf.math.reduce_mean(cnn_x, axis=1)
        cnn_x = keras.layers.Flatten()(cnn_x)
        all_layers.append(cnn_x)
      
    # agg feature layer
    if agg is not None:
        in_agg = keras.layers.Input(shape=agg["shape"])
        all_inputs.append(in_agg)
        agg_x = keras.layers.Dropout(agg["drop"])(in_agg)
        if agg["dense"]:
            agg_x = keras.layers.Dense(
                agg["n"],
                activation=agg["acti"],
                kernel_regularizer=keras.regularizers.l2(agg["l2_reg"])
            )(agg_x)
        all_layers.append(agg_x)
    
    # add Q layers
    for q in range(Q["n_q"]):
        in_q = keras.layers.Input(shape=Q["shape"])
        all_inputs.append(in_q)
        q_x = keras.layers.Dropout(Q["drop"])(in_q)
        if Q["dense"]:
            q_x = keras.layers.Dense(
                Q["n"],
                activation=Q["acti"],
                kernel_regularizer=keras.regularizers.l2(Q["l2_reg"])
            )(q_x)
        all_layers.append(q_x)
        
    # concat layers and add extra dense layer if desired
    all_x = keras.layers.Concatenate()(all_layers)
    if extra["use"]:
        all_x = keras.layers.Dense(
            extra["n"],
            activation=extra["acti"],
            kernel_regularizer=keras.regularizers.l2(extra["l2_reg"])
        )(all_x)
    
    # output layer
    out_y = keras.layers.Dense(
        1, 
        activation=out["acti"],
        kernel_regularizer=keras.regularizers.l2(out["l2_reg"])
    )(all_x)
    
    return keras.Model(inputs=all_inputs, outputs=out_y)
