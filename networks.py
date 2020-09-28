__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"

"""
Code for creating some neural network models. Don't judge them, please. They are just born this way.
"""

import keras
import keras.backend as K
import numpy as np
from keras.layers import (BatchNormalization, Dropout, Dense, Input, Activation, LSTM, Add, MaxPool1D,
                          Bidirectional, Concatenate, Embedding, TimeDistributed, AveragePooling1D)


def make_resnet(input_layer, regularizer_weight, res_size, layers=(2, 2), dropout=0., bn=True):
    prev_layer = input_layer
    prev_block = prev_layer
    blocks = layers[0]
    res_layers = layers[1]

    shape = int(np.shape(input_layer)[1])

    for i in range(1, blocks + 1):
        for j in range(1, res_layers):
            if bn:
                prev_layer = BatchNormalization(name='resent_BN_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dense(units=res_size,
                               activation=None,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                               bias_regularizer=keras.regularizers.l2(regularizer_weight),
                               name='resnet_dense_' + str(i) + '_' + str(j)
                               )(prev_layer)
        if bn:
            prev_layer = BatchNormalization(name='BN_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dense(units=shape,
                           activation=None,
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                           bias_regularizer=keras.regularizers.l2(regularizer_weight),
                           name='resnet_dense_' + str(i) + '_' + str(res_layers)
                           )(prev_layer)

        prev_layer = Add(name='resnet_sum' + str(i))([prev_block, prev_layer])
        prev_block = prev_layer

    return prev_block


# TODO: not clear
# TODO: he_normal as initialization?
# TODO: outputs as dictionary
def make_embedder_layers(regularizer_weight, shape, layers_size, layers=2, dropout=0.1,
                         temporal_bn=False):
    bn_list_prop = []
    layers_list = []
    dropout_list = []
    activation_list = []
    bn_list_text = []

    if layers > 0:
        layer = Dense(units=shape,
                      activation=None,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                      bias_regularizer=keras.regularizers.l2(regularizer_weight),
                      name='dense_' + str(layers))
        layers_list.append(layer)
        if temporal_bn:
            bn_list_prop.append(BatchNormalization(axis=-2, name="TBN_claim_" + str(layers)))
            bn_list_text.append(BatchNormalization(axis=-2, name="TBN_topic_" + str(layers)))
        else:
            bn_list_prop.append(BatchNormalization(name="BN_" + str(layers)))
        dropout_list.append(Dropout(dropout, name='Dropout_' + str(layers)))
        activation_list.append(Activation('relu', name='ReLU_' + str(layers)))

    for i in range(1, layers):
        layer = Dense(units=layers_size,
                      activation=None,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                      bias_regularizer=keras.regularizers.l2(regularizer_weight),
                      name='dense_' + str(i))
        if temporal_bn:
            bn_list_prop.append(BatchNormalization(axis=-2, name="TBN_prop_" + str(i)))
            bn_list_text.append(BatchNormalization(axis=-2, name="TBN_text_" + str(i)))
        else:
            bn_list_prop.append(BatchNormalization(name="BN_" + str(i)))
        layers_list.append(layer)
        dropout_list.append(Dropout(dropout, name='Dropout_' + str(i)))
        activation_list.append(Activation('relu', name='ReLU_' + str(i)))

    add_layer = Add(name='sum')

    return layers_list, bn_list_prop, dropout_list, activation_list, add_layer, bn_list_text


def make_embedder_with_layers(input_layer, layer_name, layers, dropout=0., bn=True, temporal_bn=False, residual=True):
    prev_layer = input_layer

    for i in range(1, len(layers)):

        if bn:
            if temporal_bn:
                prev_layer = BatchNormalization(axis=-2, name=layer_name + '_TBN_' + str(i))(prev_layer)
            else:
                prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(i))(prev_layer)

        prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(i))(prev_layer)

        prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(i))(prev_layer)

        prev_layer = TimeDistributed(layers[i],
                                     name=layer_name + '_TD_' + str(i))(prev_layer)
    if bn:
        if temporal_bn:
            prev_layer = BatchNormalization(axis=-2, name=layer_name + '_TBN_' + str(len(layers)))(prev_layer)
        else:
            prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(len(layers)))(prev_layer)

    prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(len(layers)))(prev_layer)

    prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(len(layers)))(prev_layer)

    prev_layer = TimeDistributed(layers[0],
                                 name=layer_name + '_TD_' + str(len(layers)))(prev_layer)

    if residual:
        prev_layer = Add(name=layer_name + '_sum')([input_layer, prev_layer])

    return prev_layer


def make_embedder_with_all_layers(input_layer, layer_name, layers, bn=True, temporal_bn=False, residual=True,
                                  dropout=0.):
    prev_layer = input_layer

    bn_layers = layers[1]
    dropout_layers = layers[2]
    activation_layers = layers[3]
    add_layer = layers[4]
    bn_t_layers = layers[5]
    layers = layers[0]

    for i in range(1, len(layers)):

        if bn:
            if temporal_bn:
                if layer_name == 'text':
                    prev_layer = bn_t_layers[i](prev_layer)
                else:
                    prev_layer = bn_layers[i](prev_layer)
            else:
                prev_layer = bn_layers[i](prev_layer)

        prev_layer = dropout_layers[i](prev_layer)

        prev_layer = activation_layers[i](prev_layer)

        prev_layer = TimeDistributed(layers[i],
                                     name=layer_name + '_TD_' + str(i))(prev_layer)
    if bn:
        if temporal_bn:
            if layer_name == 'text':
                prev_layer = bn_t_layers[0](prev_layer)
            else:
                prev_layer = bn_layers[0](prev_layer)
        else:
            prev_layer = bn_layers[0](prev_layer)

    prev_layer = dropout_layers[0](prev_layer)

    prev_layer = activation_layers[0](prev_layer)

    prev_layer = TimeDistributed(layers[0],
                                 name=layer_name + '_TD_' + str(len(layers)))(prev_layer)

    if residual:
        prev_layer = add_layer([input_layer, prev_layer])

    return prev_layer


def build_net_7_nc(embedding_weights,
                   embedding_dimension,
                   text_max_length,
                   distance=0,
                   l2_regularization=0.0001,
                   dropout_embedder=0.1,
                   dropout_resnet=0.1,
                   dropout_final=0.,
                   reduced_embedding_dimension=50,
                   embedder_layers=4,
                   resnet_layers=(1, 2),
                   res_size=5,
                   final_size=20,
                   bn_embed=True,
                   bn_res=True,
                   bn_final=True,
                   share_rnn=True,
                   pooling=10,
                   pooling_type='avg',
                   share_embedder=False,
                   temporal_bn=False):
    """
    Creates a neural network that classifies two argumentative components and their relation.
    It takes as input two components (propositions) and their distance.
    It ouputs the classes of the two components, whether a relation from source to target does exists (link class),
    and the class of that relation.
    The two components must be represented as a bi-dimensional tensors of features, or as mono-dimensional tensors if a
    matrix to convert each mono-dimensional feature into a bidimensional one is provided (bow input of the function).
    For example if the bow matrix contains the pre-trained embeddings of each word, the input to the network can be the
    integer sequence that represent the words.
    The distance must be encoded with twice the number of features as it is specified in the parameters.
    The oputputs will be, in order, the link class, the relation class, the source class, and the target class.

    :param embedding_weights: If it is different from None, it is the matrix with the pre-trained embeddings used by the Embedding
                layer of keras, the input is supposed to be a list of integer which represent the words.
                If it is None, the input is supposed to already contain pre-trained embeddings.
    :param embedding_dimension: input word embeddings dimension
    :param text_max_length: The temporal length of the proposition input
    :param distance: The maximum distance that is taken into account, the input is expected to be twice that size.
    :param l2_regularization: Regularization weight
    :param dropout_embedder: Dropout used in the embedder
    :param dropout_resnet: Dropout used in the residual network
    :param dropout_final: Dropout used in the final classifiers
    :param reduced_embedding_dimension: Size of the spatial reduced embeddings
    :param embedder_layers: Number of layers in the initial embedder (int)
    :param resnet_layers: Number of layers in the final residual network. Tuple where the first value indicates the
                          number of blocks and the second the number of layers per block
    :param res_size: Number of neurons in the residual blocks
    :param final_size: Number of neurons of the final layer
    :param bn_embed: Whether the batch normalization should be used in the embedding block
    :param bn_res: Whether the batch normalization should be used in the residual blocks
    :param bn_final: Whether the batch normalization should be used in the final layer
    :param share_rnn: Whether the same LSTM should be used both for processing the target and the source
    :param pooling:
    :param pooling_type: 'avg' or 'max' pooling
    :param share_embedder: Whether the deep embedder layers should be shared between source and target
    :param temporal_bn: Whether temporal batch-norm is applied
    :return:
    """

    # Embeddings
    if embedding_weights is not None:
        claim_il = Input(shape=(text_max_length,), name="claim_input_L")
        topic_il = Input(shape=(text_max_length,), name="topic_input_L")

        prev_claim_l = Embedding(embedding_weights.shape[0],
                                 embedding_weights.shape[1],
                                 weights=[embedding_weights],
                                 input_length=text_max_length,
                                 trainable=True,
                                 name="claim_embeddings")(claim_il)

        prev_topic_l = Embedding(embedding_weights.shape[0],
                                 embedding_weights.shape[1],
                                 weights=[embedding_weights],
                                 input_length=text_max_length,
                                 trainable=True,
                                 name="topic_embeddings")(topic_il)
    else:
        claim_il = Input(shape=(text_max_length, embedding_dimension), name="claim_input_L")
        topic_il = Input(shape=(text_max_length, embedding_dimension), name="topic_input_L")
        prev_claim_l = claim_il
        prev_topic_l = topic_il

    # Distance (if any)
    if distance > 0:
        dist_il = Input(shape=(int(distance * 2),), name="dist_input_L")

    # Build deep embedder
    shape = int(np.shape(prev_claim_l)[2])
    layers = make_embedder_layers(l2_regularization, shape=shape, layers=embedder_layers,
                                  layers_size=reduced_embedding_dimension, temporal_bn=temporal_bn)
    if share_embedder:
        make_embedder = make_embedder_with_all_layers
    else:
        make_embedder = make_embedder_with_layers
        layers = layers[0]

    if embedder_layers > 0:
        prev_claim_l = make_embedder(prev_claim_l, 'claim', dropout=dropout_embedder,
                                     layers=layers, bn=bn_embed, temporal_bn=temporal_bn)
        prev_topic_l = make_embedder(prev_topic_l, 'topic', dropout=dropout_embedder,
                                     layers=layers, bn=bn_embed, temporal_bn=temporal_bn)

    if share_embedder:
        if bn_embed:
            if temporal_bn:
                bn_layer = BatchNormalization(name="TBN_DENSE_input", axis=-2)
            else:
                bn_layer = BatchNormalization(name="BN_DENSE_generic")

            prev_claim_l = bn_layer(prev_claim_l)
            prev_topic_l = bn_layer(prev_topic_l)

        drop_layer = Dropout(dropout_embedder)

        prev_claim_l = drop_layer(prev_claim_l)
        prev_topic_l = drop_layer(prev_topic_l)

    else:
        if bn_embed:
            if temporal_bn:
                prev_claim_l = BatchNormalization(axis=-2)(prev_claim_l)
                prev_topic_l = BatchNormalization(axis=-2)(prev_topic_l)
            else:
                prev_claim_l = BatchNormalization()(prev_claim_l)
                prev_topic_l = BatchNormalization()(prev_topic_l)

        prev_claim_l = Dropout(dropout_embedder)(prev_claim_l)
        prev_topic_l = Dropout(dropout_embedder)(prev_topic_l)

    relu_embedder = Dense(units=reduced_embedding_dimension,
                          activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=keras.regularizers.l2(l2_regularization),
                          bias_regularizer=keras.regularizers.l2(l2_regularization),
                          name='relu_embedder')

    TD_prop = TimeDistributed(relu_embedder, name='TD_input_embedder')
    prev_claim_l = TD_prop(prev_claim_l)
    prev_topic_l = TD_prop(prev_topic_l)

    if pooling > 0:
        if pooling_type == 'max':
            pooling_class = MaxPool1D
        else:
            pooling_class = AveragePooling1D
        prop_pooling = pooling_class(pool_size=pooling, name='input_pooling')
        prev_claim_l = prop_pooling(prev_claim_l)
        prev_topic_l = prop_pooling(prev_topic_l)

    if share_rnn:
        if bn_embed:

            if temporal_bn:
                bn_layer = BatchNormalization(name="TBN_LSTM_input", axis=-2)
                prev_claim_l = bn_layer(prev_claim_l)
                prev_topic_l = bn_layer(prev_topic_l)
            else:
                bn_layer = BatchNormalization(name="BN_LSTM_input")
                prev_claim_l = bn_layer(prev_claim_l)
                prev_topic_l = bn_layer(prev_topic_l)

        embed2 = Bidirectional(LSTM(units=reduced_embedding_dimension,
                                    dropout=dropout_embedder,
                                    recurrent_dropout=dropout_embedder,
                                    kernel_regularizer=keras.regularizers.l2(l2_regularization),
                                    recurrent_regularizer=keras.regularizers.l2(l2_regularization),
                                    bias_regularizer=keras.regularizers.l2(l2_regularization),
                                    return_sequences=False,
                                    unroll=False,  # not possible to unroll if the time shape is not specified
                                    name='input_LSTM',
                                    ),
                               merge_mode='mul',
                               name='input_biLSTM'
                               )

        source_embed2 = embed2(prev_claim_l)
        target_embed2 = embed2(prev_topic_l)

    else:
        if bn_embed:
            if temporal_bn:
                prev_claim_l = BatchNormalization(name="TBN_LSTM_claim", axis=-2)(prev_claim_l)
                prev_topic_l = BatchNormalization(name="TBN_LSTM_topic", axis=-2)(prev_topic_l)
            else:
                prev_claim_l = BatchNormalization(name="BN_LSTM_claim")(prev_claim_l)
                prev_topic_l = BatchNormalization(name="BN_LSTM_topic")(prev_topic_l)

        source_embed2 = Bidirectional(LSTM(units=reduced_embedding_dimension,
                                           dropout=dropout_embedder,
                                           recurrent_dropout=dropout_embedder,
                                           kernel_regularizer=keras.regularizers.l2(l2_regularization),
                                           recurrent_regularizer=keras.regularizers.l2(l2_regularization),
                                           bias_regularizer=keras.regularizers.l2(l2_regularization),
                                           return_sequences=False,
                                           unroll=False,  # not possible to unroll if the time shape is not specified
                                           name='claim_LSTM'),
                                      merge_mode='mul',
                                      name='claim_biLSTM'
                                      )(prev_claim_l)

        target_embed2 = Bidirectional(LSTM(units=reduced_embedding_dimension,
                                           dropout=dropout_embedder,
                                           recurrent_dropout=dropout_embedder,
                                           kernel_regularizer=keras.regularizers.l2(l2_regularization),
                                           recurrent_regularizer=keras.regularizers.l2(l2_regularization),
                                           bias_regularizer=keras.regularizers.l2(l2_regularization),
                                           return_sequences=False,
                                           unroll=False,  # not possible to unroll if the time shape is not specified
                                           name='topic_LSTM'),
                                      merge_mode='mul',
                                      name='topic_biLSTM'
                                      )(prev_topic_l)

    if distance > 0:
        prev_l = Concatenate(name='embed_merge')([source_embed2, target_embed2, dist_il])
    else:
        prev_l = Concatenate(name='embed_merge')([source_embed2, target_embed2])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(l2_regularization),
                   bias_regularizer=keras.regularizers.l2(l2_regularization),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(input_layer=prev_l, regularizer_weight=l2_regularization,
                         layers=resnet_layers, res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_final, name='final_dropout')(prev_l)

    # Classifiers

    topic_target_sentiment = Dense(units=2,
                                   name='topic_target_sentiment',
                                   activation='softmax',
                                   )(prev_l)

    claim_target_sentiment = Dense(units=2,
                                   name='claim_target_sentiment',
                                   activation='softmax',
                                   )(prev_l)

    stance_rel = Dense(units=2,
                       name='stance_relation',
                       activation='softmax',
                       )(prev_l)

    targets_relation = Dense(units=2,
                             name='targets_relation',
                             activation='softmax',
                             )(prev_l)

    model_inputs = [claim_il, topic_il]
    if distance > 0:
        model_inputs.append(dist_il)

    full_model = keras.Model(inputs=model_inputs,
                             outputs=(topic_target_sentiment, claim_target_sentiment, stance_rel, targets_relation),
                             )

    return full_model
