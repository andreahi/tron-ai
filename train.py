import numpy as np
import redis
import json

import json
import os
import shutil
import time

import tensorflow as tf
import redis
import numpy as np
from keras import regularizers
from tensorflow.python.saved_model.simple_save import simple_save

r = redis.Redis(host='localhost', port=6379, db=0)


def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)

def shuffle_in_unison(a, b, c, d, e, f):
    if len(a) != len(b) or len(b) != len(c) or len(c) != len(d) or len(d) != len(e) or len(e) != len(f):
        print(len(a))
        print(len(b))
        print(len(c))
        print(len(d))
        print(len(e))
        print(len(f))
        raise Exception('inconsistent data length')
    rng_state = np.random.get_state()
    np.random.shuffle(a)

    np.random.set_state(rng_state)
    np.random.shuffle(b)

    np.random.set_state(rng_state)
    np.random.shuffle(c)

    np.random.set_state(rng_state)
    np.random.shuffle(d)

    np.random.set_state(rng_state)
    np.random.shuffle(e)

    np.random.set_state(rng_state)
    np.random.shuffle(f)

def build_layer(x, num_units, keep_prob, layer_n = False, dropout = False):
    """Builds a layer with input x; dropout and layer norm if specified."""

    init_s = 0.3


    nn = tf.contrib.layers.fully_connected(
        x,
        num_units,
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=None if not layer_n else tf.contrib.layers.layer_norm,
        normalizer_params={},
        weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
    )

    nn = tf.keras.layers.Dense(num_units,
                               activation=tf.nn.leaky_relu,
                               activity_regularizer=regularizers.l1(0.01)
                               )(x)
    if dropout:
        nn = tf.nn.dropout(nn, keep_prob)

    return nn


def forward_pass(x, single_value_inputs, keep_prob):
    init_s = 0.3

    with tf.variable_scope("model_weighted", reuse=tf.AUTO_REUSE):
        nn1 = tf.concat([tf.layers.flatten(x)], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn1 = build_layer(nn1, num_units, keep_prob, dropout=False)

        nn2 = tf.concat([individual_values], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn2 = build_layer(nn2, num_units, keep_prob, dropout=False)

        layer_y_0 = build_layer(build_layer(tf.concat([nn1, nn2], axis=1), 10, keep_prob, dropout=False), 5, keep_prob, dropout=False)

        reward_error_pred = tf.layers.dense(layer_y_0, 1, kernel_initializer=tf.random_uniform_initializer(-init_s, init_s))
        
        nn1 = tf.concat([tf.layers.flatten(x)], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn1 = build_layer(nn1, num_units, keep_prob, dropout=False)

        nn2 = tf.concat([individual_values], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn2 = build_layer(nn2, num_units, keep_prob, dropout=False)

        layer_y_0 = build_layer(build_layer(tf.concat([nn1, nn2], axis=1), 10, keep_prob, dropout=False), 5, keep_prob, dropout=False)

        reward_pred = tf.layers.dense(layer_y_0, 1, kernel_initializer=tf.random_uniform_initializer(-init_s, init_s))


        nn1 = tf.concat([tf.layers.flatten(x)], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn1 = build_layer(nn1, num_units, keep_prob, dropout=False)

        nn2 = tf.concat([individual_values], axis=1)
        for num_units in [5, 5, 5]:
            if num_units > 0:
                nn2 = build_layer(nn2, num_units, keep_prob, dropout=False)

        layer_y_1 = build_layer(build_layer(tf.concat([nn1, nn2], axis=1), 10, keep_prob, dropout=False), 5, keep_prob, dropout=False)

        action_pred = None
        for _ in range(5):
            dense = layer_y_1
            for num_units in [5, 5]:
                dense = build_layer(dense, num_units, keep_prob, dropout=False)
            if action_pred != None:
                action_pred = tf.concat([action_pred, [build_layer(dense, 1, keep_prob, dropout=False)]], axis=2)
            else:
                action_pred = [build_layer(dense, 1, keep_prob, dropout=False)]


    return addNameToTensor(reward_pred, "reward_pred"), addNameToTensor(action_pred, "action_pred"), addNameToTensor(reward_error_pred, "reward_error_pred")


num_actions = 5

with tf.Session() as sess:
    max_grad_norm = 50.0
    batch_size = 6300

    food = tf.placeholder(
        shape=[None, 50, 50],
        dtype=tf.float32,
        name="food")

    individual_values = tf.placeholder(
        shape=[None, 2],
        dtype=tf.float32,
        name="individual_values")

    # reward vector
    reward = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")
    reward_error = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")
    next_reward_error = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")

    # reward vector
    next_pred_reward = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")

    # weights (1 for selected action, 0 otherwise)
    weights = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))

    actions_performed = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))


    actions_target = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))



    graph = tf.Graph()

    keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

    reward_pred, action_pred, reward_error_pred = forward_pass(food, individual_values, keep_prob)


    loss = tf.abs(action_pred - actions_target)
    print("shape loss: ", str(loss.get_shape()))
    expected_diff = tf.stop_gradient(tf.divide(tf.abs(reward_pred - next_pred_reward), tf.abs(reward_error) + tf.abs(next_reward_error)))
    loss_action = tf.multiply(actions_performed, loss)
    #loss = tf.Print(loss_action, [loss_action])
    weighted_action_loss = tf.reduce_sum(tf.multiply(expected_diff, loss_action))
    #weighted_loss = loss

    reward_diff = tf.abs(reward_pred - reward)
    reward_loss = tf.reduce_sum(reward_diff)

    reward_error_loss = tf.reduce_sum(tf.abs(reward_diff - reward_error_pred))

    cost = (weighted_action_loss / batch_size)



    optimizer = tf.train.AdamOptimizer(0.001)
    optimizer_reward = tf.train.AdamOptimizer(0.001)
    optimizer_reward_error = tf.train.AdamOptimizer(0.001)

    train_op = optimizer.minimize(loss)
    train_op_reward = optimizer_reward.minimize(reward_loss)
    train_op_reward_error = optimizer_reward.minimize(reward_error_loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with graph.as_default():
        sess.run(init)

    if os.path.isdir('model'):
        saver.restore(sess, "model_tmp/model.ckpt")

    x_train = []
    reward_train = []
    actions_train = []
    individual_values_train = []
    next_x_train = []
    next_individual_values_train = []

    while True:


        '''
        stored_next_x_train = r.get("next_x_train")
        if stored_next_x_train:
            next_x_train = np.array(json.loads(stored_next_x_train))

        stored_x_train = r.get("x_train")
        if stored_x_train:
            x_train = np.array(json.loads(stored_x_train))

        stored_reward_train = r.get("reward_train")
        if stored_reward_train:
            reward_train = np.array(json.loads(stored_reward_train))

        stored_actions_train = r.get("actions_train")
        if stored_actions_train:
            actions_train = np.array(json.loads(stored_actions_train))

        stored_individual_values_train = r.get("individual_values")
        if stored_individual_values_train:
            individual_values_train = np.array(json.loads(stored_individual_values_train))

        stored_next_individual_values_train = r.get("next_individual_values_train")
        if stored_next_individual_values_train:
            next_individual_values_train = np.array(json.loads(stored_next_individual_values_train))

        '''

        new_data = r.lpop('sample')

        if new_data:
            data = json.loads(new_data.decode('utf-8'))
            #print(r.get('data'))


            if not data["myBody"]:
                print("Got empty dataset :s ")
                continue

            next_x_train = next_x_train + data['otherBody'][1:]

            x_train = x_train + data['otherBody'][0:-1]


            score = np.ones(len(data['action'])-1) * (len(data['action'])/100.0)

            if len(reward_train) > 0:
                reward_train = np.concatenate([reward_train, np.expand_dims(score, axis=1)])
            else:
                reward_train = np.expand_dims(score, axis=1)

            actions_train = actions_train + data['action'][0:-1]

            individual_values_train = individual_values_train + data['myHead'][0:-1]

            next_individual_values_train = next_individual_values_train + data['myHead'][1:]



            #r.set("x_train", json.dumps(x_train.tolist()))
            #r.set("reward_train", json.dumps(reward_train.tolist()))
            #r.set("actions_train", json.dumps(actions_train.tolist()))
            #r.set("individual_values", json.dumps(individual_values_train.tolist()))
            #r.set("next_individual_values_train", json.dumps(next_individual_values_train.tolist()))
            #r.set("next_x_train", json.dumps(next_x_train.tolist()))

        print("dataset size: ", len(x_train))

        step = 10000

        if step > len(x_train):
            time.sleep(0.1)
            continue

        x_train = np.array(x_train)
        reward_train = np.array(reward_train)
        actions_train = np.array(actions_train)
        individual_values_train = np.array(individual_values_train)
        next_x_train = np.array(next_x_train)
        next_individual_values_train = np.array(next_individual_values_train)

        shuffle_in_unison(next_x_train, next_individual_values_train, x_train, individual_values_train, reward_train, actions_train)


        for _ in range(100):
            for i in range(0, 100000, step):
                if i + step > len(x_train):
                    break

                _, reward_loss_v = sess.run(
                    [train_op_reward, reward_loss],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], reward: reward_train[i:i+step],
                               keep_prob: 0.99})

                _, reward_error_loss_v = sess.run(
                    [train_op_reward_error, reward_error_loss],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], reward: reward_train[i:i+step],
                               keep_prob: 0.99})


                print(" reward_loss_v: " + str(reward_loss_v) +  " reward_error_loss_v: " + str(reward_error_loss_v))

        for _ in range(20):
            for i in range(0, 100000, step):
                if i + step > len(x_train):
                    break


                y_error_pred_v = sess.run(
                    [reward_error_pred],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], keep_prob: 1.0})[0]

                y_pred_v = sess.run(
                    [reward_pred],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], keep_prob: 1.0})[0]


                next_y_pred_v = sess.run(
                    [reward_pred],
                    feed_dict={food: next_x_train[i:i+step], individual_values:next_individual_values_train[i:i+step], keep_prob: 1.0})[0]

                next_y_error_pred_v = sess.run(
                    [reward_error_pred],
                    feed_dict={food: next_x_train[i:i+step], individual_values:next_individual_values_train[i:i+step], keep_prob: 1.0})[0]


                actions_target_train = np.zeros([len(reward_train[i:i+step]), num_actions])
                positive_target = (next_y_pred_v - y_pred_v) > 0.
                negative_target = (next_y_pred_v - y_pred_v) <= 0.

                actions_target_train[np.squeeze(positive_target)] = 1
                actions_target_train[np.squeeze(negative_target)] = 0

                #print(y_pred_v)
                #print(next_y_pred_v)
                #print(actions_target_train)
                #print(actions_train[i:i+step])

                action_pred_v = sess.run(
                    [action_pred],
                    feed_dict={food: x_train, individual_values: individual_values_train, keep_prob: 1.0})

                #print(action_pred_v)


                _, cost_train, weighted_action_loss_v = sess.run(
                    [train_op, cost, weighted_action_loss],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], reward: reward_train[i:i+step], actions_performed: actions_train[i:i+step],
                               actions_target: actions_target_train, next_pred_reward: next_y_pred_v,
                               reward_error: y_error_pred_v, next_reward_error: next_y_error_pred_v,
                               keep_prob: 0.99})

                print("cost_train: " + str(cost_train) + " weighted_action_loss_v: " + str(weighted_action_loss_v))




        shutil.rmtree('model', ignore_errors=True)

        simple_save(sess,
                    "model",
                    inputs={"input": food},
                    outputs={"action_pred": action_pred, "reward_pred": reward_pred})
        save_path = saver.save(sess, "model_tmp/model.ckpt")
        #r.flushall()
        x_train = []
        reward_train = []
        actions_train = []
        individual_values_train = []
        next_x_train = []
        next_individual_values_train = []
        #action_pred_v = sess.run(
        #    [action_pred],
        #    feed_dict={food: x_train, individual_values: individual_values_train, keep_prob: 1.0})
        #print(action_pred_v)


