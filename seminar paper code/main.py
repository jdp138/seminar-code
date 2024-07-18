import os
import numpy as np
import argparse
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(model):
    prelosses, postlosses, preaccs, postaccs = [], [], [], []
    best_acc = 0
    patience = 5000  # Number of iterations to wait before early stopping
    wait = 0  # Counter for early stopping

    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(log_dir)

    for iteration in range(600000):
        with tf.GradientTape() as tape:
            support_pred = model.forward(model.support_x, model.weights, training=True)
            support_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=model.support_y))
            grads = tape.gradient(support_loss, model.weights.values())
            model.optimizer.apply_gradients(zip(grads, model.weights.values()))

        if iteration % 200 == 0:
            query_pred = model.forward(model.query_x, model.weights, training=False)
            query_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=model.query_y))
            query_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.argmax(model.query_y, axis=1), tf.argmax(query_pred, axis=1)))

            prelosses.append(support_loss.numpy())
            postlosses.append(query_loss.numpy())
            preaccs.append(query_acc.numpy())

            with summary_writer.as_default():
                tf.summary.scalar('support_loss', support_loss.numpy(), step=iteration)
                tf.summary.scalar('query_loss', query_loss.numpy(), step=iteration)
                tf.summary.scalar('query_acc', query_acc.numpy(), step=iteration)

            print(iteration, 'loss:', np.mean(prelosses), '=>', np.mean(postlosses), 'acc:', np.mean(preaccs))
            prelosses, postlosses, preaccs = [], [], []

        if iteration % 2000 == 0:
            test_accs = []
            for _ in range(200):
                test_query_pred = model.forward(model.query_x_test, model.weights, training=False)
                test_query_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.argmax(model.query_y_test, axis=1), tf.argmax(test_query_pred, axis=1)))
                test_accs.append(test_query_acc.numpy())

            acc = np.mean(test_accs)
            print('Validation acc:', acc, 'best:', best_acc)

            if acc > best_acc:
                best_acc = acc
                wait = 0  # Reset the counter
                model.save_weights('ckpt/model.ckpt')
                print('Model saved.')
            else:
                wait += 1

            if wait >= patience:
                print("Early stopping due to no improvement in validation accuracy.")
                break

def test(model):
    np.random.seed(1)
    random.seed(1)

    test_accs = []
    for i in range(600):
        if i % 100 == 0:
            print(i)
        test_query_pred = model.forward(model.query_x_test, model.weights, training=False)
        test_query_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.argmax(model.query_y_test, axis=1), tf.argmax(test_query_pred, axis=1)))
        test_accs.append(test_query_acc.numpy())

    test_accs = np.array(test_accs)
    means = np.mean(test_accs, axis=0)
    stds = np.std(test_accs, axis=0)
    ci95 = 1.96 * stds / np.sqrt(600)

    print('Test accuracy mean:', means)
    print('Test accuracy stds:', stds)
    print('Test accuracy ci95:', ci95)

def main():
    training = not args.test
    kshot = 1
    kquery = 15
    nway = 5
    meta_batchsz = 4
    K = 5

    db = DataGenerator(nway, kshot, kquery, meta_batchsz, 200000)

    if training:
        image_tensor, label_tensor = db.make_data_tensor(training=True)
        support_x = tf.slice(image_tensor, [0, 0, 0], [-1, nway * kshot, -1], name='support_x')
        query_x = tf.slice(image_tensor, [0, nway * kshot, 0], [-1, -1, -1], name='query_x')
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, nway * kshot, -1], name='support_y')
        query_y = tf.slice(label_tensor, [0, nway * kshot, 0], [-1, -1, -1], name='query_y')

    image_tensor, label_tensor = db.make_data_tensor(training=False)
    support_x_test = tf.slice(image_tensor, [0, 0, 0], [-1, nway * kshot, -1], name='support_x_test')
    query_x_test = tf.slice(image_tensor, [0, nway * kshot, 0], [-1, -1, -1], name='query_x_test')
    support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1, nway * kshot, -1], name='support_y_test')
    query_y_test = tf.slice(label_tensor, [0, nway * kshot, 0], [-1, -1, -1], name='query_y_test')

    model = MAML(84, 3, 5)

    if training:
        model.build(support_x, support_y, query_x, query_y, K, meta_batchsz, mode='train')
        model.query_x_test = query_x_test
        model.query_y_test = query_y_test
        model.support_x_test = support_x_test
        model.support_y_test = support_y_test
    else:
        model.build(support_x_test, support_y_test, query_x_test, query_y_test, K + 5, meta_batchsz, mode='test')
        model.query_x_test = query_x_test
        model.query_y_test = query_y_test
        model.support_x_test = support_x_test
        model.support_y_test = support_y_test

    if training:
        train(model)
    else:
        test(model)

if __name__ == "__main__":
    main()
