import tensorflow as tf


@tf.function
def masked_accuracy(true, pred, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(true, 1), tf.argmax(pred, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


@tf.function
def masked_softmax_cross_entropy(true, pred, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=true)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)
