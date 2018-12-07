from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def lr_annealing(learning_rate, global_step, alpha, beta, name=None):
    """
    Applies learning rate annealing to the initial learning rate
    return lr_p = learning_rate * (1 + alpha * global_step)^(-beta)

    Args:   learning_rate:
            global_step: number of iterations
            alpha:
            beta:
    """
    with ops.name_scope(name, "Lr_Annealing", [learning_rate, global_step, alpha, beta]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        alpha = math_ops.cast(alpha, dtype)
        beta = math_ops.cast(beta, dtype)
        base = math_ops.multiply(alpha, global_step)
        base = math_ops.add(1., base)
        return math_ops.multiply(learning_rate, math_ops.pow(base, -beta), name=name)


def reverse_gradient_weight(current_epoch, total_epochs, gamma):
    dtype = 'float32'
    current_epoch = math_ops.cast(current_epoch, dtype)
    total_epochs = math_ops.cast(total_epochs, dtype)
    gamma = math_ops.cast(gamma, dtype)
    p = math_ops.divide(current_epoch, total_epochs)
    alpha = math_ops.multiply(-gamma, p)
    alpha = math_ops.exp(alpha)
    alpha = math_ops.add(1., alpha)
    alpha = math_ops.divide(2., alpha)
    return math_ops.add(alpha, -1.)
