import math
import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import variance_scaling_initializer

from functools import reduce
from operator import mul


initializer = lambda : variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)
initializer_relu = lambda : variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
regularizer = l2_regularizer(scale=3e-7)


def glu(x):
    """Gated Linear Units
    Args:
        x: [batch_size, seq_len, 2 * hidden_size]

    Returns:
        glu(x): [batch_size, seq_len, hidden_size]
    """
    x, x_h = tf.split(x, 2, axis=-1)
    return tf.sigmoid(x) * x_h

def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization.

    apply normalization along the last dimension.
    """
    with tf.name_scope(scope, default_name='noam_norm', values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) \
               * tf.sqrt(tf.to_float(shape[-1]))

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name='layer_norm', values=[x], reuse=reuse):
        scale = tf.get_variable(
            name='layer_norm_scale', shape=[filters], regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            name='layer_norm_bias', shape=[filters], regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

norm_fn = layer_norm

def highway(x, size=None, activation=None, num_layers=2, scope='highway', dropout=0.0, reuse=None):
    """Use two convolution layers to implement highway"""
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name='input_projection', reuse=reuse)

        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid, name='gate_%d'%i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation, name='activation_%d'%i, reuse=reuse)

            H = tf.nn.dropout(H, 1.0 - dropout)

            x = T * H + (1.0 - T) * x

        return x

def layer_dropout(inputs, residual, dropout):
    """implement stochastic residual connection"""
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred,
                   lambda : residual,
                   lambda : tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size,
                   mask=None, num_filters=128, input_projection=False, num_heads=8,
                   seq_len=None, scope='res_block', is_training=True, reuse=None, bias=True, dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name='input_projection', reuse=reuse)

        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs) # add 1d time signal to every block's inputs

            outputs, sublayer = conv_block(outputs,
                                           num_conv_layers,
                                           kernel_size,
                                           num_filters,
                                           seq_len=seq_len,
                                           scope='encoder_block_%d'%i,
                                           reuse=reuse,
                                           bias=bias,
                                           dropout=dropout,
                                           sublayers=(sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs,
                                                     num_filters,
                                                     seq_len,
                                                     mask=mask,
                                                     num_heads=num_heads,
                                                     scope='self_attention_layer_%d'%i,
                                                     reuse=reuse,
                                                     is_training=is_training,
                                                     bias=bias,
                                                     dropout=dropout,
                                                     sublayers=(sublayer, total_sublayers))
        return outputs

def conv_block(inputs, num_conv_block, kernel_size, num_filters, seq_len=None, scope='conv_block', is_training=True,
               reuse=None, bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.expand_dims(inputs, axis=2)
        l, L = sublayers
        for i in range(num_conv_block):
            residual = outputs  # the input to the current layer
            outputs = norm_fn(outputs, scope='layer_norm_%d'%i, reuse=reuse)    # norm layer
            if i % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout) # apply dropout at even layer
            output = depthwise_separable_convolution(
                outputs, kernel_size=(kernel_size, 1), num_filters=num_filters,
                scope='depthwise_conv_layers_%d'%i, is_training=is_training, reuse=reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)  # according to the layer num
            l += 1
        return tf.squeeze(outputs, axis=2), l

def self_attention_block(inputs, num_filters, seq_len, mask=None, num_heads=8,
                         scope='self_attention_ffn', reuse=None, is_training=True,
                         bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayers

        # Self attention
        outputs = norm_fn(inputs, scope='layer_norm_1', reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multi_head_attention(
            outputs, num_filters, num_heads=num_heads, seq_len=seq_len, reuse=reuse,
            mask=mask, is_training=is_training, bias=bias, dropout=dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1

        # Feed forward layer
        outputs = norm_fn(residual, scope='layer_norm_2', reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name='FFN_1', reuse=reuse)
        outputs = conv(outputs, num_filters, True, None, name='FFN_2', reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l

def multi_head_attention(queries, units, num_heads,
                         memory=None,
                         seq_len=None,
                         scope='Multi_Head_Attention',
                         reuse=None,
                         mask=None,
                         is_training=True,
                         bias=True,
                         dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        # Self Attention
        if memory is None:
            memory = queries

        # Linear projection use convolution nets
        memory = conv(memory, 2 * units, name='memory_projection', reuse=reuse)
        query = conv(queries, units, name='query_projection', reuse=reuse)

        # Turn to multi head
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, num_or_size_splits=2, axis=2)]

        key_depth_per_head = units / num_heads
        Q *= key_depth_per_head ** -0.5 # scale
        x = dot_product_attention(Q, K, V,
                                  bias=bias,
                                  seq_len=seq_len,
                                  mask=mask,
                                  is_training=is_training,
                                  scope='dot_production_attention',
                                  reuse=reuse,
                                  dropout=dropout)
        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def conv(inputs, output_size, bias=None,
         activation=None, kernel_size=1, name='conv', reuse=None):
    """inputs shape can be 3D or 4D.

        [batch_size, seq_len, hidden_size] or [batch_size, 1, seq_len, hidden_size]
        depend on the inputs shape, we choose to use tf.nn.conv1d or tf.nn.conv2d.

    """
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:  # [batch_size, 1, seq_len, hidden_size]
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]

        else:                   # [batch_size, seq_len, hidden_size]
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1

        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable(
            name='kernel_', shape=filter_shape, dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, 'VALID')  # valid or same ?
        if bias:
            outputs += tf.get_variable(
                name='bias_', shape=bias_shape,
                regularizer=regularizer,
                initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs

def mask_logits(inputs, mask, mask_value=-1e30):
    """inputs and mask have same shape"""
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope='depthwise_separable_convolution',
                                    bias=True, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable(
            name='depthwise_filter',
            shape=[kernel_size[0], kernel_size[1], shapes[-1], 1], # output_size == 1
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu())
        pointwise_filter = tf.get_variable(
            name='pointwise_filter',
            shape=[1, 1, shapes[-1], num_filters],  # kernel_size == 1
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=initializer_relu())

        outputs = tf.nn.separable_conv2d(
            input=inputs,
            depthwise_filter=depthwise_filter,
            pointwise_filter=pointwise_filter,
            strides=[1, 1, 1, 1],
            padding='SAME')

        if bias:
            b = tf.get_variable(
                name='bias', shape=outputs.shape[-1],
                regularizer=regularizer,
                initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs

def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimension.
    The first of these two dimensions is n.

    Two steps: 1) Reshape 2) Transpose

    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]    # m
    new_shape = old_shape[: -1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[: -1], [n, -1]], axis=0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])

def dot_product_attention(q, k, v,
                          bias,
                          seq_len=None,
                          mask=None,
                          is_training=True,
                          scope=None,
                          reuse=None,
                          dropout=0.0):
    """Dot product attention.

    Args:
        q: [batch, heads, length_q, depth_k]
        k: [batch, heads, length_kv, depth_k]
        v: [batch, heads, length_kv, depth_v]
        bias: attention_bias()
        is_training: a bool of training
        scope: an optional string
    Returns:
        A Tensor.
    """
    with tf.variable_scope(scope, default_name='dot_product_attention', reuse=reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable(
                name='bias',
                shape=logits.shape[-1],
                regularizer=regularizer,
                initializer=tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, shape=[shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)

        weights = tf.nn.softmax(logits, name='attention_weights')

        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)

        return tf.matmul(weights, v)

def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
        x: a Tensor with shape [..., a, b]
    Returns:
        a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2: ]
    new_shape = old_shape[: -2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[: -2], [-1]], axis=0))
    ret.set_shape(new_shape)
    return ret

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Args:
        x: Tensor with shape of [batch, length, channels]
        min_timescale: float
        max_timescale: float

    Returns:
        a Tensor with the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[-1]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing emebddings to create.
            The number of different timescales is equal to channels / 2
        min_timescale: float
        max_timescale: float

    Returns:
        a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, axis=1) * tf.expand_dims(inv_timescales, axis=0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def ndim(x):
    """Returns the number of exes in a tensor, as an integer."""
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def dot(x, y):
    """"""
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)

        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)

        y_shape = tuple(y_shape)

        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, shape=[-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim),
                        shape=[y_shape[-2], -1])

        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[: -1] + y_shape[: -2] + y_shape[-1: ])

    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Batch-wise dot product.
    keras==2.0.6

    'batch_dot' is used to compute dot product of 'x' and 'y' when
    'x' and 'y' are data in batch, i.e. in a shape of '(batch_size, :)'.

    'batch_dot' results in a tensor or variable with less dimensions than the input.
    If the number of dimensions is reduced to 1, we use 'expand_dims' to make sure
    that ndim is at least 2.

    Args:
        x: A Tensor or variable with 'ndim >= 2'
        y: A Tensor of variable with 'ndim >= 2'
        axes: list of (or single) in with target dimensions.
            The lengths of 'axes[0]' and 'axes[1]' should be the same.

    Returns:
        A Tensor with shape equal to the concatenation of 'x''s shape
        (less the dimension that was summed over) and 'y''s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to '(batch_size, 1)'
    """
    if isinstance(axes, int):
        axes = (axes, axes)

    x_ndim = ndim(x)
    y_ndim = ndim(y)

    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0

    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes=[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axis=[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, axis=1)
    return out

def optimized_trilinear_for_attention(args,
                                      c_maxlen,
                                      q_maxlen,
                                      input_keep_prob=1.0,
                                      scope = 'efficient_trilinear',
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_initializer=initializer()):
    """trilinear for attention.

    c, q = args : [batch_size, {c/q}_len, depth]

    """
    assert len(args) == 2, 'just use for computing attention with two input'
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()

    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("'args' must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("The last dimension of 'args' must be equal")

    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    print('dtype: {}'.format(dtype))
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            name='linear_kernel4arg0',
            shape=[arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            name='linear_kernel4arg1',
            shape=[arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mul = tf.get_variable(
            name='linear_kernel4mul',
            shape=[1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            name='linear_bias',
            shape=[1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)

        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=[0, 2, 1]), [1, c_maxlen, 1])
        subres2 = batch_dot(droped_args[0] * weights4mul, tf.transpose(droped_args[1], perm=[0, 2, 1]))
        res = subres0 + subres1 + subres2
        # res = nn_ops.bias_add(res, biases)  # [batch_size, c_maxlen, q_maxlen]
        return res

def trilinear(args,
              output_size=1,
              bias=True,
              squeeze=False,
              wd=0.0,
              input_keep_prob=1.0,
              scope='trilinear'):
    with tf.variable_scope(scope):
        flat_args = [flatten(arg, 1) for arg in args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = reconstruct(flat_out, args[0], 1)
        return tf.squeeze(out, -1)

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()

    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep

    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]

    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def _linear(args,
            output_size,
            bias,
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=initializer(),
            scope=None,
            reuse=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
            (default is all zeros).
        kernel_initializer: starting value to initialize the weight.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: If some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("'args' must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError('linear is expecting 2D arguments: %s' % shapes)
        if shape[1].value is None:
            raise ValueError('linear expects shape[1] to be provided for shape %s, '
                             'but saw %s' % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now to computatin.
    with tf.variable_scope(scope, reuse=reuse) as outer_scope:
        weights = tf.get_variable(
            name='linear_kernel',
            shape=[total_arg_size, output_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, axis=1), weights)
        if not bias:
            return res

        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                name='linear_bias',
                shape=[output_size],
                dtype=dtype,
                regularizer=regularizer,
                initializer=initializer)
            return nn_ops.bias_add(res, biases)

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total number of trainable parameters: {}'.format(total_parameters))

#####################################################################################################
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell, GRUCell


class BiRNNEncoder(object):
    def __init__(self, hidden_size, num_layers, name):
        self._cell = GRUCell(num_units=hidden_size)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.name = name

    def __call__(self, inputs, sequence_length, dropout_prob=0.3, is_last_states=False):
        with tf.variable_scope(self.name, initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE):

            # cell_fw = MultiRNNCell(cells=[self._cell for _ in range(self._num_layers)])
            # cell_bw = MultiRNNCell(cells=[self._cell for _ in range(self._num_layers)])
            cell_fw = self._cell
            cell_bw = self._cell
            outputs, last_states = bidirectional_dynamic_rnn(cell_bw=cell_bw,
                                                             cell_fw=cell_fw,
                                                             dtype="float32",
                                                             sequence_length=sequence_length,
                                                             inputs=inputs,
                                                             swap_memory=True,
                                                             scope=self.name)
            # print('{} ouputs: {}'.format(self.name, outputs))
            # print('{} last_states: {}'.format(self.name, last_states))

            if is_last_states:
                # outputs shape: (None, hidden_size * 2)
                outputs = tf.concat(last_states, axis=-1)
            else:
                # outputs shape: (None, max_length, hidden_size * 2)
                outputs = tf.concat(outputs, axis=-1)

            # dropout
            if dropout_prob:
                outputs = tf.layers.dropout(outputs, dropout_prob)

            return outputs
