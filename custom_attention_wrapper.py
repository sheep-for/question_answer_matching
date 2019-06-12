from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops, gen_math_ops, state_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import control_flow_ops

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.

    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.

    Returns:
      A (possibly masked), checked, new `memory`.

    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    memory = nest.map_structure(
        lambda m: ops.convert_to_tensor(m, name="memory"), memory)
    if memory_sequence_length is not None:
        memory_sequence_length = ops.convert_to_tensor(
            memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError("Expected memory %s to have fully defined inner dims, "
                                 "but saw shape: %s" % (m.name, m.get_shape()))

        nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = array_ops.sequence_mask(
            memory_sequence_length,
            maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
            dtype=nest.flatten(memory)[0].dtype)
        seq_len_batch_size = (
                memory_sequence_length.shape[0].value
                or array_ops.shape(memory_sequence_length)[0])

    def _maybe_mask(m, seq_len_mask_):
        rank = m.get_shape().ndims
        rank = rank if rank is not None else array_ops.rank(m)
        extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
        m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
        if memory_sequence_length is not None:
            message = ("memory_sequence_length and memory tensor batch sizes do not "
                       "match.")
            with ops.control_dependencies([
                check_ops.assert_equal(
                    seq_len_batch_size, m_batch_size, message=message)]):
                seq_len_mask_ = array_ops.reshape(
                    seq_len_mask_,
                    array_ops.concat((array_ops.shape(seq_len_mask_), extra_ones), 0))
                return m * seq_len_mask_
        else:
            return m

    return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    message = "All values in memory_sequence_length must greater than zero."
    with ops.control_dependencies(
            [check_ops.assert_positive(memory_sequence_length, message=message)]):
        score_mask = array_ops.sequence_mask(
            memory_sequence_length, maxlen=array_ops.shape(score)[1])
        score_mask_values = score_mask_value * array_ops.ones_like(score)
        return array_ops.where(score_mask, score, score_mask_values)


def _bahdanau_score(processed_query, keys, normalize):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set `normalize=True`.

    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      normalize: Whether to normalize the score function.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = array_ops.expand_dims(processed_query, 1)
    v = variable_scope.get_variable(
        "attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        return math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + b), [2])
    else:
        return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])


# _CustomBaseAttentionMechanism for CustomBahdanauAttentionV2
class _CustomBaseAttentionMechanism(AttentionMechanism):
    """A base AttentionMechanism class providing common functionality.

    Common functionality includes:
      1. Storing the query and memory layers.
      2. Preprocessing and storing the memory.
    """

    def __init__(self,
                 query_layer,
                 memory,
                 probability_fn,
                 memory_sequence_length=None,
                 memory_layer=None,
                 check_inner_dims_defined=True,
                 score_mask_value=float("-inf"),
                 name=None):
        """Construct base AttentionMechanism class.

        Args:
          query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
            must match the depth of `memory_layer`.  If `query_layer` is not
            provided, the shape of `query` must match that of `memory_layer`.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          probability_fn: A `callable`.  Converts the score and previous alignments
            to probabilities. Its signature should be:
            `probabilities = probability_fn(score, previous_alignments)`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
            depth must match the depth of `query_layer`.
            If `memory_layer` is not provided, the shape of `memory` must match
            that of `query_layer`.
          check_inner_dims_defined: Python boolean.  If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        """
        if (query_layer is not None
                and not isinstance(query_layer, layers_base.Layer)):
            raise TypeError(
                "query_layer is not a Layer: %s" % type(query_layer).__name__)
        if (memory_layer is not None
                and not isinstance(memory_layer, layers_base.Layer)):
            raise TypeError(
                "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        self._memory_sequence_length = memory_sequence_length  # add
        self._score_mask_value = score_mask_value  # add
        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" %
                            type(probability_fn).__name__)
        self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
            probability_fn(
                _maybe_mask_score(score, memory_sequence_length, score_mask_value),
                prev))
        with ops.name_scope(
                name, "CustomBaseAttentionMechanismInit", nest.flatten(memory)):
            self._values = _prepare_memory(
                memory, memory_sequence_length,
                check_inner_dims_defined=check_inner_dims_defined)
            self._keys = (
                self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
                else self._values)
            self._batch_size = (
                    self._keys.shape[0].value or array_ops.shape(self._keys)[0])
            self._alignments_size = (self._keys.shape[1].value or
                                     array_ops.shape(self._keys)[1])

    @property
    def memory_layer(self):
        return self._memory_layer

    @property
    def query_layer(self):
        return self._query_layer

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)


class CustomBahdanauAttentionV2(_CustomBaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 top_alignment_number,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="CustomBahdanauAttentionV2"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        """
        self._top_alignment_number = top_alignment_number
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(CustomBahdanauAttentionV2, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)

        # mask with memory_sequence_length
        mask_score = _maybe_mask_score(score, self._memory_sequence_length, self._score_mask_value)

        # choose top_k alignments among dimension 1. replace others with -inf
        top_k = control_flow_ops.cond(gen_math_ops.less(
            self.alignments_size, self._top_alignment_number),
            lambda: self.alignments_size, lambda: self._top_alignment_number)
        _, score_mask_index = nn_ops.top_k(mask_score, top_k)
        score_mask_index_final = array_ops.concat(
            [array_ops.reshape(
                [i * array_ops.ones([top_k], dtypes.int32) for i in range(self.batch_size)],
                [-1, 1]),
                array_ops.reshape(score_mask_index, [-1, 1])],
            axis=-1)
        score_mask_ = sparse_ops.sparse_to_dense(
            sparse_indices=score_mask_index_final,
            output_shape=[self.batch_size, self.alignments_size],
            sparse_values=True, default_value=False, validate_indices=False)
        score_mask_values_ = self._score_mask_value * array_ops.ones_like(mask_score)
        keywords_score = array_ops.where(score_mask_, mask_score, score_mask_values_)

        alignments = nn_ops.softmax(keywords_score)

        return alignments


# alternative
class CustomBahdanauAttentionVplus(_CustomBaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 top_alignment_number,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="CustomBahdanauAttentionV2"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          top_alignment_number: int32, the same shape with memory_sequence_length. the keyword_num
            for top_k alignment.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths. [batch_size]
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        """
        self._top_alignment_number = top_alignment_number
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(CustomBahdanauAttentionVplus, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)

        # mask with memory_sequence_length
        mask_score = _maybe_mask_score(score, self._memory_sequence_length, self._score_mask_value)
      
        top_keyword_mask = array_ops.sequence_mask(
            self._top_alignment_number, maxlen=self.alignments_size)
        score_mask_index = nn_ops.top_k(mask_score, self.alignments_size, False).indices
        score_mask_index_reshape = array_ops.reshape(
            math_ops.cast(
                array_ops.where(top_keyword_mask, math_ops.cast(score_mask_index, dtypes.float32),
                                array_ops.zeros_like(mask_score)),
                dtypes.int32),
            [-1, 1])
        temp_index = array_ops.reshape(
            [i * array_ops.ones([self.alignments_size], dtypes.int32) for i in range(self.batch_size)],
            [-1, 1])
        score_mask_index_final = array_ops.concat([temp_index, score_mask_index_reshape], axis=-1)
        score_mask_ = sparse_ops.sparse_to_dense(
            sparse_indices=score_mask_index_final,
            output_shape=[self.batch_size, self.alignments_size],
            sparse_values=True, default_value=False, validate_indices=False)
        score_mask_values_ = self._score_mask_value * array_ops.ones_like(mask_score)
        keywords_score = array_ops.where(score_mask_, mask_score, score_mask_values_)

        alignments = nn_ops.softmax(keywords_score)

        return alignments
