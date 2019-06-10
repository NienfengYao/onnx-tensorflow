from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import numpy as np
import tensorflow as tf
from onnx_tf.backend import run_node
from onnx_tf.common import supports_device
from onnx_tf.common.legacy import legacy_onnx_pre_ver, legacy_opset_pre_ver
from onnx import helper
from onnx import TensorProto
from onnx import defs


class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def _get_irnd(self, shape):
    return np.arange(np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def _elu(self, x):
    # f(x) = alpha * (exp(x) - 1.) for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return np.expm1(x)
    return x

  def _leaky_relu(self, x, alpha):
    # f(x) = alpha * x for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return alpha * x
    return x

  def _pooling(self, inputMap, poolSize=3, poolStride=2, mode='max'):
    """INPUTS:
      inputMap - input array of the pooling layer
      poolSize - X-size(equivalent to Y-size) of receptive field
      poolStride - the stride size between successive pooling squares

      OUTPUTS:
      outputMap - output array of the pooling layer

      Padding mode - 'edge'
    """

    # inputMap sizes
    in_batch, in_channel, in_row, in_col = np.shape(inputMap)

    # outputMap sizes
    out_row, out_col = int(np.floor(in_row/poolStride)), int(np.floor(in_col/poolStride))
    row_remainder, col_remainder = np.mod(in_row,poolStride), np.mod(in_col,poolStride)
    if row_remainder != 0:
      out_row +=1
    if col_remainder != 0:
      out_col +=1
    outputMap = np.zeros((in_batch, in_channel, out_row, out_col))

    for i in range(0, in_batch):
        for j in range(0, in_channel):
          temp_map = np.lib.pad(inputMap[i][j], ((0,poolSize-row_remainder),(0,poolSize-col_remainder)), 'edge')
          for r_idx in range(0,out_row):
            for c_idx in range(0,out_col):
              startX = c_idx * poolStride
              startY = r_idx * poolStride
              poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
              if mode == 'max':
              	poolOut = np.max(poolField)
              elif mode == 'average':
              	poolOut = np.average(poolField)
              else:
              	poolOut = np.min(poolField)
              outputMap[i, j, r_idx,c_idx] = poolOut
    return  outputMap

  def test_abs(self):
    node_def = helper.make_node("Abs", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.abs(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10, 1, 1])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.add(x, y.reshape([1, 10, 1, 1])))

#  def test_average_pool(self):
#    device = "CUDA"
#    if not supports_device(device):
#      raise unittest.SkipTest(
#          "Backend doesn't support device {}".format(device))
#    shape = [1, 1, 40, 40]
#    node_def = helper.make_node(
#        "AveragePool", ["X"], ["Y"],
#        kernel_shape=[2, 2],
#        pads=[1, 1],
#        strides=[1, 1])
#    x = self._get_rnd(shape)
#    output = run_node(node_def, [x], device=device)
#    test_output = np.zeros(shape)
#    for i1 in range(0, shape[0]):
#      for i2 in range(0, shape[1]):
#        for j1 in range(0, shape[2]):
#          for j2 in range(0, shape[3]):
#            test_output[i1][i2][j1][j2] = 0
#            count = 0
#            for k in range(j2, min(j2 + 2, shape[3])):
#              test_output[i1][i2][j1][j2] += x[i1][i2][j1][k]
#              count += 1
#            test_output[i1][i2][j1][j2] /= count
#    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_average_pool(self):
    shape = [1, 1, 5, 5]
    x = self._get_irnd(shape)
    # print(x.shape)
    # print(x)
    test_output = self._pooling(x, 2, 2, 'average')
    # print(test_output.shape)
    # print(test_output)
    return

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    if legacy_opset_pre_ver(6):
      raise unittest.SkipTest("Backend doesn't support consumed flag")
    node_def = helper.make_node(
        "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"],
        epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd(x_shape, 0, 1)
    m = self._get_rnd(param_shape, 0, 1)
    _m = m.reshape(_param_shape)
    v = self._get_rnd(param_shape, 0, 1)
    _v = v.reshape(_param_shape)
    scale = self._get_rnd(param_shape, 0, 1)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd(param_shape, 0, 1)
    _bias = bias.reshape(_param_shape)
    golden = self._batch_normalization(x, _m, _v, _bias, _scale, 0.001)
    output = run_node(node_def, [x, scale, bias, m, v])
    np.testing.assert_almost_equal(output["Y"], golden, decimal=5)

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in range(len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd(shape)
      x2 = self._get_rnd(shape)
      output = run_node(node_def, [x1, x2])
      np.testing.assert_almost_equal(output["Y"], np.concatenate((x1, x2),
                                                                 axis))
  def test_conv(self):
    device = "CUDA"
    if not supports_device(device):
      raise unittest.SkipTest(
          "Backend doesn't support device {}".format(device))

    N, C, H, W = 4, 3, 5, 5
    x_shape = [N, C, H, W]
    K, kH, kW = 6, 3, 3
    weight_shape = [K, C, kH, kW]
    node_def = helper.make_node(
        "Conv", ["X", "weights"], ["Y"],
        pads=[1, 1, 1, 1],
        kernel_shape=[kH, kW])

    x = self._get_rnd(x_shape)
    weights = self._get_rnd(weight_shape)
    output = run_node(node_def, [x, weights], device=device)

    out_shape = [N, K, H, W]
    test_output = np.zeros(out_shape)
    for n in range(N):
      for c in range(C):
        for h in range(H):
          for w in range(W):
            for k in range(K):
              for kh in range(kH):
                for kw in range(kW):
                  h_in_range = (h - kH // 2 + kh) < H and (
                      h - kH // 2 + kh) >= 0
                  w_in_range = (w - kW // 2 + kw) < W and (
                      w - kW // 2 + kw) >= 0
                  if h_in_range and w_in_range:
                    test_output[n][k][h][w] += (
                        x[n][c][h - kH // 2 + kh][w - kW // 2 + kw] *
                        weights[k][c][kh][kw])

    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_flatten(self):
    # If input tensor has shape (d_0, d_1, ... d_n) then the
    # output will have shape:
    #
    # (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
    #
    # TODO: pass axis attribute which is supported in newer
    # versions of onnx
    node_def = helper.make_node("Flatten", ["X"], ["Y"])
    x = self._get_rnd([10, 2, 3, 4, 5])
    output = run_node(node_def, [x])
    # TODO: pass axis=3 and uncomment the line below
    # np.testing.assert_almost_equal(output["Y"], x.reshape([60, 20]))
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 120]))

  def test_gemm(self):
    # Compute Y = alpha * A * B + beta * C
    node_def = helper.make_node(
        "Gemm", ["A", "B", "C"], ["Y"], transA=0, transB=0, alpha=1.0, beta=1.0)
    x = np.floor(self._get_rnd([10, 10]))
    y = np.floor(self._get_rnd([10, 10]))
    z = np.floor(self._get_rnd([10, 10]))
    output = run_node(node_def, [x, y, z])
    test_output = np.matmul(x, y) + z
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_average_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        sum = 0
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            sum += x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = sum / 6.
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_max_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalMaxPool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        max = x[i1][i2][0][0]
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            if max < x[i1][i2][j1][j2]:
              max = x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = max
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_l_r_n(self):
    # Each input value is divided by:
    #
    # (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    alpha = 2.0
    beta = 1.0
    bias = 5.0
    size = 3
    node_def = helper.make_node(
        "LRN", ["X"], ["Y"], alpha=alpha, beta=beta, bias=bias, size=size)
    x = self._get_rnd([10, 2, 10, 10])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 10, 2])
    x = np.transpose(x, axes=[0, 2, 3, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 10):
          for j2 in range(0, 2):
            sqr_sum = 0.
            # size of 3 means radius 1 in TF speak
            # i.e. the immediate neighbouring values
            # if "previous" neighbour exists
            if j2 > 0:
              sqr_sum += x[i1][i2][j1][j2 - 1] * x[i1][i2][j1][j2 - 1]
            # current value
            sqr_sum += x[i1][i2][j1][j2] * x[i1][i2][j1][j2]
            # if "next" neighbour exists
            if j2 < 2 - 1:
              sqr_sum += x[i1][i2][j1][j2 + 1] * x[i1][i2][j1][j2 + 1]
            test_output[i1][i2][j1][j2] = \
              x[i1][i2][j1][j2] / ((bias + (alpha * 1. / size) * sqr_sum) ** beta)
    test_output = np.transpose(test_output, axes=[0, 3, 1, 2])
    np.testing.assert_almost_equal(output["Y"], test_output)

#  def test_max_pool(self):
#    return
#    node_def = helper.make_node(
#        "MaxPool", ["X"], ["Y"],
#        dilations=[1, 1],
#        kernel_shape=[1, 2],
#        pads=[0, 0],
#        strides=[1, 2])
#    x = self._get_rnd([10, 10, 4, 4])
#    output = run_node(node_def, [x])
#    test_output = np.zeros([10, 10, 4, 2])
#    for i1 in range(0, 10):
#      for i2 in range(0, 10):
#        for j1 in range(0, 4):
#          for j2 in range(0, 2):
#            test_output[i1][i2][j1][j2] = \
#              max(x[i1][i2][j1][2*j2], x[i1][i2][j1][2*j2 + 1])
#    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_max_pool(self):
    shape = [1, 1, 5, 5]
    x = self._get_irnd(shape)
    # print(x.shape)
    # print(x)
    test_output = self._pooling(x, 2, 2, 'max')
    # print(test_output.shape)
    # print(test_output)
    return

  def test_mul(self):
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"])
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10, 1, 1])
    output = run_node(node_def, [x, y])
    # output["z"].shape = (5, 10, 5, 5)
    np.testing.assert_almost_equal(output["Z"],
                                   np.multiply(x, y.reshape([1, 10, 1, 1])))

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_reshape(self):
    x = self._get_rnd(100)
    shape = [10, 10]
    if defs.onnx_opset_version() < 5:
      node_def = helper.make_node("Reshape", ["X"], ["Z"], shape=shape)
      output = run_node(node_def, [x])
    else:
      node_def = helper.make_node("Reshape", ["X", "Y"], ["Z"])
      output = run_node(node_def, [x, shape])

    np.testing.assert_almost_equal(output["Z"], x.reshape([10, 10]))

  def test_sum(self):
    node_def = helper.make_node("Sum", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4])
    test_output = x1 + x2 + x3 + x4
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

  def test_matmul(self):
    node_def = helper.make_node("MatMul", ["X", "Y"], ["Z"])
    # 2d 
    x = self._get_rnd([2, 3, 4])
    y = self._get_rnd([2, 4, 3])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.matmul(x, y))

    # 3d 
    x = self._get_rnd([2, 3, 4])
    y = self._get_rnd([2, 4, 3])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.matmul(x, y))

    # 4d 
    x = self._get_rnd([1, 2, 3, 4])
    y = self._get_rnd([1, 2, 4, 3])
    output = run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.matmul(x, y))

  def test_softmax(self):
    node_def = helper.make_node("Softmax", ["X"], ["Y"])
    x = np.array([[-1, 0, 1]]).astype(np.float32)
    # expected output [[0.09003058, 0.24472848, 0.66524094]]
    y = np.exp(x) / np.sum(np.exp(x), axis=1)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],  y)

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2])
    x = np.array([[[0], [1], [2]]])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.squeeze(x, axis=2))

  def test_unsqueeze(self):
    node_def = helper.make_node("Unsqueeze", ["X"], ["Y"], axes=[0])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.expand_dims(x, axis=0)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], y)

if __name__ == '__main__':
  unittest.main()
