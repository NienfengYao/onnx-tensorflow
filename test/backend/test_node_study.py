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


if __name__ == '__main__':
  unittest.main()
