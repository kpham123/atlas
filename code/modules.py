import numpy as np
import tensorflow as tf


class NeuralNetwork(object):
  def conv2d(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv2d(input, W, padding="SAME", strides=strides)
      out = tf.nn.bias_add(out, b)
      return out

  def conv2d_relu(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv2d(input, W, padding="SAME", strides=strides)
      out = tf.nn.bias_add(out, b)
      out = tf.nn.relu(out, name="out")
      return out

  def maxpool2d(self, input, scope_name, pool_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(scope_name):
      out = tf.nn.max_pool(input,
                           ksize=pool_shape,
                           name="out",
                           padding="SAME",
                           strides=strides)
      return out

  def dropout(self, input, keep_prob, scope_name):
    with tf.variable_scope(scope_name):
      out = tf.nn.dropout(input, keep_prob, name="out")
      return out

  def fc(self, input, output_shape, scope_name):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      input_shape = input.shape[1:]
      input_shape = int(np.prod(input_shape))
      W = tf.get_variable(name="W",
                          shape=[input_shape, output_shape],
                          initializer=xavier_initializer(uniform=False))
      b = tf.get_variable(name="b",
                          shape=[output_shape],
                          initializer=xavier_initializer(uniform=False))
      input = tf.reshape(input, [-1, input_shape])
      # out = tf.nn.relu(tf.add(tf.matmul(input, W), b), name="out")
      out = tf.add(tf.matmul(input, W), b, name="out")
      return out

  def deconv2d(self, input, filter_shape, num_outputs, scope_name, strides=[1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    xavier_initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d
    with tf.variable_scope(scope_name):
      out = tf.contrib.layers.conv2d_transpose(input,
                                               # activation_fn=tf.nn.relu,
                                               activation_fn=None,
                                               biases_initializer=xavier_initializer(uniform=False),
                                               kernel_size=filter_shape,
                                               num_outputs=num_outputs,
                                               padding="SAME",
                                               stride=strides,
                                               weights_initializer=xavier_initializer_conv2d(uniform=False))
      out = tf.identity(out, name="out")
      return out

  def upsample(self, input, scope_name, factor=[2, 2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(scope_name):
      out = tf.image.resize_bilinear(input, size=size, align_corners=None, name="out")
      return out


class ConvEncoder(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, scope_name="encoder"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 8], scope_name="conv1")  # (232, 196, 8)
      pool1 = self.maxpool2d(conv1, scope_name="pool1")  # (116, 98, 8)
      drop1 = self.dropout(pool1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d_relu(drop1, filter_shape=[5, 5, 8, 16], scope_name="conv2")  # (116, 98, 16)
      pool2 = self.maxpool2d(conv2, scope_name="pool2")  # (58, 49, 16)
      drop2 = self.dropout(pool2, keep_prob=self.keep_prob, scope_name="drop2")
      drop2 = tf.reshape(drop2, shape=[-1, 58*49*16])  # (45472,)
      fc1 = self.fc(drop2, output_shape=1024, scope_name="fc1")
      drop3 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop3")
      fc2 = self.fc(drop3, output_shape=256, scope_name="fc2")
      out = tf.identity(fc2, name="out")

    return out


class DeconvDecoder(NeuralNetwork):
  def __init__(self, keep_prob, output_shape, scope_name="decoder"):
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      fc1 = self.fc(input, output_shape=1024, scope_name="fc1")
      drop1 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop1")
      fc2 = self.fc(drop1, output_shape=58*49*16, scope_name="fc2")
      drop2 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop2")
      drop2 = tf.reshape(drop2, shape=[-1, 58, 49, 16])
      up1 = self.upsample(drop2, scope_name="up1", factor=[2, 2])  # (116, 98, 16)
      deconv1 = self.deconv2d(up1, filter_shape=[5, 5], num_outputs=8, scope_name="deconv1")  # (116, 98, 8)
      up2 = self.upsample(deconv1, scope_name="up2", factor=[2, 2])
      deconv2 = self.deconv2d(up2, filter_shape=[3, 3], num_outputs=1, scope_name="deconv2")  # (232, 196, 1)
      out = tf.identity(deconv2, name="out")

    return out


class UNet(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      # Conv
      conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 64], scope_name="conv1")  # (b, 232, 196, 64)
      drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, 64, 64], scope_name="conv2")  # (b, 232, 196, 64)
      drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

      pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, 116, 98, 64)
      conv3 = self.conv2d_relu(pool1, filter_shape=[3, 3, 64, 128], scope_name="conv3")  # (b, 116, 98, 128)
      drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
      conv4 = self.conv2d_relu(drop3, filter_shape=[3, 3, 128, 128], scope_name="conv4")  # (b, 116, 98, 128)
      drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

      pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, 58, 49, 128)
      conv5 = self.conv2d_relu(pool2, filter_shape=[3, 3, 128, 256], scope_name="conv5")  # (b, 58, 49, 256)
      drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
      conv6 = self.conv2d_relu(drop5, filter_shape=[3, 3, 256, 256], scope_name="conv6")  # (b, 58, 49, 256)
      drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

      # Deconv
      up1 = self.upsample(drop6, scope_name="up1", factor=[2, 2])  # (b, 116, 98, 256)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=128, scope_name="deconv1")  # (b, 116, 98, 128)
      concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, 116, 98, 256)
      conv7 = self.conv2d_relu(concat1, filter_shape=[3, 3, 256, 128], scope_name="conv7")  # (b, 116, 98, 128)
      drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
      conv8 = self.conv2d_relu(drop7, filter_shape=[3, 3, 128, 128], scope_name="conv8")  # (b, 116, 98, 128)
      drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

      up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, 232, 196, 128)
      deconv2 = self.deconv2d(up2, filter_shape=[2, 2], num_outputs=64, scope_name="deconv2")  # (b, 232, 196, 64)
      concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, 232, 196, 128)
      conv9 = self.conv2d_relu(concat2, filter_shape=[3, 3, 128, 64], scope_name="conv9")  # (b, 232, 196, 64)
      drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
      conv10 = self.conv2d_relu(drop9, filter_shape=[3, 3, 64, 64], scope_name="conv10")  # (b, 232, 196, 64)
      drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

      conv11 = self.conv2d(drop10, filter_shape=[1, 1, 64, 1], scope_name="conv11")  # (b, 232, 196, 1)
      out = tf.identity(conv11, name="out")
    return out

class Planar2DConvLSTMCell(tf.contrib.rnn.Conv2DLSTMCell):
  """ Extension of convolutional LSTM recurrent network cell to accept multiple inputs (9 from the previous plane). 

      Modified from https://github.com/tensorflow/tensorflow/pull/8891/files
  """

  def __init__(self,activation=tf.tanh,*args,**kwargs):
    super(Planar2DConvLSTMCell,self).__init__(*args,**kwargs)
    self._activation = activation

  def call(self,inputs,state,scope=None):
    """ Arrangement as follows, where the output cell would be positioned at c22 and shifted by one. 
        c11 c12 c13
        c21 c22 c23
        c31 c32 c33
    """
    c11,c12,c13,c21,c22,c23,c31,c32,c33,h11,h12,h13,h21,h22,h23,h31,h32,h33 = state
    new_hidden = tf.nn.rnn_cell._conv([inputs,h11,h12,h13,h21,h22,h23,h31,h32,h33],
                       self._kernel_shape,
                       (4+8)*self._output_channels,
                       self._use_bias)
    gates = tf.contrib.rnn.python.ops.array_ops.split(value=new_hidden,
                                          num_or_size_splits=4+8,
                                          axis=self._conv_ndims+1)

    input_gate,new_input,f11,f12,f13,f21,f22,f23,f31,f32,f33,output_gate = gates
    new_cell = (tf.sigmoid(f11 + self._forget_bias) * c11) + \
               (tf.sigmoid(f12 + self._forget_bias) * c12) + \
               (tf.sigmoid(f13 + self._forget_bias) * c13) + \
               (tf.sigmoid(f21 + self._forget_bias) * c21) + \
               (tf.sigmoid(f22 + self._forget_bias) * c22) + \
               (tf.sigmoid(f23 + self._forget_bias) * c23) + \
               (tf.sigmoid(f31 + self._forget_bias) * c31) + \
               (tf.sigmoid(f32 + self._forget_bias) * c32) + \
               (tf.sigmoid(f33 + self._forget_bias) * c33) + \
               (tf.sigmoid(input_gate) * self._activation(new_input))
    output = self._activation(new_cell) * tf.sigmoid(output_gate)

    if self._skip_connection:
      output = tf.contrib.rnn.python.ops.array_ops.concat([output, inputs], axis=-1)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_cell, output)
      return output, new_state

class PyramidLSTM(NeuralNetwork):
  def __init__(self,input_shape,keep_prob,mode,scope_name="p"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.mode = mode
    self.scope_name = scope_name

  def build_graph(self,input):
    """
    Naming convention is 0,0 for top left

    Note that input data is (batch,h,w,d,1)

    d1: (batch,h,w,d)
    d2:-(batch,h,w,d)
    d3: (batch,w,d,h)
    d4:-(batch,w,d,h)
    d5: (batch,d,h,w)
    d6:-(batch,d,h,w)
    """

    # Padding unnecessary since conv2d has padding=same under the hood
    # Note if this is uncommented, need to adjust axis in each direction
    # Pad the input with zeros all around; should be n-1 for n-D
    # padSize = 2
    # paddings = tf.constant([[0,0,],[padSize,padSize,],[padSize,padSize,],[padSize,padSize,]])
    # padded_input = tf.pad(input,paddings,"CONSTANT")
    padded_input = input

    # Padding 1: If the dimension is even, add another row to the end so that pyramid lstm ends up with 1 pixel's result on each pyramid
    isEven = np.array(padded_input.get_shape().as_list()[1:]) % 2 == 0
    isEven = isEven.astype(int)
    paddings = tf.constant([[0,0,],[0,isEven[0],],[0,isEven[1],],[0,isEven[2],],[0,0]]) # Extra dimension due to expand_dims
    padded_input = tf.pad(padded_input,paddings,"CONSTANT")
    self.padded_input_shape = padded_input.get_shape().as_list()[1:4]

    # Padding 2: Pad so all dimensions are equal to allow adding the results from each pyramid
    maxSize = max(self.padded_input_shape)
    paddings = tf.constant([[0,0,],
                            [int((maxSize-self.padded_input_shape[0])/2),int((maxSize-self.padded_input_shape[0])/2),],
                            [int((maxSize-self.padded_input_shape[1])/2),int((maxSize-self.padded_input_shape[1])/2),],
                            [int((maxSize-self.padded_input_shape[2])/2),int((maxSize-self.padded_input_shape[2])/2),],
                            [0,0,]])
    padded_input = tf.pad(padded_input,paddings,"CONSTANT")
    self.padded_input_shape = padded_input.get_shape().as_list()[1:4]

    filter_size = [7,7]
    hidden_units_per_pixel = 8

    # finalCellCount should be h*w*c from self.input_shape
    # finalCellCount = np.prod(np.array(self.padded_input_shape)) # finalCount should be h*w*3

    def pyramid(inputTensor,directionName,ordering,reverse=False):
      """ 
      Note 1: inputTensor is [batch_size,a,b,c,1]
      Note 2: ordering is an anagram list of [1,2,3]
      """

      # Reorder based on the 6 pyramids and add reversal if needed
      p_input = tf.transpose(inputTensor,[0,*ordering,4])
      _,time_index_length,dim1,dim2,_ = p_input.get_shape().as_list()
      if not reverse:
        p_input = p_input[:,0:int((time_index_length+1)/2),:,:,:]
      elif reverse:
        p_input = tf.reverse(p_input[:,int((time_index_length-1)/2):,:,:,:],axis=[2])

      # Confirm that each run of pyramid becomes 117
      # print(p_input.get_shape().as_list()[1])
      assert(p_input.get_shape().as_list()[1] == 117)

      # Initialize a cell (i.e. tensorflow layer) for every plane in the pyramid
      cell_input_shape = [dim1,dim2,1]
      cells = []
      scope_name = '{}_{}'.format(self.scope_name,directionName)
      for t in range(0,int((time_index_length+1)/2)):
        with tf.variable_scope('{}_t{}'.format(scope_name,t)):
          cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=cell_input_shape,
                                              kernel_shape=filter_size,
                                              output_channels=hidden_units_per_pixel)
        cells.append(cell)

      # Connect layers together
      with tf.variable_scope(scope_name):
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs,_ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,inputs=p_input,dtype=tf.float32)
        # final_state_tuple = multi_state_outputs[-1]
        # result = final_state_tuple[1] # e.g. 197x189x4 or 233x233x4
        # results = [multi_state_output[-1][1] for multi_state_output in multi_state_outputs]
        # results = tf.stack(results,axis=1)
        results = outputs
        print('results.get_shape().as_list():',results.get_shape().as_list())
      return results

    def conv3d(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1], padding="SAME"):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv3d(input, W, padding=padding, strides=strides)
      out = tf.nn.bias_add(out, b)
      out = tf.nn.relu(out, name="out")
      return out

    def conv3d_relu(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1], padding="SAME"):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv3d(input, W, padding=padding, strides=strides)
      out = tf.nn.bias_add(out, b)
      out = tf.nn.relu(out, name="out")
      return out

    # Initialize each pyramid
    clstm1 = pyramid(padded_input,'d1',[1,2,3],reverse=False) # (b,117,233,233,8)
    clstm2 = pyramid(padded_input,'d2',[1,2,3],reverse=True)  # (b,117,233,233,8)
    clstm3 = pyramid(padded_input,'d3',[2,3,1],reverse=False) # (b,117,233,233,8)
    clstm4 = pyramid(padded_input,'d4',[2,3,1],reverse=True)  # (b,117,233,233,8)
    clstm5 = pyramid(padded_input,'d5',[3,1,2],reverse=False) # (b,117,233,233,8)
    clstm6 = pyramid(padded_input,'d6',[3,1,2],reverse=True)  # (b,117,233,233,8)

    # Convert rectangular prisms to pyramids and stitch together into (b,233,233,233,8)
    #  Pad back to cubic
    x = tf.pad(x,[[0,0],[0,0],[0,0],[0,2]])

    def aggregate(x):
        # Clear pyramid along initial dimension 0
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[2])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[2])

        # Clear pyramid along initial dimension 1
        x = tf.transpose(x,[1,0,2])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[2])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[2])
        x = tf.transpose(x,[1,0,2])

        # Rotate all pyramids back to their orientation in the 3-d volume
        x2 = tf.reverse(x,axis=[2])
        x3 = tf.transpose(x,[2,0,1])
        x4 = tf.reverse(x3,axis=[0])
        x5 = tf.transpose(x,[1,2,0])
        x6 = tf.reverse(x5,axis=[1])
        z = x + x2 + x3 + x4 + x5 + x6
        return z

    result = tf.add_n([clstm1,clstm2,clstm3,clstm4,clstm5,clstm6]) # (b,117,233,233,4)

    conv1 = self.conv2d_relu(result, filter_shape=[3,3,3,] + [hidden_units_per_pixel,hidden_units_per_pixel], scope_name="conv1")  # (b, 117, 233, 233, 8)
    drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
    conv2 = self.conv2d(drop1, filter_shape=[1, 1, 1, hidden_units_per_pixel, 1], scope_name="conv2")  # (b, 117, 233, 233, 1)

    # Option 1: Remove introduced padding
    # Option 2: Convolv

    # # d1 will travel along h since inputs should be [batch_size, max_time, cell_state_size]
    # direction = 'd1'
    # d1_input = tf.transpose(padded_input,[0,1,2,3,4])
    # assert((time_index_length==233) and (dim1==197) and (dim2==189))
    # d1_cell_input_shape = [dim1,dim2,1]
    # scope_name = '{}_{}'.format(self.scope_name,direction)
    # d1_cells = []
    # for t in range(0,int((time_index_length-1)/2)):
    #   with tf.variable_scope('{}_c{}_{}'.format(scope_name,t,direction)):
    #     cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=d1_cell_input_shape,
    #                                         kernel_shape=filter_size,
    #                                         output_channels=hidden_units_per_pixel)
    #   d1_cells.append(cell)
    # d1_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(d1_cells)
    # outputs,multi_state_outputs = tf.nn.dynamic_rnn(cell=d1_multi_rnn_cell,inputs=d1_input,dtype=tf.float32)
    # final_state_tuple = multi_state_outputs[-1]
    # d1_result = final_state_tuple[1] # e.g. 197x189x4
    # # print('clstm_result.get_shape().as_list():',clstm_result.get_shape().as_list())

    # direction = 'd2'
    # d2_input = tf.reverse(d1_input[:,int((time_index_length-1)/2):,:,:,:],axis=1)
    # assert((time_index_length==233) and (dim1==197) and (dim2==189))
    # d2_cell_input_shape = [dim1,dim2,1]
    # direction = 'd2'
    # scope_name = '{}_{}'.format(self.scope_name,direction)
    # d2_cells = []
    # for t in range(0,int((time_index_length-1)/2)):
    #   with tf.variable_scope('{}_c{}_{}'.format(scope_name,t,direction)):
    #     cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=d2_cell_input_shape,
    #                                         kernel_shape=filter_size,
    #                                         output_channels=hidden_units_per_pixel)
    #   d2_cells.append(cell)
    # d2_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(d2_cells)
    # outputs,multi_state_outputs = tf.nn.dynamic_rnn(cell=d2_multi_rnn_cell,inputs=d2_input,dtype=tf.float32)
    # final_state_tuple = multi_state_outputs[-1]
    # d2_result = final_state_tuple[1]
    # # print('clstm_result.get_shape().as_list():',clstm_result.get_shape().as_list())

    exit();exit();exit()

    direction = 'd3'
    direction = 'd4'
    direction = 'd5'
    direction = 'd6'

      # with tf.variable_scope(scope_name):

    # with tf.variable_scope(self.scope_name):
    #   cells = []
    #   cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=self.input_shape,kernel_shape=[3, 3],output_channels=8)
    #   (outputs,state) = tf.nn.dynamic_rnn(cell,x_image,time_major=False,dtype=tf.float32)
