from Models import tf, K, keras
import Models

from keras.engine import Layer, InputSpec
from keras.utils import Sequence, conv_utils
from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Lambda, Input
from kapre.time_frequency import Spectrogram

import numpy as np
import librosa




class Conv1D_tied(Layer):
    # deconv layer - uses kernel from tied_to, transposes it and performs convolution. not trainable.
        
    def __init__(self, filters,
                 kernel_size,
                 tied_to,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 input_length=None,
                 data_format='channels_last',
                 **kwargs):

        if padding not in {'valid', 'same', 'causal'}:
            raise Exception('Invalid padding mode for Convolution1D:', padding)
        
        super(Conv1D_tied, self).__init__(**kwargs)
        
        self.rank = 1
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.common.normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.tied_to = tied_to
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        
    def build(self, input_shape):
        
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
           
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
                
                
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (input_dim, self.filters)

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
                self.trainable_weights = [self.bias]
            else:
                self.bias = None
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
            self.built = True        

            
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)
    
    
            
    def call(self, x, mask=None):
        
        x = K.expand_dims(x, -1)  # add a dimension of the right
        x = K.permute_dimensions(x, (0, 3, 1, 2))

        W = self.tied_to.kernel   
        W = K.expand_dims(W, -1)
        W = tf.transpose(W, (1, 0, 2, 3))
        
        output = K.conv2d(x, W, 
                          strides=(self.strides,self.strides),
                          padding=self.padding,
                          data_format=self.data_format)
        if self.bias:
            output += K.reshape(self.bias, (1, self.filters, 1, 1))
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (0, 2, 1))
        output = self.activation(output)
        return output


    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'tied_to': self.tied_to.name
        }
        base_config = super(Conv1D_tied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    
class Conv1D_local(Layer):

    # Locally-connected 1D convolutional layer. Performs one-to-one convolutions to input feature map.
        
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 input_length=None,
                 **kwargs):

        if padding not in {'valid', 'same', 'causal'}:
            raise Exception('Invalid padding mode for Convolution1D:', padding)
        
        super(Conv1D_local, self).__init__(**kwargs)
        
        self.rank = 1
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        
        

        
    def build(self, input_shape):
        
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
                
                
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (1, self.filters)
            
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            # Set input spec.
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
            self.built = True        
       

    def compute_output_shape(self, input_shape):
        
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
                
        return (input_shape[0], input_shape[1], self.filters) 
    
    def call(self, inputs):
        

        x = tf.split(inputs, self.filters, axis = 2)
        W = tf.split(self.kernel, self.filters, axis = 2)
        outputs = []
        
        for i in range(self.filters):
            output = K.conv1d(x[i], W[i],
                              strides=self.strides[0],
                              padding=self.padding,
                              data_format=self.data_format,
                              dilation_rate=self.dilation_rate[0])
    
    
            outputs.append(output)
    
        outputs = K.concatenate(outputs,axis=-1)
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs


    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_local, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    
class SAAF(Layer):
    '''[references]
    [1] ConvNets with Smooth Adaptive Activation Functions for Regression, Hou, Le
    and Samaras, Dimitris and Kurc, Tahsin M and Gao, Yi and Saltz, Joel H,
    Artificial Intelligence and Statistics, 2017
     '''
    def __init__(self,
                 break_points,
                 break_range = 0.2,
                 magnitude = 1.0,
                 order = 2,
                 tied_feamap = True,
                 kernel_initializer = 'random_normal',
                 kernel_regularizer = None,
                 kernel_constraint = None,
                 **kwargs):
        super(SAAF, self).__init__(**kwargs)
        self.break_range = break_range
        self.break_points = list(np.linspace(-self.break_range, self.break_range, break_points, dtype=np.float32))
        self.num_segs = int(len(self.break_points) / 2)
        self.magnitude = float(magnitude)
        self.order = order
        self.tied_feamap = tied_feamap
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        
        

    def build(self, input_shape):

        if self.tied_feamap:
            kernel_dim = (self.num_segs + 1, input_shape[2])
        else:
            kernel_dim = (self.num_segs + 1,) + input_shape[2::]
            
        self.kernel = self.add_weight(shape=kernel_dim,
                                     name='kernel',
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        
    def basisf(self, x, s, e):
        cpstart = tf.cast(tf.less_equal(s, x), tf.float32)
        cpend = tf.cast(tf.greater(e, x), tf.float32)
        if self.order == 1:
            output = self.magnitude * (0 * (1 - cpstart) + (x - s) * cpstart * cpend + (e - s) * (1 - cpend))
        else:
            output = self.magnitude * (0 * (1 - cpstart) + 0.5 * (x - s)**2 * cpstart
                                     * cpend + ((e - s) * (x - e) + 0.5 * (e - s)**2) * (1 - cpend))
        
        return tf.cast(output, tf.float32)
        
        self.built = True

    def call(self, x):
         
        output = tf.zeros_like(x)
        
        if self.tied_feamap:
            output += tf.multiply(x,self.kernel[-1])
        else:
            output += tf.multiply(x,self.kernel[-1])
        for seg in range(0, self.num_segs):
            if self.tied_feamap:
                output += tf.multiply(self.basisf(x, self.break_points[seg * 2], self.break_points[seg * 2 + 1]), 
                                      self.kernel[seg])
            else:
                output += tf.multiply(self.basisf(x, self.break_points[seg * 2], self.break_points[seg * 2 + 1]), 
                                      self.kernel[seg])

        return output

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'break_points': self.break_points,
            'magnitude': self.magnitude,
            'order': self.order,
            'tied_feamap': self.tied_feamap

        }
        base_config = super(SAAF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    
    
class Conv1D_localTensor(Layer):

        
    def __init__(self, filters,
                 kernel_size,
                 batch,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 input_length=None,
                 **kwargs):

        if padding not in {'valid', 'same', 'causal'}:
            raise Exception('Invalid padding mode for Convolution1D:', padding)
        
        super(Conv1D_localTensor, self).__init__(**kwargs)
        
        self.rank = 1
        self.filters = filters
        self.batch = batch
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        
        

        
    def build(self, input_shape):
        
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
                
                
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (1, self.filters)

            self.bias = None
            self.kernel = None
            # Set input spec.
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
            self.built = True        
            
            
        
        
        
        

    def compute_output_shape(self, input_shape):
        
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)

        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
                
        # Remove this last return. 
        return (input_shape[0], input_shape[-2], self.filters) 
    
    def call(self, inputs):
        

        x = tf.split(inputs, 2, axis = 2)
    
        y = tf.split(x[0], self.filters, axis = -1)
        w = tf.split(x[1], self.filters, axis = -1)

        
        outputs = []
        for j in range(self.batch):
            outputs_ = []
            for i in range(self.filters):
                
                X = y[i][j]
                X = tf.reshape(X,(1,X.shape[0],1))
                W = w[i][j][:]
                W = tf.reshape(W,(W.shape[0],1,1))

                output = K.conv1d(X, W,
                                  strides=self.strides[0],
                                  padding=self.padding,
                                  data_format=self.data_format,
                                  dilation_rate=self.dilation_rate[0])

                
                outputs_.append(output)
       
    
            outputs_ = K.concatenate(outputs_,axis=-1)

            outputs.append(outputs_)
        outputs = K.concatenate(outputs,axis=0)

        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs


    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_localTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    

    
    
class VelvetNoise(Layer):

        
    def __init__(self, pulse_sec,
                 batch_size,
                 input_dim=None,
                 input_length=None,
                 **kwargs):
        
        super(VelvetNoise, self).__init__(**kwargs)
        
        self.batch_size = batch_size
        self.pulse_sec = pulse_sec
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            

        
    def build(self, input_shape):
        
        N = self.input_length
        ps = self.pulse_sec
        self.t = tf.range(0,N-1, delta=N/ps, dtype=tf.float32)
        self.t = tf.reshape(self.t, [self.t.shape[0],1])
        self.t = tf.broadcast_to(self.t, [self.t.shape[0],1])
        self.t = tf.cast(self.t, tf.int64)
        
        self.built = True   

            
    def compute_output_shape(self, input_shape):

        return (input_shape[0],) + (self.input_length,) + (self.input_dim,)
    
            
    def call(self, inputs):
                 
        N = self.input_length
        ps = self.pulse_sec
        
        
        
        x = tf.split(inputs, 2, axis = 2)
        sgn = tf.split(x[0], self.input_dim, axis = -1)
        idxs = tf.split(x[1], self.input_dim, axis = -1)
        t = self.t
   
        output = []
       
        for j in range(self.batch_size):
        
            output_ = []
            for i in range(self.input_dim):
                
                s = sgn[i][j][:,0]

                s = tf.reshape(s, [-1])
                
                idx = idxs[i][j]
            
                idx = (N/ps-1)*idx #
               
                idx = castIntSoftMax(idx)
                
                idx = t + idx   
                
                zeros = tf.zeros((ps, 1),dtype=tf.int64)
                
                idx = tf.concat([idx, zeros], axis=-1)
                
                sparse_tensor = tf.SparseTensor(values=s,indices=idx, dense_shape=[N, 1] )

                sparse_tensor = tf.sparse_add(tf.zeros([N, 1]), sparse_tensor)
                output_.append(sparse_tensor)
                
            output.append(K.concatenate(output_, axis=1))
 
        output = tf.stack(output)
        return output


    
    def get_config(self):
        config = {
            'batch_size': self.batch_size,
            'pulse_sec': self.pulse_sec
        }
        base_config = super(VelvetNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

    
    
    
@tf.custom_gradient
def castIntSoftMax(x):
    def grad(dy):
        return dy
    return tf.cast(x,tf.int64), grad
    
        
#Generators for training. dry and wet tensors should be of tensor shape (number_of_recordings, number_of_samples, 1) 

# Generator for pretraining.

class Generator(Sequence):

    def __init__(self, x_set, y_set, win_length, hop_length, win = False):
        self.x, self.y = x_set, y_set
        self.win_length = win_length
        self.hop_length = hop_length
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.win = win

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = np.zeros((self.batch_size, self.win_length, 1))
        
        batch_y = np.zeros((self.batch_size, self.win_length, 1))
        
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        
        x_w = slicing(x_w, self.win_length, self.hop_length, windowing = self.win)
        y_w = slicing(y_w, self.win_length, self.hop_length, windowing = self.win)
        
        for i in range(self.batch_size):
            
            batch_x[i] = x_w[i].reshape(self.win_length,1)  
            batch_y[i] = y_w[i].reshape(self.win_length,1) 

            
        return batch_x, batch_y
   
    

# Generator for model_1, model_2. Audio samples should be already zero padded at the end (0.5seconds.)    


class GeneratorContext(Sequence):

    def __init__(self, x_set, y_set, context, win_length, hop_length, win = False, win_input = None):
        self.x, self.y = x_set, y_set
        self.win_length = win_length
        self.hop_length = hop_length
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.win_output = win
        if win_input == None:
            self.win_input = win
        else:
            self.win_input = win_input
        self.context = context

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = []
        for i in range(self.context*2+1):
            batch_x.append(np.zeros((self.batch_size, self.win_length, 1)))
        batch_y = np.zeros((self.batch_size, self.win_length, 1))
        
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        
        x_w = slicing(x_w, self.win_length, self.hop_length, windowing = self.win_input)

        x_w = np.pad(x_w, ((self.context, self.context),(0, 0)), 'constant', constant_values=(0))
        a = []
        for i in range(x_w.shape[0]):
            a.append(x_w[i:i+self.context*2+1])
        del a[-self.context*2:]
        a = np.asarray(a)
       
        y_w = slicing(y_w, self.win_length, self.hop_length, windowing = self.win_output)
        
        for i in range(self.batch_size):
            
            for j in range(self.context*2+1):
                batch_x[j][i] = a[:,j,:][i].reshape(self.win_length,1)
                       
            batch_y[i] = y_w[i].reshape(self.win_length,1) 
            
        batch_x = np.swapaxes(np.asarray(batch_x), 0, 1)
        
        return batch_x, batch_y  
    

def preProcessingSample(win_length, window = False, preEmphasis = False,
                        spec = False, n_fft = None, n_hop = None, log = False, power = 1.0):

    x = Input(shape=(win_length, 1))
    y = x
    if window:
        y = Lambda((Models.Window), name='output')(y)     
    if preEmphasis:
        y = Lambda((Models.PreEmphasis), name='preEmph')(y)
    if spec:
        spectrogram = Spectrogram(n_dft=n_fft, n_hop=n_hop, input_shape=(1, win_length), 
              return_decibel_spectrogram=log, power_spectrogram=power, 
              trainable_kernel=False, name='spec')
        
        y_D = Lambda((Models.toPermuteDimensions), name='perm_mel')(y)
        y_D = spectrogram(y_D)
        model = Model(inputs=[x], outputs=[y_D, y])
    else:
        model = Model(inputs=[x], outputs=[y])
    
    return model

def preProcessingData(X, win_length, hop,
                      window = False, preEmphasis = False,
                      spec = False, n_fft = None, n_hop = None, log = False, power = 1.0 ):
    
    model = preProcessingSample(win_length, window = window, preEmphasis = preEmphasis,
                        spec = spec, n_fft = n_fft, n_hop = n_hop, log = log, power = power)
    
    output = []
    for i in range(len(X)):
        
        x = np.expand_dims(X[i],0)
        x_data_gen = GeneratorContext(x, x, 1, win_length, hop)
        x = x_data_gen[0][1]
        output_ = model.predict(x)
        output.append(output_)
    del model
    
    x1 = []
    x2 = []
    x3 = []
    for i in range(len(X)):
        x1.append(output[i][0])
        x2.append(output[i][1])
        x3.append(output[i])
        
    if spec:
        return np.asarray(x1), np.asarray(x2)
    else:
        return np.asarray(x3)



def slicing(x, win_length, hop_length, center = True, windowing = True):
    # Pad the time series so that frames are centered
    if center:
        x = np.pad(x, int(win_length // 2), mode='constant')
    # Window the time series.
    y_frames = librosa.util.frame(x, frame_length=win_length, hop_length=hop_length)

    f = []
    for i in range(len(y_frames.T)):
        f_ = y_frames.T[i]
        if windowing:
            y = Window(tf.convert_to_tensor(np.float32(np.reshape(f_,(1,f_.shape[0],1)))))
            with tf.Session() as sess:
                sess.run([y])
                y = y.eval()
                sess.close()
            f_ = y.reshape(f_.shape[0])
               
        f.append(f_)
    return np.float32(np.asarray(f)) 

class GeneratorContextSpec(Sequence):

    def __init__(self, x_set, y_set,
                 context, win_length, hop_length,
                 n_dft, n_hop, n_mels = None, win = False, win_input = None):
        
        self.x, self.y_time, self.y_d = x_set, y_set[1], y_set[0]
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_dft = n_dft
        self.n_hop = n_hop
        self.n_mels = n_mels
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.n_frame = int(self.win_length / self.n_hop)
        self.win = win
        if win_input == None:
            self.win_input = win
        else:
            self.win_input = win_input
        self.context = context

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = []
        for i in range(self.context*2+1):
            batch_x.append(np.zeros((self.batch_size, self.win_length, 1)))
        
        batch_y_time = np.zeros((self.batch_size, self.win_length, 1))
        
        if self.n_mels is None:
            batch_y_d = np.zeros((self.batch_size, self.n_dft//2+1, self.n_frame, 1))
        else:
            batch_y_d = np.zeros((self.batch_size, self.n_mels, self.n_frame, 1))
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y_time[idx]
        y_d = self.y_d[idx]

        
        x_w = slicing(x_w, self.win_length, self.hop_length, windowing = self.win_input)

        x_w = np.pad(x_w, ((self.context, self.context),(0, 0)), 'constant', constant_values=(0))
        a = []
        for i in range(x_w.shape[0]):
            a.append(x_w[i:i+self.context*2+1])
        del a[-self.context*2:]
        a = np.asarray(a)
   
        for i in range(self.batch_size):
            
            for j in range(self.context*2+1):
                batch_x[j][i] = a[:,j,:][i].reshape(self.win_length,1)
            
            batch_y_time[i] = y_w[i]
            
            batch_y_d[i] = y_d[i]
       
        return np.swapaxes(np.asarray(batch_x), 0, 1), [batch_y_d, batch_y_time] 