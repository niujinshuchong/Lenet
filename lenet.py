import numpy as np
import cPickle, gzip
import cv2

class InnerProduct:
    def __init__(self, input_num, batch_size, output_num):
        self.input_num = input_num
        self.batch_size = batch_size
        self.output_num = output_num
        self.W = np.random.normal(0, 1, (output_num, input_num)) * 0.01
        self.b = np.zeros((output_num, 1))
        self.dW = np.zeros((output_num, input_num))
        self.db = np.zeros((output_num, 1))
        self.WX_plus_b = np.zeros((output_num, batch_size))
        self.dX = np.zeros((input_num, batch_size))
        self.X = np.zeros((input_num, batch_size))
        
    def forward(self, X):
        self.X[...] = X[...]
        self.WX_plus_b = np.dot(self.W, X) + self.b
        return self.WX_plus_b
        
    def backward(self, dWX_plus_b):
        self.dX = np.dot(self.W.T, dWX_plus_b)
        self.dW = np.dot(dWX_plus_b, self.X.T)
        self.db = np.sum(dWX_plus_b, axis=1).reshape([self.output_num, 1])
        return self.dX
        
    def update(self, learning_rate):
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate
        
        
class ReLU:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.RX = np.zeros(input_shape)
        self.dX = np.zeros(input_shape)
        self.index = None
        
    def forward(self, X):
        self.RX[...] =  X[...]
        self.index = self.RX < 0.0
        self.RX[self.index] = 0.0
        return self.RX
        
    def backward(self, dRX):
        self.dX[...] = dRX[...]
        self.dX[self.index] = 0.0
        return self.dX
    
class SoftmaxLoss:
    def __init__(self, input_num, batch_size):
        self.input_num = input_num
        self.batch_size = batch_size
        self.R = np.zeros((input_num, batch_size))
        self.dR = np.zeros((input_num, batch_size))
        self.loss = 0.0
        self.labels = None
        
    def forward(self, X, labels):
        self.labels = labels
        self.R = X - np.max(X, axis=0).reshape([1, self.batch_size])
        np.exp(self.R, self.R) # in place
        self.R /= np.sum(self.R, axis=0).reshape([1, self.batch_size])
        
        loss = 0.0
        for i in range(self.batch_size):
            loss += np.log(self.R[labels[i], i])
        self.loss = -loss / self.batch_size
        return self.loss
        
    def backward(self):
        self.dR[...] = self.R[...]
        for i in range(self.batch_size):
            self.dR[self.labels[i], i] -= 1 
        return self.dR
    
class Convolution:
    def __init__(self, input_shape, kernel_shape, stride=1): 
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.generate_shape()
        self.input_X = np.zeros(input_shape)
        self.dinput_X = np.zeros(input_shape)
        
        self.conv_result = None
        
    def generate_shape(self):
        batch_size, width, height, channel = self.input_shape
        kernel_width, kernel_height, kernel_num = self.kernel_shape
        
        r_width = kernel_width*kernel_height*channel
        r_k = (width - kernel_width // 2 * 2) * (height - kernel_height // 2 * 2)
         
        self.X = np.zeros((batch_size*r_k, r_width))

        #self.kernel = np.random.normal(0, 1, (r_width, kernel_num))
        low = -1.0 * np.sqrt(6.0 / (r_width + kernel_num))
        high = 1.0 * np.sqrt(6.0 / (r_width + kernel_num))
        self.kernel = np.random.uniform(low=low, high=high,size=(r_width, kernel_num))
        self.dkernel = np.zeros(self.kernel.shape)
        self.dX = np.zeros(self.X.shape)
        self.result_shape = [batch_size, (width - kernel_width // 2 * 2), (width - kernel_width // 2 * 2), kernel_num]
        self.mat_mul_result = np.zeros((batch_size*r_k,kernel_num))
        
        print('X_shape: %s, kernel_shape: %s', self.X.shape, self.kernel.shape)
        
        
    def input_to_X(self):
        batch_size, width, height, channel = self.input_shape
        kernel_width, kernel_height, kernel_num = self.kernel_shape
        r_width = kernel_width*kernel_height*channel
        r_k = (width - kernel_width // 2 * 2) * (height - kernel_height // 2 * 2)
        
        index = 0
        for num in range(batch_size):
            for i in range(width-kernel_width+1):
                for j in range(height-kernel_height+1):
                    self.X[index, :] = self.input_X[num, i:i+kernel_width, j:j+kernel_height, :].reshape(r_width)[:]
                    index += 1
        assert(index == self.X.shape[0])
        
    def forward(self, X):
        assert(self.input_shape == X.shape)
        self.input_X[...] = X[...]
        self.input_to_X()
        self.mat_mul_result[...] = np.dot(self.X, self.kernel)[...]
        self.conv_result = self.mat_mul_result.reshape(self.result_shape)
        
        return self.conv_result
        
    def backward(self, dR):
        batch_size, width, height, channel = self.input_shape
        kernel_width, kernel_height, kernel_num = self.kernel_shape
        
        r_width = kernel_width*kernel_height*channel
        r_k = (width - kernel_width // 2 * 2) * (height - kernel_height // 2 * 2)
        
        # reshape
        dmat_mul_result = dR.reshape([batch_size*r_k, kernel_num])
        self.dX[:,:] = np.dot(dmat_mul_result, self.kernel.T)
        self.dkernel[:,:] = np.dot(self.X.T, dmat_mul_result)       
        
        self.dinput_X[...] = 0.0
        index = 0
        for num in range(batch_size):
            for i in range(width-kernel_width+1):
                for j in range(height-kernel_height+1):
                    self.dinput_X[num, i:i+kernel_width, j:j+kernel_height, :] += self.dX[index, :].reshape(kernel_width, kernel_height, channel)
                    index += 1
        assert(index == self.dX.shape[0])
        return self.dinput_X

    def update(self, learning_rate):
        self.kernel -= self.dkernel * learning_rate
        
            
class Pooling:
    def __init__(self, input_shape, kernel_shape, stride, method='Max'):
        '''support only 2*2 stride 2 max pooling now'''
        assert(kernel_shape == (2, 2))
        assert(stride == 2)
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.stride = stride
        
    def forward(self, input_X):
        batch_size, width, height, n_channel = self.input_shape
        kernel_width, kernel_height = self.kernel_shape
        
        # stride and reshape
        X = np.zeros([batch_size * n_channel * (width / kernel_width) * (height / kernel_height), kernel_width * kernel_height])        
        index = 0
        for num in range(batch_size):
            for i in range(width / kernel_width):
                for j in range(height / kernel_height):
                    X[index*n_channel:(index+1)*n_channel, :] = input_X[num, i*kernel_width:(i+1)*kernel_width,  j*kernel_height:(j+1)*kernel_height, :].reshape(kernel_width * kernel_height, n_channel).T
                    index += 1
        # max pooling
        self.X = X
        self.pooling_X = np.max(self.X, axis=1).reshape([-1, 1])
        self.max_index = np.argmax(self.X, axis=1)

        # reshape to tensor
        self.pooling_images = np.reshape(self.pooling_X, [batch_size, width / kernel_width, height / kernel_height, n_channel])
        return self.pooling_images
        
    def backward(self, dout):
        assert( dout.shape == self.pooling_images.shape)
        batch_size, width, height, n_channel = self.input_shape
        kernel_width, kernel_height = self.kernel_shape
        # reshape to matrix
        reshape_dout = dout.reshape([-1])        
        
        # copy derivate to origal data
        self.dX = np.zeros(self.X.shape)
        self.dX[range(self.X.shape[0]), self.max_index] = reshape_dout
        # stride and reshape
        self.dinput_x = np.zeros(self.input_shape)
        index = 0
        for num in range(batch_size):
            for i in range(width / kernel_width):
                for j in range(height / kernel_height):
                    self.dinput_x[num, i*kernel_width:(i+1)*kernel_width,  j*kernel_height:(j+1)*kernel_height, :] = self.dX[index*n_channel:(index+1)*n_channel, :].T.reshape(kernel_width, kernel_height, n_channel)
                    index += 1
        return self.dinput_x
        
    def update(self):
        pass


        
class MLP:
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fc1 = InnerProduct(32*4*4, batch_size, 100)
        self.relu1 = ReLU([100, batch_size])
        self.fc2 = InnerProduct(100, batch_size, 10)
        self.softmaxLoss = SoftmaxLoss(10, batch_size)

    def forward(self, input_X, input_y):
        fc1_out = self.fc1.forward(input_X) #print fc1_out.max()
        relu1_out = self.relu1.forward(fc1_out)

        fc2_out = self.fc2.forward(relu1_out)
        loss = self.softmaxLoss.forward(fc2_out, input_y)
        return fc2_out, loss
        
    def backward(self):
        dfc2_out = self.softmaxLoss.backward()
        drelu1_out = self.fc2.backward(dfc2_out)
        dfc1_out = self.relu1.backward(drelu1_out)
        dinput_x = self.fc1.backward(dfc1_out)
        #print dfc2_out.shape
        #print drelu1_out.shape
        #print dfc1_out.shape
        #print dinput_x.shape
        return dinput_x
	    
    def update(self):
        self.fc1.update(self.learning_rate)
        self.fc2.update(self.learning_rate)
	
# load data
f = gzip.open('./data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

print train_x.shape, train_y.shape
print valid_x.shape, valid_y.shape
print test_x.shape, test_y.shape


batch_size = 50
learning_rate = 0.001

input_x = train_x[0:batch_size]
print input_x.shape
image = input_x.reshape([batch_size, 28, 28, 1])
print image.shape
k = map(lambda x:x.reshape(28,28,1), input_x)
image = np.require(k)
print image.shape

conv1 = Convolution((batch_size, 28, 28, 1), (5, 5, 20))
relu1 = ReLU([batch_size, 24, 24, 20])
pool1 = Pooling([batch_size, 24, 24, 20], (2, 2), stride=2)

conv2 = Convolution((batch_size, 12, 12, 20), (5, 5, 32))
relu2 = ReLU([batch_size, 8, 8, 32])
pool2 = Pooling([batch_size, 8, 8, 32], (2, 2), stride=2)

mlp = MLP(batch_size, learning_rate)

for epoch in range(200):
    for i in range(train_x.shape[0] // batch_size):
        input_X = train_x[i*batch_size:(i+1)*batch_size, :]
        image = input_X.reshape([batch_size, 28, 28, 1])
        input_y = train_y[i*batch_size:(i+1)*batch_size]
        conv1_out = conv1.forward(image)
        relu1_out = relu1.forward(conv1_out)
        pool1_out = pool1.forward(relu1_out)

        conv2_out = conv2.forward(pool1_out)
        relu2_out = relu2.forward(conv2_out)
        pool2_out = pool2.forward(relu2_out)

        reshape_pool2 = pool2_out.reshape([batch_size, -1]).T
        
        _, loss = mlp.forward(reshape_pool2, input_y)
        
        dreshape_pool2 = mlp.backward()
        #print dreshape_pool1.shape
        dpool2_out = dreshape_pool2.T.reshape([batch_size, 4, 4, 32])

        drelu2_out = pool2.backward(dpool2_out)
        dconv2_out = relu2.backward(drelu2_out)
        dpool1_out = conv2.backward(dconv2_out)

        drelu1_out = pool1.backward(dpool1_out)
        dconv1_out = relu1.backward(drelu1_out)
        dimage = conv1.backward(dconv1_out)
        
        # update
        mlp.update()
        conv2.update(learning_rate)
        conv1.update(learning_rate)
        
        print 'step: %d, loss: %lf' % (i, loss)
        
    accuracy_num = 0
    for i in range(test_x.shape[0] // batch_size):
        input_X = test_x[i*batch_size:(i+1)*batch_size, :]
        image = input_X.reshape([batch_size, 28, 28, 1])
        input_y = test_y[i*batch_size:(i+1)*batch_size]
        conv1_out = conv1.forward(image)
        relu1_out = relu1.forward(conv1_out)
        pool1_out = pool1.forward(relu1_out)

        conv2_out = conv2.forward(pool1_out)
        relu2_out = relu2.forward(conv2_out)
        pool2_out = pool2.forward(relu2_out)

        reshape_pool2 = pool2_out.reshape([batch_size, -1]).T
        
        fc2_out, loss = mlp.forward(reshape_pool2, input_y)
        accuracy_num += np.sum(fc2_out.argmax(axis=0) == input_y)
    print('accuracy: %s'%(accuracy_num))



