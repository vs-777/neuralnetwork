import numpy as np

class Network:
    
    def __init__(self, shape, imgs, labels, lr):
        '''
        Neural network class initialised with:
        - list containing the structure for the hidden and output layers
        - training images data
        - training labels data
        - learning rate
        '''
        self.shape = shape
        self.imgs = imgs
        self.labels = labels
        self.lr = lr
        self.shape.insert(0, self.imgs[0].flatten().shape[0])

        self.layers = [np.zeros((layer,1)) for i, layer in enumerate(self.shape)]
        self.biases = [np.zeros((layer,1)) for i, layer in enumerate(self.shape[1:])]
        self.weights = [np.random.uniform(-0.5, 0.5, size=(layer,self.shape[i])) for i, layer in enumerate(self.shape[1:])] 

    def feedforward(self, i):
        '''
        Calculate network layers with ReLU activation for hidden layers and softmax activation for output.
        '''
        self.sample = i
        self.inputlayer = self.imgs[self.sample].flatten().reshape(-1,1)
        self.layers[0] = self.inputlayer

        for i in range(len(self.layers[1:-1])):
            self.layers[i+1] = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
            self.layers[i+1] = np.maximum(0,self.layers[i+1])

        self.layers[-1] = np.dot(self.weights[-1], self.layers[-2]) + self.biases[-1]
        self.layers[-1] = np.exp(self.layers[-1]-np.max(self.layers[-1]))/(np.exp(self.layers[-1]-np.max(self.layers[-1])).sum(axis=0))

    def backprop(self):
        '''
        Back propopgation with stochastic gradient descent. Cross entropy loss function. 
        '''
        true = np.zeros((10,1))
        true[self.labels[self.sample]]=1

        loss = -np.sum(true*np.log2(self.layers[-1]))

        self.dcdw = []
        self.dcdb = []

        dcda = self.layers[-1]-true
        dadz = 1
        
        for i in range(len(self.layers[1:])):
            self.dcdw.insert(0,np.dot(self.layers[-i-2], np.transpose(dadz*dcda)))
            self.dcdb.insert(0,dadz*dcda)
            dcda = np.dot(np.transpose(self.weights[-i-1]), dadz*dcda)
            dadz = self.layers[-i-2]
            dadz[dadz>0] = 1

    def update(self):
        '''
        Adjust weights and biases based on calculated partial derivatives.
        '''
        for i in range(len(self.layers[1:])):
            self.weights[i] -= self.lr*np.transpose(self.dcdw[i])
            self.biases[i] -= self.lr*self.dcdb[i]

    def train(self, epoch):
        '''
        Train network.
        '''
        for e in range(epoch):
            for i in range(self.imgs.shape[0]):

                self.feedforward(i)
                self.backprop()
                self.update()

            print("epoch {} complete".format(e+1))
            
        np.save('weights.npy',self.weights)
        np.save('biases.npy',self.biases)

        print("training complete")

    def test(self, imgs_test, labels_test, weights = None, biases = None):
        '''
        Evaluate test data.  Weights and biases from previous runs can be given as arguments.
        '''
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases

        self.imgs = imgs_test
        self.labels = labels_test
        results = []

        for i in range(self.imgs.shape[0]):
            self.feedforward(i)
            if self.labels[self.sample]-np.argmax(self.layers[-1]) == 0:
                results.append(1)
            else:
                results.append(0)
        
        success = results.count(1)/len(results)*100
        print(str(success) + '% \accuracy')
