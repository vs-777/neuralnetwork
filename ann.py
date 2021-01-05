import numpy as np
import idx2numpy

class Network:

    def __init__(self, shape, imgs, labels, lr):

        self.shape = shape
        self.imgs = imgs
        self.labels = labels
        self.lr = lr
        self.shape.insert(0, self.imgs[0].flatten().shape[0])

        self.layers = [np.zeros((layer,1)) for i, layer in enumerate(self.shape)]
        self.biases = [np.zeros((layer,1)) for i, layer in enumerate(self.shape[1:])]
        self.weights = [np.random.uniform(-0.5, 0.5, size=(layer,self.shape[i])) for i, layer in enumerate(self.shape[1:])] 

    def feedforward(self, i):

        self.sample = i

        self.inputlayer = self.imgs[self.sample].flatten().reshape(-1,1)
        self.layers[0] = self.inputlayer

        for i in range(len(self.layers[1:-1])):
            #hidden layer
            self.layers[i+1] = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
            #hidden layer relu
            self.layers[i+1] = np.maximum(0,self.layers[i+1])

        #output layer
        self.layers[-1] = np.dot(self.weights[-1], self.layers[-2]) + self.biases[-1]
        #output layer softmax
        self.layers[-1] = np.exp(self.layers[-1]-np.max(self.layers[-1]))/(np.exp(self.layers[-1]-np.max(self.layers[-1])).sum(axis=0))

    def backprop(self):

        true = np.zeros((10,1))
        true[self.labels[self.sample]]=1

        #cross-entropy loss 
        loss = -np.sum(true*np.log2(self.layers[-1]))

        self.dcdw = []
        self.dcdb = []

        dcda = self.layers[-1]-true
        dadz = 1
        
        for i in range(len(self.layers[1:])):
            self.dcdw.insert(0,np.dot(self.layers[-i-2], np.transpose(dadz*dcda)))
            self.dcdb.insert(0,dadz*dcda)
            dcda = np.dot(np.transpose(self.weights[-i-1]), dadz*dcda)
            dadz = self.layers[-i-2]#np.copy(self.layers[-i-2])
            dadz[dadz>0] = 1

    def update(self):

        for i in range(len(self.layers[1:])):
            self.weights[i] -= self.lr*np.transpose(self.dcdw[i])
            self.biases[i] -= self.lr*self.dcdb[i]

    def train(self, epoch):

        for e in range(epoch):
            for i in range(self.imgs.shape[0]):
                #for image in range(1):#self.imagesarray.shape[0]):
                self.feedforward(i)
                self.backprop()
                self.update()
                #print("batch {} complete".format(i))
            print("epoch {} complete".format(e))

        
        np.save('weights.npy',self.weights)
        np.save('biases.npy',self.biases)

        print("training complete")


    def test(self, imgs_test, labels_test, weights = None, biases = None):
        
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

    def quicktest(self):
        for i in range(20):
            sample_rand = np.random.randint(50000)
            self.feedforward(sample_rand)
            print(self.labels[self.sample])
            print(np.argmax(self.layers[-1]))
            print('')



imagesfile, labelsfile = 'data/fashion/train-images-idx3-ubyte', 'data/fashion/train-labels-idx1-ubyte'
images = idx2numpy.convert_from_file(imagesfile)/255
labels = idx2numpy.convert_from_file(labelsfile)

imagesfile_test, labelsfile_test = 'data/fashion/t10k-images-idx3-ubyte', 'data/fashion/t10k-labels-idx1-ubyte'
images_test = idx2numpy.convert_from_file(imagesfile_test)/255
labels_test = idx2numpy.convert_from_file(labelsfile_test)

shape = [64,64,10]

# weights = np.load('200_80_x4_w.npy', allow_pickle=True)
# biases = np.load('200_80_x4_b.npy', allow_pickle=True)

# shape = [weights[i].shape[0] for i in range(len(weights))]


network = Network(shape, images, labels, 0.001)
network.train(3)
network.test(images_test, labels_test)#, weights, biases)





