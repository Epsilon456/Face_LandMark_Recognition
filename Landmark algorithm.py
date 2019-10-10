#Install libraries
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

dataFile = #[Insert Path to data file here]
landMarkFiles = #[Insert path to landmark file here.]


#Load File containing the images 
data = load(dataFile)

##############################################################################
####################Preprocessing Function Definitions#######################
##############################################################################
def deltaFunction(a,b):

    alpha = .1
    distance = np.linalg.norm(a-b,axis=1)
    return np.exp(-alpha*distance)

def GenerateHeatMap(landMarks):
    
    heatMap = []
    for i in range(96):
        _heatMap = []
        for j in range(96):
            element = np.expand_dims(np.array([i,j]),0)
            left = deltaFunction(element,landMarks[:,0:2])
            right = deltaFunction(element,landMarks[:,2:4])
            _heatMap.append([left,right])
        heatMap.append(_heatMap)
    HeatMap = np.array(heatMap)
    return np.transpose(HeatMap,(3,1,0,2))
            
################################################################################################
########################Load Image Data##############################################
################################################################################################
print("Loading data")
#Read npz file
lst = data.files
a = data[lst[0]]
#flip rows and columns. Divide result by 255 so that each pixel is a value between 0 and 1
_images = np.transpose(a,(2,0,1))/255

########################Load Landmark Data##############################################
#create pandas dataframe.
df = pd.read_csv(landMarkFiles)
#obtain x and y coordinates for left and right eyes.
lx = np.round(np.array(df['left_eye_center_x']),0)
ly = np.round(np.array(df['left_eye_center_y']),0)
rx = np.round(np.array(df['right_eye_center_x']),0)
ry = np.round(np.array(df['right_eye_center_y']),0)
#Stack this data into one big array.
_landMarks = np.stack((lx,ly,rx,ry),axis=1)

#Filter the data poitns. Some data fields have missing data that will show up as 'nan'
    #The sum along axis 1 (the xy coordinates of each eye) will be 'nan' if any one field is 'nan'
    #Using np.where, the indices where there is no 'nan' value in the row will be stored as 'goodIndices'.
goodIndices = np.where(np.logical_not(np.isnan(np.sum(_landMarks,axis=1))))[0]
#obtain the image arrays corresponding to the 'goodIndices'
images = _images[goodIndices]
#obtain the landmark xy coordinates for the 'goodIndices'. Convert coordinates to an integer so that 
    #each coordinate pair coresponds to an individual pixel.
landMarks = _landMarks[goodIndices].astype(np.int32)
#Given the landmark data, create a heatmap which is 'warmest' around the landmark points.
HeatMap = GenerateHeatMap(landMarks)
#Free up memory by deleting these variables which are no longer needed.
del _images,_landMarks

#########################Split Train/test data##############################################
#Take the image arrays as the input 'U' and the heatmaps as the labels 'Z'.  Break each into a training and 
    #testing dataset.  (It is assumed that the data was shuffled before it was originally saved.)
UTrain = images[0:7000]
UTest = images[7000:-1]
ZTrain = HeatMap[0:7000]
ZTest = HeatMap[7000:-1]
    
################################################################################################
##########################Data Sets#################################################
################################################################################################
#Define the batch size and the number of training epochs
batchSize = 8
epochs = 10000

#Set the graph to run on the gpu
with tf.device('/gpu:1'):
    
    #Define placeholders
    
    #the input data is an array consisting of the images of shape [batchSize,rows,colums]
    _U = tf.placeholder(tf.float32,shape=[None,96,96])
    #The label data array consisting of heatmaps of the shape [batchSize,rows,columns,lefteye/righteye]
    _Z = tf.placeholder(tf.float32,shape=[None,96,96,2])
    
    #Define datasets
    dU = tf.data.Dataset.from_tensor_slices(_U)
    dZ = tf.data.Dataset.from_tensor_slices(_Z)
    dataset = tf.data.Dataset.zip((dU,dZ))
    
    #Shuffle and repeat if training
    Dataset = dataset.batch(batchSize).shuffle(320).prefetch(128).repeat()

    ####################################Set iterator###################################
    iterator = Dataset.make_initializable_iterator()
    U,ZZ = iterator.get_next()
    
############################Graph Functions#######################################################
    def He(fanin):
        """Defines the needed variance to initialize a variable  with Xavier-He initialization
        """
        return 1/fanin
    
    def Convolve(inputTensor,width,filters,mode='2d'):
        """Adds a convolutional and optional pooling layer to the graph.
        Inputs:
            inputTensor - [batch,res,res,inputChannels] A 4d tensor which is to undergo convolution.
            width - The width of the convolution kernel (This will usually be 3).
            filters - the number of kernels to apply to the layer
            mode - select '2d' for a classic 2d convolution with a pooling layer
                - select 'simple' to provide a convolution without the pooling laye (used for a 1x1 convolution in this script).
        Outputs:
            A completed convolution 
                if mode is '2d' the output is [batch,res/2,res/2,filters]
                if mode is 'simple' the output is [batch,res,res,filters]
                
        """
        #Comput the number of input channels
        inputSize = int(inputTensor.shape[-1])
        #Calculate number of inputs to be used with the Xavier-He variance calculation
        variance = width**2*inputSize
        #Define a kernel variable of size [width,width,inputChannels,filters]
        WC = tf.Variable(tf.truncated_normal([width,width,inputSize,filters],stddev=He(variance),dtype=tf.float32))
        #define a bias variable of size [filters,]
        BC = tf.Variable(tf.truncated_normal((filters,),stddev = He(filters),dtype=tf.float32))
        
        #Perform a 2D convolution with no stride. Add bias.
        CV = tf.nn.conv2d(inputTensor,WC,[1,1,1,1],"SAME")+BC
        if mode == '2d':
            #Perform a pooling operation with a stride of 2. Add a leaky_relu activation function.
            Pool = tf.nn.leaky_relu(tf.nn.pool(CV,[2,2],"MAX","SAME",strides=[2,2]),.01)
            return Pool
        elif mode == 'simple':
            return CV

    def Pool(inputTensor):
        """Performs a max pooling poeration on an input tensor which will reduce the resolution by 2.
        Input shape [batch,res,res,filters]
        Output shape [batch,res/2,res/2,filters]
        """
        return tf.nn.pool(inputTensor,[2,2],"MAX","SAME",strides=[2,2])
    
    def UpSample(inputTensor,width,filters):
        """Performs a convolution transpose operation to double the resolution of the input tensor. This can be thought of 
        as the inverse of a convolution-pooling pair.
        Inputs:
            inputTensor - The input image [batchSize,res,res,inChannels]
            width - The width of the convolutional kernel (usually 3)
            fitlers - The number of kernels used in the convolution (also the number of output channels.)
        Outputs:
            A convolved tensor of shape [batchSize,2res,2res,filters]
        """
        #Computes the number of input channels to the input tensor.
        inputSize = int(inputTensor.shape[-1])
        #Computs the size of the output tensor (res*2)
        outSize = int(inputTensor.shape[1])*2
        #Computes the number of inputs used for the Xavier-He variance calculation.
        variance = width**2*inputSize
        
        #Define the kernel and bias
        WC = tf.Variable(tf.truncated_normal([width,width,filters,inputSize],stddev=He(variance),dtype=tf.float32))
        B = tf.Variable(tf.truncated_normal((filters,),stddev=He(filters),dtype=tf.float32))
        
        #Perform the convolution transpose operation
        CVT = tf.nn.conv2d_transpose(inputTensor,WC,strides=[1,2,2,1],padding="SAME",output_shape=np.array([batchSize,outSize,outSize,filters]))
        #Performs a pooling operation to fill in the gaps between the strides portions of the convolution transpose
        result = tf.nn.pool(CVT,[2,2],"MAX","SAME",strides=[1,1])
        #Use a leaky_relu activation function.
        Result = tf.nn.leaky_relu(result+B)
        return Result
    
    def DownResidual(tensor,filters):
        """Adds a residual layer to the convolutional layer. This creates two convolutions which are added to getehr at the end
        The first convoution uses a 3x3 kernel to identify local features.  The second convolution uses a 1x1 convolution
        to adjust the number of filters while preserving features relative to each other. Adding these two together ensures
        that each layer keeps some information from the previous layer.
        Inputs:
            Tensor - The input tensor of size [batch,res,res,inputChannels]
            filters - The number of output channels desired
        Outputs:
            A tensor of size [batchSize,res/2,res/2,filters]
        """
        #Perform a regular convolution and pool to reduce resolution by 2
        B = Convolve(tensor,3,filters)
        #Perform a 1x1 convolution on the original input keeping the same resolution as the original
        A = Convolve(tensor,1,filters,mode='simple')
        #Perform a pooling operation on A to reduce resolution by 2 (without an activation function)
        a = Pool(A)
        #Add the results of the 1x1 convolution and the 3x3 convolution
        return B+a
    
    def GetIndex(out2):
        """Obtains the xy indices of the pixels for both the left and right eyes. This is done by looking over the heatmap
        which represents the probability of each pixel being the pixel being in the center of a given eye. This function
        will identify the pixel with the max probability of being each eye.
        Inputs:
            out2 - [batchSize,96,96,2] A predicted heatmap that tries to mimic the original labels of 
        Outpus:
            
    
        """
        #Flatten the array such that the axes 1 and 2 (corresponding to rows and columns of the image) lay along axis 1.
        flattened = tf.reshape(out2,(-1,96*96))
        #Obtain the indices of the maximum values of each eye for each image [batchSize*2]
        index = tf.argmax(flattened,axis=1)
        #Obtain the row index of each by finding the quotient of each index and the number of elements in each row.
        x = tf.cast(tf.expand_dims(index/96,-1),dtype=tf.int64)
        #Find the column index by finding the remainder of the previous quotient.
        y = tf.expand_dims(index%96,-1)
        #Create an array of size [2batch,2,1] to represent the placement of the left and right eyes.
        xy = tf.stack((x,y),axis=1)
        return xy


###################################Tensorflow Graph################################################# 
with tf.device('/gpu:1'):
    #Expand the tensor UU by one dimension to shape [batchSize,rows,columns,1]
        #[batch,96,96,1]
    UU = tf.expand_dims(U,-1)
    
    #############Downsampling###############
    """The following part will generate 4 layers of convolution operations. This will occur in 3 steps.
    CV: First, the DownResidual function will perform a convolution and pooling operation followed by a resdidual layer
    CVB: Next, a version of CV will be run through a 1x1 convolution to add more filters so that it will match the 
        opposite block in the network "LD".
    ZP: The heatmap will undergo a pooling operation so that it is in thesame resolution as the corresponding CV layer.
    
    This process is repeated 4 times - each time cutting the resolution in half. 
    """

    
    CV1 = DownResidual(UU,4) #[batch,48,48,4]
    CV1B = Convolve(CV1,1,8,mode='simple') #[batch,48,48,8]
    ZP1 = Pool(ZZ) #[batch,48,48,2]
    
    CV2 = DownResidual(CV1,8) #[batch,24,24,8]
    CV2B = Convolve(CV2,1,16,mode='simple') #[bath,24,24,16]
    ZP2 = Pool(ZP1) #[batch,24,24,2]
    
    #The results of this downresidual is scaled by 1/10 to keep all layers at a relatively consistent scale.
    CV3 = DownResidual(CV2,16)/10 #[batch,12,12,16]
    CV3B = Convolve(CV3,1,32,mode='simple') #[batch,12,12,32]
    ZP3 = Pool(ZP2) #[batch,12,12,2]
    
    CV4 = DownResidual(CV3,32) #[batch,6,6,32]
    ZP4 = Pool(ZP3) # [batch,6,6,2]

    ##########Upsampling#####################
    """This portion of the network will use transpose convolutions to bring the resolution back up to the original
    resolution. At each step, the layer is added to a corresponding 'CVB' layer before undergoing a convolutional transpose
    The resolution is doubled at each step.
    """
    
    L4D = UpSample(CV4,3,32) # [batch,12,12,32]
    L3D = UpSample(L4D+CV3B,3,16) #[batch,24,24,16]
    L2D = UpSample(L3D+CV2B,3,8) #[batch,48,48,8]
    L1D = UpSample(L2D+CV1B,3,2) #[batch,96,96,2]
    
    ##########Loss and optimization#####
    #Use a 1x1 convolution to bring CV4 (lowest res) to have 2 filters so that it will be the same size as ZZ.
        #Perform a signmoid activation function to scale the results between 0 and 1.
    out1 = tf.nn.sigmoid(Convolve(CV4,1,2,mode='simple'))
    #Scale the results of L1D (highest res) using a sigmoid function.
    out2 = tf.nn.sigmoid(L1D)
    
    #Obtain the index of the left and right eye (only used for validation)
    left = GetIndex(out2[:,:,:,0])
    right = GetIndex(out2[:,:,:,1])

    #Set Loss1 to be the error between the lowest and ZP4 (the version of the labels that is the same res as CV4)
    Loss1 = tf.reduce_sum(tf.square(out1-ZP4))
    #Set Loss2 to be the error between the highest res an the original labels.
    Loss2 = tf.reduce_sum(tf.square(out2-ZZ))
    #Sum the high res and low res losses together.
    Loss = Loss1+Loss2
    #Apply an adam optimizer
    optimizer = tf.train.AdamOptimizer(.001).minimize(Loss)
    
###################################Initialize Session###################################      
#Configure the session to allow soft placement. This causes operations which cannot be run on the GPU 
    #to instead be run on the CPU.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#Initializes the trainable varaibles.
sess.run(tf.global_variables_initializer())
#Initializes the iterator and feeds the data/labels
sess.run(iterator.initializer,feed_dict={_U:UTrain,_Z:ZTrain})

###################################Run Session###################################
Losses = []
#Run the training session and record losses.
for i in range(epochs):
    _,loss = sess.run([optimizer,Loss])
    Losses.append(loss)
    print(i,loss)

print("Train fit",Loss)
###################################Evaluate Results###################################
#Initialize the iterator with the test data set
sess.run(iterator.initializer,feed_dict={_U:UTest,_Z:ZTest})

#Perform an error calculation which averages the differences between the output and the heat map.
def ErrorCalc(Out,Labels):
    return np.sum(abs(Out-Labels))/Out.shape[0]
#Iterate through 3 batches of data
for I in range(3):
    #Pull the tensors that we want to look at from the session.
    A,B,C,D,E,G,F = sess.run([out1,ZP4,out2,ZZ,left,right,UU])
    #Go through each item in the batch
    for i in range(batchSize):
        a = A[i,:,:,0]
        b = B[i,:,:,0]
        c = C[i,:,:,0]
        d = D[i,:,:,0]
        e = E[i,:,0]
        g = G[i,:,0]
        f = F[i,:,:,0]*96
        #Display the image
        plt.imshow(f)
        #Plot a point over each eye.
        plt.scatter(e[1],e[0],color='r')
        plt.scatter(g[1],g[0],color='b')
        #Show the graph
        plt.show()
        
        #Pause until the user presses a key. Break out of both loops if the user presses 'q'.
        q = input("Press any key to go to next image\nPress 'q' to quit.")
        if q == 'q':
            break
    if q == 'q':
        break
    
