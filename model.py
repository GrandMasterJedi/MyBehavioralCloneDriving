
## Architecture for the neural network
def MyKerasArchitecture(inputShape):
    ## Initiate Keras Model and crop data
    input0 = Input(inputShape)
    layer0 = Lambda(lambda x: x/255.0 - 0.5)(input0)

    layer01 = Cropping2D(cropping = ((60,20), (0,0)))(layer0)

    # Convolution and Max Pooling layers
    layer1 = Conv2D(9, (6,6), padding='valid', activation = "relu")(layer01)
    layer2 = MaxPooling2D(pool_size=(2,2), padding="valid")(layer1)

    layer3 = Conv2D(27, (6,6), padding='valid', activation = "relu")(layer2)
    layer4 = MaxPooling2D(pool_size=(2,2), padding="valid")(layer3)

    layer5 = Conv2D(81, (6,6), padding='valid', activation = "relu")(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2), padding="valid")(layer5)
    
    layer7 = Conv2D(81, (3,3), padding='valid', activation = "relu")(layer6)

    # Dropout
    #layer8 = Dropout(0.2)(layer7)

    # Flat layer 
    layer8 = Flatten()(layer7)

    # relu(xw +b) of Fully connected layers
    layer9 = Dense(200, activation = "relu")(layer8)
    layer10 = Dense(50, activation = "relu")(layer9)
    layer11 = Dense(1)(layer10)
    
    model = Model(inputs=input0, outputs=layer11)
    
    return model




## Image Processing Functions

# Multiple (camera) images as input
def read3camerasImg(pathCenter, pathLeft, pathRight, steering, steerAdj = 0.2):
    """
    input: 
        "pathCenter", "pathLeft" and "pathRight" are all (nx1) vector of strings, containing the path
        of the saved image
        example: pathCenter[0], pathLeft[0] and pathRight[0]
        data\IMG\center_2018_04_05_13_39_08_155.jpg
        data\IMG\left_2018_04_05_13_39_08_155.jpg
        data\IMG\right_2018_04_05_13_39_08_155.jpg
        
        "steering" is a vector of double for the steering angle. Left steering is positive, right is negative
        example: steeing[0] = -0.36
    output:
        X, images in numpy array of (n, h, w, d)
        y, steer value in numpy array of size (n, )
    
    """
    assert len(pathCenter) == len(pathLeft) == len(pathRight) == len(steering)
    images3 = []
    steer = []

    nim = len(pathCenter)
    for i in range(nim):
        imgC = cv.cvtColor(cv.imread(xCenter[i]), cv.COLOR_BGR2RGB)
        imgL = cv.cvtColor(cv.imread(xLeft[i]), cv.COLOR_BGR2RGB)
        imgR = cv.cvtColor(cv.imread(xRight[i]), cv.COLOR_BGR2RGB)
        
        # convert 
        # correct steering angle (y) for lect image and right image
        yC = steering[i]
        yL = steering[i] + steerAdj
        yR = steering[i] - steerAdj
        images3.extend([imgC, imgL, imgR])
        steer.extend([yC, yL, yR])
        if (i%1000==0): print("Images imported: " + str(i) + "x 3")
        
    y = np.array(steer)
    X = np.array(images3)
    
    return X, y


def flipBatchImg(images, measurement):
    """
    Data augmentation: Create a new set of images by flipping all images and the measurement
    Input:  images:          (n, w, d, channels)
            measurement:     (n, )
            
    Output  newImages        (n, w, d, channels)
            newMeasurement   (n, )
    """
    assert images.shape[0] == measurement.shape[0]
    
    fImages = []
    fMeasurement = []
    
    for i, img in enumerate(images):
        image_flipped = np.fliplr(img)
        measurement_flipped = - measurement[i]
        fImages.append(image_flipped)
        fMeasurement.append(measurement_flipped)
        if (i%1000==0): print("Images flipped: " + str(i))
        
    newImages = np.array(fImages)
    newMeasurement = np.array(fMeasurement)
        
    return newImages, newMeasurement



