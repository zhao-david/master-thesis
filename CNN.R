rm(list=ls())

require(mxnet)

#########################################################

### load and preprocess data ###

#########################################################

# load data 
train <- read.csv("C:\\Users\\David\\Documents\\ETHZ 2015-2017\\'16 HERBST\\THESIS\\R\\train_6000.csv", header=TRUE)
test <- read.csv("C:\\Users\\David\\Documents\\ETHZ 2015-2017\\'16 HERBST\\THESIS\\R\\test_6000.csv", header=TRUE)

train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]
train.x <- t(train.x/255)

test.x <- test[,-1]
test.y <- test[,1]
test.x <- t(test.x/255)

table(train.y)

#########################################################

### 2-layer neural network (with softmax regression) ###

#########################################################

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax,
                                     X=train.x,
                                     y=train.y,
                                     ctx=devices,
                                     num.round=20,
                                     array.batch.size=100,
                                     learning.rate=0.07,
                                     momentum=0.9,
                                     eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, test.x)
dim(preds)

pred.label <- max.col(t(preds)) - 1
table(pred.label)

num_correct <- sum(test.y == pred.label)
accuracy <- num_correct / 1000
accuracy




#########################################################

### 2-layer CNN (LeNet-5) ###

#########################################################

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))

device.cpu <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=20, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
print(proc.time() - tic)

preds <- predict(model, test.array)
dim(preds)

pred.label <- max.col(t(preds)) - 1
table(pred.label)

num_correct <- sum(test.y == pred.label)
accuracy <- num_correct / 1000
accuracy

model_params <- model$arg.params
dim(model_params$convolution0_weight)
dim(model_params$convolution1_weight)

model_params$convolution0_weight
model_params$convolution1_weight
