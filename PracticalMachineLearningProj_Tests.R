

library(caret)
library(ggplot2)
library(dplyr)
library(rattle)
library(rpart)
library(rpart.plot)

set.seed(1235)

#Load
trainingDestfile=".//pml-training.csv"
trainingData = read.csv(file=trainingDestfile)

testingDestfile=".//pml-testing.csv"
testingData = read.csv(file=testingDestfile)

# Partition
inTrain = createDataPartition(y=trainingData$classe, p=0.75, list= FALSE)
training = trainingData[inTrain,]
testing = trainingData[-inTrain,]

dim(training); dim(testing)

# Initial
trainingStep = training
trainingStepB = training

# Selecting relevant
# Option

# Imputing missing values
# Option

#Creating Dummy Variables

#Removing zero- and near-zero-variance variables
nzv = nearZeroVar(trainingStep, saveMetrics = TRUE)
nzv
dim(trainingStep)
trainingNZV = trainingStep[, !nzv$zeroVar & !nzv$nzv ]
dim(trainingNZV)
trainingStep = trainingNZV

nzvB = nearZeroVar(trainingStepB, saveMetrics = TRUE)
nzvB
dim(trainingStepB)
trainingNZVB = trainingStepB[, !nzvB$zeroVar & !nzvB$nzv ]
dim(trainingNZVB)
trainingStepB = trainingNZVB

# Removing highly correlated variables

# Option
#descrCor <- cor(trainingStep, use="complete.obs")
descrCor <- cor(trainingStep)
highlyCorrIndices = findCorrelation(descrCor, cutoff = 0.75)
trainingCor = trainingStep[,-highlyCorrIndices]
#descrCor2 = cor(trainingCor, use="complete.obs")
#summary(descrCor2)
trainingStep = trainingCor

# Option
# instanceconvert2 <- colnames(trainingStepB)
# for (j in instanceconvert2) {
#   print(j)
#   trainingStepB[,j]=as.numeric(trainingStepB[,j])
# }
#descrCorB <- cor(trainingStepB, use="complete.obs")

descrCorB <- cor(trainingStepB)
highlyCorrIndicesB = findCorrelation(descrCorB, cutoff = 0.75)
trainingCorB = trainingStepB[,-highlyCorrIndicesB]
trainingStepB = trainingCorB


# Final Adjustments
trainingSemifinal = trainingStep
trainingSemifinalB = trainingStepB

# Training

# Option
#ctrl = trainControl(method="repeatedcv", repeats = 3)
#modelFit <- train(classe ~., data = trainingSemifinal, method="rpart", tuneLength = 15, trControl=ctrl)
modelFit <- train(classe ~ .,method="rpart",data=trainingSemifinal)
print(modelFit$finalModel)
plot(modelFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modelFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
fancyRpartPlot(modelFit$finalModel)

# Option
ctrlB = trainControl(method="repeatedcv", number = 10, repeats = 10)
modelFitB <- train(classe ~., data = training, method="rpart", tuneLength = 15, trControl=ctrl)
#modelFitB <- train(classe ~ .,method="rpart",data=trainingSemifinalB)
print(modelFitB$finalModel)
plot(modelFitB$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modelFitB$finalModel, use.n=TRUE, all=TRUE, cex=.8)
fancyRpartPlot(modelFitB$finalModel)
# Predictions
predicted = predict(modelFit, newdata=testing, type="class")
confusionMatrix(predicted, testing$classe)
length(predicted)

predictedB = predict(modelFitB, newdata=testing, type="class")
confusionMatrix(predictedB, testing$classe)
length(predictedB)