
library(caret)
library(ggplot2)
library(dplyr)
library(rattle)
library(rpart)
library(rpart.plot)

set.seed(1235)
rm(list=ls())
#Load
trainingDestfile=".//pml-training.csv"
trainingDataRaw = read.csv(file=trainingDestfile)

testingDestfile=".//pml-testing.csv"
testingDataRaw = read.csv(file=testingDestfile)

# Cleaning Data 
trainingDataClean = trainingDataRaw[,colSums(is.na(trainingDataRaw)) == 0]
irrelevant_indices = 1:7
relevant_indices = - irrelevant_indices
trainingDataClean = trainingDataClean[,relevant_indices]

testingDataClean = testingDataRaw[,colSums(is.na(testingDataRaw)) == 0]
irrelevant_indices = 1:7
relevant_indices = - irrelevant_indices
testingDataClean = testingDataClean[,relevant_indices]

# Partition
inTrain = createDataPartition(y=trainingDataClean$classe, p=0.75, list= FALSE)
training = trainingDataClean[inTrain,]
testing = trainingDataClean[-inTrain,]

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
#descrCor <- cor(trainingStep, use="complete.obs")#
trainingTemp = trainingStep[,-53]
descrCor <- cor(trainingTemp)
highlyCorrIndices = findCorrelation(descrCor, cutoff = 0.75)
trainingCor = trainingTemp[,-highlyCorrIndices]
trainingCor$classe = trainingStep$classe
trainingStep = trainingCor


# Option
# instanceconvert2 <- colnames(trainingStepB)
# for (j in instanceconvert2) {
#   print(j)
#   trainingStepB[,j]=as.numeric(trainingStepB[,j])
# }
#descrCorB <- cor(trainingStepB, use="complete.obs")
trainingTempB = trainingStepB[,-53]
descrCorB <- cor(trainingTempB)
highlyCorrIndicesB = findCorrelation(descrCorB, cutoff = 0.75)
trainingCorB = trainingTempB[,-highlyCorrIndicesB]
trainingCorB$classe = trainingStepB$classe
trainingStepB = trainingCorB


# Final Adjustments
trainingSemifinal = trainingStep
trainingSemifinalB = trainingStepB

# Training - Decision Tree

# Option
#ctrl = trainControl(method="repeatedcv", repeats = 3)
#modelFit <- train(classe ~., data = trainingSemifinal, method="rpart", tuneLength = 15, trControl=ctrl)
modelFit <- train(classe ~ .,method="rpart",data=trainingSemifinal)
print(modelFit$finalModel)
#plot(modelFit$finalModel, uniform=TRUE, main="Classification Tree")
#text(modelFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
fancyRpartPlot(modelFit$finalModel)

# Predictions
predicted = predict(modelFit, newdata=testing)
confusionMatrix(predicted, testing$classe)


# StepB
ctrlB = trainControl(method="repeatedcv", number = 10, repeats = 10)
modelFitB <- train(classe ~., data = trainingSemifinalB, method="rpart", tuneLength = 15, trControl=ctrlB)
#modelFitB <- train(classe ~ .,method="rpart",data=trainingSemifinalB)
print(modelFitB$finalModel)
#plot(modelFitB$finalModel, uniform=TRUE, main="Classification Tree")
#text(modelFitB$finalModel, use.n=TRUE, all=TRUE, cex=.8)
fancyRpartPlot(modelFitB$finalModel)

# Predictions
predictedB = predict(modelFitB, newdata=testing)
confusionMatrix(predictedB, testing$classe)
length(predictedB)

# Training - Random Forests
# StepB

# Too much time
#ctrlC = trainControl(method="repeatedcv", number = 10, repeats = 10)
#modelFitC <- train(classe ~., data = trainingSemifinalB, method="rf", tuneLength = 10, trControl=ctrlC, verbose = TRUE, trace = TRUE)

# Moderate Time
ctrlC = trainControl(method="repeatedcv", number = 10, repeats = 2)
modelFitC <- train(classe ~., data = trainingSemifinalB, method="rf", tuneLength = 2, trControl=ctrlC, verbose = TRUE, trace = TRUE)
#modelFitB <- train(classe ~ .,method="rpart",data=trainingSemifinalB)
print(modelFitC$finalModel)
#plot(modelFitB$finalModel, uniform=TRUE, main="Classification Tree")
#text(modelFitB$finalModel, use.n=TRUE, all=TRUE, cex=.8)
#fancyRpartPlot(modelFitB$finalModel)

# Predictions
predictedC = predict(modelFitC, newdata=testing)
confusionMatrix(predictedC, testing$classe)
length(predictedC)

answers = predict(modelFitC, newdata=testingDataClean)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)

