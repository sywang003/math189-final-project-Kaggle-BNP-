setwd("C:/Users/Andy/Desktop")
train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)
NAs.Percent = NAs / dim(train)[1]
sum(NAs.Percent) / (ncol(train) - 2)

length(unique(train$ID))
length(unique(test$ID))

#Divide into categorical and numeric subsets
Train.categorical = train[,sapply(train, is.factor)]
Train.numeric = train[,sapply(train, is.numeric)]
Test.categorical = test[, sapply(test, is.factor)]
Test.numeric = test[,sapply(test, is.numeric)]

#Numerical
NAs.Num = sapply(Train.numeric, function(x) sum(is.na(x)))
NAs.Num
NAs.Percent.Num = NAs.Num / dim(Train.numeric)[1]
sum(NAs.Percent.Num) / (ncol(Train.numeric) - 2)
summary(Train.numeric)

#Categorical
Train.NA = Train.categorical
Train.NA[Train.NA==""] = NA
NAs.Cat = sapply(Train.NA, function(x) sum(is.na(x)))
NAs.Cat
NAs.Percent.Cat = NAs.Cat / dim(Train.categorical)[1]
sum(NAs.Percent.Cat) / (ncol(Train.categorical) - 2)

#Graphs for certain categorical variables
#red = 1, blue = 0
barplot(table(train$target, train$v3), xlab = "Response", ylab = "Values", main = "Number of counts (v3)", col=c("darkblue","red"))
barplot(table(train$target, train$v30), xlab = "Response", ylab = "Values", main = "Number of counts (v30)", col=c("darkblue","red"))
barplot(table(train$target, train$v47), xlab = "Response", ylab = "Values", main = "Number of counts (v47)", col=c("darkblue","red"))
barplot(table(train$target, train$v66), xlab = "Response", ylab = "Values", main = "Number of counts (v66)", col=c("darkblue","red"))
barplot(table(train$target, train$v71), xlab = "Response", ylab = "Values", main = "Number of counts (v71)", col=c("darkblue","red"))
barplot(table(train$target, train$v74), xlab = "Response", ylab = "Values", main = "Number of counts (v74)", col=c("darkblue","red"))
barplot(table(train$target, train$v75), xlab = "Response", ylab = "Values", main = "Number of counts (v75)", col=c("darkblue","red"))
barplot(table(train$target, train$v110), xlab = "Response", ylab = "Values", main = "Number of counts (v110)", col=c("darkblue","red"))

#Check the pattern of missing data
#Categorical columns removed
library(VIM)
miceplot1 <- aggr(Train.numeric[, -c(1:2)], col=c("blue","red"), 
                  numbers=TRUE, combined=TRUE, varheight=FALSE, border="black",
                  sortVars=TRUE, sortCombs=FALSE, ylabs=c("Pattern of missing data"),
                  labels=names(Train.numeric[-c(1:2)]), cex.axis=.7)


#handling missing data

#Impute missing values with the median
library(randomForest)
set.seed(35)
Train.numeric.median = na.roughfix(Train.numeric)
Test.numeric.median = na.roughfix(Test.numeric)

#Impute with k-nearest-neighbor with k = 3
library(DMwR)
knnimputed <- knnImputation(train[, 3:133], k = 3)

#screening numerical data
#find all entires without nas
complete.numeric.cases <- complete.cases(Train.numeric)
length(which(complete.numeric == TRUE))
complete.numeric <- Train.numeric[complete.numeric.cases, ]
dim(complete.numeric)

#store id and target somewhere else and chop them off from complete numeric
complete.idtarget <- complete.numeric[,1:2]
View(complete.idtarget)
complete.numeric <- complete.numeric[, 3:114]
View(complete.numeric)
dim(complete.numeric)

#correlation analysis
correlations <- cor(complete.numeric)
library(corrplot)
corrplot(correlations, order = "hclust")

#eliminate high correlation data with cutoff 0.80
library(caret)
highcorid <- findCorrelation(correlations, cutoff = .80)
length(highcorid)
complete.numeric <- complete.numeric[,-highcorid]
dim(complete.numeric)
View(complete.numeric)
colnames(complete.numeric)

#t-test
#t-test on v1 and v50 for illustration
v1ttest <- t.test(complete.numeric$v1 ~ as.factor(complete.numeric$target))
v50ttest <- t.test(complete.numeric$v50 ~ as.factor(complete.numeric$target))

#use for loop to do t-test on each variable
ttest <- list()
minuslogp <- c()
for(i in 1:55)
{
  ttest[[i]] <- t.test(complete.numeric[,i] ~ as.factor(complete.idtarget$target))
  minuslogp[i] <- -log(ttest[[i]][[3]])
}
logorder <- order(minuslogp, decreasing = TRUE)
par(mfrow = c(1,1))
orderedlogp <- minuslogp[logorder]
plot(minuslogp[logorder], xaxt = "n", main = "-log(p) of Numerical Variables in t-test", ylab = "-log(p)", xlab = "variabl names")
axis(1, at = 1:55, labels = colnames(complete.numeric)[logorder], las = 2, cex.axis = 0.8)
points( y = orderedlogp[which(orderedlogp > -log(0.0001))], x = 1: length(which(orderedlogp > -log(0.0001))),pch = 19, col = "blue")
abline(h = -log(0.0001), lty = 5, col = "blue", lwd = 2)

#output t-test screened variables
ttestscreened <- colnames(complete.numeric)[which(minuslogp > -log(0.0001))]

#ROC
library(caret)
library(pROC)
dim(complete.numeric)
rocvalues <- filterVarImp(x = complete.numeric, y = as.factor(complete.idtarget$target))
rocrownames <- rownames(rocvalues)[order(rocvalues$X0, decreasing = TRUE)]
orderedrocvalues = rocvalues$X0[order(rocvalues$X0, decreasing = TRUE)]
rocvalues$X0[order(rocvalues$X0, decreasing = TRUE)]
par(mfrow = c(1,1))
plot(rocvalues$X0[order(rocvalues$X0, decreasing = TRUE)], xaxt = "n", main = "AUC Values of Numerical Variables", xlab = "variable names", ylab = "AUC values")
axis(1, at = 1:55 , labels= rocrownames, cex.axis= 0.8, las = 2)
axis(1, at = 1:length(which(orderedrocvalues > 0.52)) , labels= rocrownames[1:length(which(orderedrocvalues > 0.52))], cex.axis= 0.8, las = 2, col = "blue")
abline(h = 0.5, lty = 5, lwd = 2 , col = "red")
abline(h = 0.52, lty = 4, lwd = 2 , col = "blue")
points(y = orderedrocvalues[which(orderedrocvalues > 0.52)], x = 1:length(which(orderedrocvalues > 0.52)), pch = 19, col = "blue")
rocrownames[1:length(which(orderedrocvalues > 0.52))]

#build a ROC curve on v1 and v50 for illustration
par(mfrow=c(1,2))
v1roc <- roc(response = as.factor(complete.numeric$target), predictor = complete.numeric$v1, plot = TRUE)
v50roc <- roc(response = as.factor(complete.numeric$target), predictor = complete.numeric$v50, plot = TRUE)
plot(v1roc, main = "v1 ROC")
plot(v50roc, main = "v50 ROC")

#output ROC screened variables
ROCscreened <- colnames(complete.numeric)[which(rocvalues$X0 > 0.52)]

#screening categorical data
table(train$v22)

#categorical data dummies creation 
library(dummies)
dim(Train.categorical)
View(Train.categorical)
table(train$target ,Train.categorical$v3)

train.dummies <- data.frame()

#test dummy variable generation on v3
dim(dummy("v3", data = train))
train.dummies = dummy("v3", data = train) [, 2:4]
train.dummies <- as.data.frame(train.dummies)

#drop v22, v56, v52, v79, v112, v113, v125 as they all have too many levels
todrop <- c("v22", "v56", "v52", "v79", "v112", "v113", "v125")
numtodrop <- length(todrop)
for(i in 1:numtodrop)
{
  Train.categorical <- Train.categorical[, -which(names(Train.categorical) == todrop[i])]
}
View(Train.categorical)
dim(Train.categorical)

#dummy variable generation on the remaining variables
counter = 1
for(i in 1:12)
{
  varnum <- length(levels(Train.categorical[,i])) - 1;
  train.dummies[, counter:(counter + varnum - 1)] <- dummy(names(Train.categorical)[i], data = Train.categorical)[, 2:(varnum+1)]
  counter = counter + varnum;
}

dim(train.dummies)
View(train.dummies)

#conduct fisher's test
fishertests <- list()
OR <- c()
fisherp <- c()
for(i in 1:57)
{
  fishertests[[i]] <- fisher.test(table(train.dummies[,i], train$target))
  OR[i] <- fishertests[[i]][[3]]
  fisherp[i] <- fishertests[[i]][[1]]
}

#find a good and a bad predictor based on odds ratio
good_cat <- fishertests[[3]]

bad_cat <- fishertests[[which(OR < 1.1 & OR > 0.9)[1]]]

good_cat
bad_cat

#plot ordered OR against ordered p-value
plot(y = OR, x = fisherp, main = "Fisher's Test p-value vs Odds Ratio (OR)", xlab = "p-value", ylab = "OR")

#we chose .667<OR<1.5 to screen the data
abline(h = .667, lty = 5, col = "blue", lwd = 2) 
abline(h = 1.5, lty = 5, col = "blue", lwd = 2)

#we chose p-value < 0.025 to screen the data
abline(v = .025, lty = 6, col = "red", lwd = 2)

#highlight the points passed the screen
counter = 0
for(i in 1:length(OR))
{
  if((OR[i] > 1.5 | OR[i] < 0.667) & fisherp[i] < 0.025)
  {
    points(y = OR[i], x = fisherp[i], pch = 19, col = "blue")
    counter = counter + 1
  }
}
counter 

##output screened categorical variables
catscreened <- train.dummies[, which((OR > 1.5 | OR < 0.667) & fisherp < 0.025)]
dim(catscreened)

#logistic regression
#build the two training set on Train numeric

#reduced
#create the numerical part
reduced <- Train.numeric[, -(highcorid + 2)]
colnames(reduced)

#create the categorical part
dim(reduced)
reduced[, 58:(58+16)] <- catscreened

dim(reduced)

#screened
#create the nuerical part
screened <- Train.numeric[, ttestscreened]
dim(screened)
screened[,1:2] <- Train.numeric[,1:2]
screened[, 3:36] <- Train.numeric[, ttestscreened]
dim(screened)
View(screened)

#create the categorical part
screened[, 37:(37+16)] <- catscreened
dim(screened)
View(screened)

##insert median to the data
library(randomForest)
set.seed(35)
median.reduced <- na.roughfix(reduced)
median.screened <- na.roughfix(screened)
median.numeric <- na.roughfix(Train.numeric)
View(median.numeric)

library(caret)
#train logistic 
median.reduced$target = as.factor(median.reduced$target )
median.screened$v5 = as.factor(median.screened$v5 )
median.numeric$target = as.factor(median.numeric$target)

ctrl <- trainControl(method = "LGOCV", classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(10)
logisticReg <- train(target~., data = median.reduced[, -1], method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 3))
logisticReg
logisticReg2 <- train(v5~., data = median.screened[, -1], method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 3))
logisticReg2
logisticReg3 <- train(target~., data = median.numeric[, -1], method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 3))
logisticReg3

#Impute missing values with the median
#Categorical columns removed 
#100 trees and 300 trees
library(randomForest)
set.seed(35)
Train.numeric.median = na.roughfix(Train.numeric)
Test.numeric.median = na.roughfix(Test.numeric)

RandomF = randomForest(as.factor(target) ~ ., data = Train.numeric.median, ntree = 100)
fit = predict(RandomF, Test.numeric.median, type="prob")[,2]
write.csv(data.frame(ID = test$ID, PredictedProb = fit), "Submission.csv", row.names = F)

#300 Trees
RandomF = randomForest(as.factor(target) ~ ., data = Train.numeric.median, ntree = 300)
fit = predict(RandomF, Test.numeric.median, type="prob")[,2]
write.csv(data.frame(ID = test$ID, PredictedProb = fit), "Submission.csv", row.names = F)

#Check the categorial columns
sapply(Train.categorical, mode)
sapply(Train.categorical, class)

#Check correlation
library(Hmisc)
Train.NumCor = rcorr(as.matrix(Train.numeric), type = "pearson")
Train.NumCor.DF = as.data.frame(Train.NumCor$r)
Train.NumCor.DF[Train.NumCor.DF == 1] = NA
apply(Train.NumCor.DF, 1, max, na.rm = TRUE)

#Categorical columns converted to numeric
#Impute median for missing values, 100 Trees
NewTrain = data.frame(sapply(train, as.numeric))
NewTest = data.frame(sapply(test, as.numeric))
NewTrain = na.roughfix(NewTrain)
NewTest = na.roughfix(NewTest)

RandomF = randomForest(as.factor(target) ~ ., data = NewTrain, ntree = 100)
fit = predict(RandomF, NewTest, type="prob")[,2]
write.csv(data.frame(ID = NewTest$ID, PredictedProb = fit), "Submission.csv", row.names = F)

install.packages("Hmisc")
library(Hmisc)
x = as.matrix(Train.numeric)
y = rcorr(x, type="pearson")
y = cor(Train.numeric)

