library(doParallel)
library(caret)       #R modeling workhorse & ggplot2)
library(tidyr)
library(dplyr)
library(plotly)
library(corrplot)
library(ggplot2)
library(GGally)
library(kableExtra)
library(Hmisc)
library(kknn)
library(reshape2)


### ---- Import Ready Data ----
iphonedata <- readRDS("iphone_dataframe.rds")

galaxydata <- readRDS("galaxy_dataframe.rds")

### ---- Preprocessing ----
## Inspect data types
iphonedata$iphonesentiment <- as.factor(iphonedata$iphonesentiment)
str(iphonedata)

galaxydata$galaxysentiment <- as.factor(galaxydata$galaxysentiment)
str(galaxydata)

### ---- Data Partition ----
## iphone
set.seed(1234)
iphonesample <- iphonedata[sample(1:nrow(iphonedata),
                                  1000, 
                                  replace = FALSE), ]
intrain1 <- createDataPartition(y = iphonesample$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain <- iphonesample[intrain1,]
iphonetest <- iphonesample[-intrain1,]

## galaxy
set.seed(2345)
intrain2 <- createDataPartition(y = galaxydata$galaxysentiment, 
                                p = 0.7, 
                                list = FALSE)
galaxytrain <- galaxydata[intrain2,]
galaxytest <- galaxydata[-intrain2,]


### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 4

## Create cluster with desired number of cores.
cl <- makeCluster(3)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4


########## train and test  model for the iphone ###################

a <- c("ranger", "C5.0", "kknn", "svmLinear")
compare_model <- c()
for(i in a) {
  model <- train(iphonesentiment ~., data = iphonetrain, method = i)
  pred <- predict(model, newdata = iphonetest)
  pred_metric <- postResample(iphonetest$iphonesentiment, pred)
  compare_model <- cbind(compare_model , pred_metric)
}
colnames(compare_model) <- a
compare_model

#####################################################################

#Structure table for plot
compare_model_melt <- melt(compare_model, varnames = c("metric", "model"))
compare_model_melt <- as_data_frame(compare_model_melt)
compare_model_melt

#plot
ggplot(compare_model_melt, aes(x=model, y=value))+
  geom_col()+
  facet_grid(metric~., scales="free")


########## train and test  model for the galaxy  ###################

a <- c("ranger", "C5.0", "kknn", "svmLinear")
compare_model <- c()
for(i in a) {
  model <- train(galaxysentiment ~., data = galaxytrain, method = i)
  pred <- predict(model, newdata = galaxytest)
  pred_metric <- postResample(galaxytest$galaxysentiment, pred)
  compare_model <- cbind(compare_model , pred_metric)
}
colnames(compare_model) <- a
compare_model

#####################################################################

#Structure table for plot
compare_model_melt <- melt(compare_model, varnames = c("metric", "model"))
compare_model_melt <- as_data_frame(compare_model_melt)
compare_model_melt

#plot
ggplot(compare_model_melt, aes(x=model, y=value))+
  geom_col()+
  facet_grid(metric~., scales="free")

#cmRF <- confusionMatrix(pred, galaxytest$galaxysentiment) 

## Stop Cluster
stopCluster(cl)
