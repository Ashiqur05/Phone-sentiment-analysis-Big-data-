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

### ---- Import Small Matrices(iphone) ----
getwd()
## Inspect the data(iphone)
iphone_sm<-read.csv("sentiment analysis/iphone_smallmatrix_labeled_8d.csv")
summary(iphone_sm)
str(iphone_sm)
dim(iphone_sm)

### ---- Import Small Matrices(galaxy) ----
## Inspect the data(galaxy)
galaxy_sm<-read.csv("sentiment analysis/galaxy_smallmatrix_labeled_8d.csv")
summary(galaxy_sm)
str(galaxy_sm)
dim(galaxy_sm)


### ---- Principal Component Analysis ----
## iphone
preprocessParamsiphone <- preProcess(iphone_sm[,-59], 
                                     method=c("center", "scale", "pca"), 
                                     thresh = 0.95)
print(preprocessParamsiphone)
dim(preprocessParamsiphone)
#iphone_pca_df<-data.frame(iphone_sm$iphonesentiment,preprocessParamsiphone$)
# use predict to apply pca parameters, create training, exclude dependant
iphone_pca <- predict(preprocessParamsiphone, iphone_sm[,-59])
dim(iphone_pca)

# add the dependent to training
iphone_pca$iphonesentiment <- iphone_sm$iphonesentiment
head(iphone_pca)
str(iphone_pca)


## Inspect data types
iphone_pca$iphonesentiment <- as.factor(iphone_pca$iphonesentiment)
str(iphone_pca)

iphonesample_pca <- iphone_pca[sample(1:nrow(iphone_pca),
                                  1000, 
                                  replace = FALSE), ]

intrain_pca <- createDataPartition(y = iphonesample_pca$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain_pca <- iphonesample_pca[intrain_pca,]
iphonetest_pca <- iphonesample_pca[-intrain_pca,]


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
ctrl <- trainControl( method = "repeatedcv",   #? rfe control
                      repeats = 5)

a <- c("ranger", "C5.0", "kknn", "svmLinear")
compare_model_pca <- c()
for(i in a) {
  model <- train(iphonesentiment ~., data = iphonetrain_pca, method = i,trControl = ctrl)
  pred_pca <- predict(model, newdata = iphonetest_pca)
  pred_metric_pca <- postResample(iphonetest_pca$iphonesentiment, pred_pca)
  compare_model_pca <- cbind(compare_model_pca , pred_metric_pca)
}
colnames(compare_model_pca) <- a
compare_model_pca

#Structure table for plot
compare_model_melt_pca <- melt(compare_model_pca, varnames = c("metric", "model"))
compare_model_melt_pca <- as_data_frame(compare_model_melt_pca)
compare_model_melt_pca

#plot
ggplot(compare_model_melt_pca, aes(x=model, y=value))+
  geom_col()+
  facet_grid(metric~., scales="free")

########################using Large matric#########################

iphone_large<-read.csv("sentiment analysis/LargeMatrix.csv")
summary(iphone_large)
str(iphone_large)
dim(iphone_large)

preprocessParamsiphone_large <- preProcess(iphone_large, 
                                     method=c("center", "scale", "pca"), 
                                     thresh = 0.95)

iphone_pca_large <- predict(preprocessParamsiphone_large, iphone_large)
dim(iphone_pca_large)

# add the dependent to training
iphone_pca_large$iphonesentiment <- NA
head(iphone_pca_large)
str(iphone_pca_large)

iphone_pca_large$iphonesentiment <- as.factor(iphone_pca_large$iphonesentiment)
str(iphone_pca_large)

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 4

## Create cluster with desired number of cores.
cl <- makeCluster(3)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

model_DT <- train(iphonesentiment ~., 
               data = iphonetrain_pca, 
               method = "C5.0",
               trControl = ctrl)

p25<-names(iphonetrain_pca)
iphone_pca_large_1<-subset(iphone_pca_large,select=p25)
dim(iphone_pca_large_1)
dim(iphonetrain_pca)

pred_pca_DT <- predict(model_DT, newdata = iphone_pca_large_1,type = "raw")
summary(pred_pca_DT)

