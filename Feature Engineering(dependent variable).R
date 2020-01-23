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
library(ROSE)
library(DMwR)


### ---- Import Ready Data ----
iphonedata <- readRDS("iphone_dataframe.rds")

galaxydata <- readRDS("galaxy_dataframe.rds")

### ---- Preprocessing ----
## Inspect data types
iphonedata$iphonesentiment <- as.factor(iphonedata$iphonesentiment)
str(iphonedata)

galaxydata$galaxysentiment <- as.factor(galaxydata$galaxysentiment)
str(galaxydata)

## Create a new dataset that will be used for recoding sentiment
iphoneRC <- iphonedata

galaxyRC <- galaxydata

## Recode sentiment to combine factor levels
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, 
                                   "VN" = "N", 
                                   "N" = "N", 
                                   "SN" = "SN", 
                                   "VP" = "VP", 
                                   "P" = "P", 
                                   "SP" = "P") 

galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, 
                                   "VN" = "N", 
                                   "N" = "N", 
                                   "SN" = "SN", 
                                   "VP" = "VP", 
                                   "P" = "P", 
                                   "SP" = "P") 



str(iphoneRC)
str(galaxyRC)
#############################################Over sampling and under sampling###################################
# Check for iphone
iphoneRC %>% 
  group_by(iphonesentiment) %>% 
  count(iphonesentiment)
# so we found there is a class imbalance and sensitivity and specificity difference is high

### ---- Sampling  ----
## iphone undersampling
set.seed(1234)
#iphonedata_under <- ovun.sample(iphonesentiment~., 
 #                               data = iphoneRC, 
#                                p = 0.5, 
 #                               seed = 1, 
 #                               method = "under")$data

#iphonedata_under %>% 
 # group_by(iphonesentiment) %>% 
 # count(iphonesentiment)

# Check for galaxy 
galaxyRC %>% 
  group_by(galaxysentiment) %>% 
  count(galaxysentiment)

## galaxy oversampling(just consider oversampling )
#set.seed(2345)
#galaxydata_over <- ovun.sample(galaxysentiment~., 
                  #             data = galaxyRC, 
                  #             p = 0.5, 
                  #             seed = 1, 
                  #             method = "over")$data

galaxydata_over %>% 
  group_by(galaxysentiment) %>% 
  count(galaxysentiment)
#############################################Over sampling and under sampling###################################

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(3)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

#### Data partition for iphone
set.seed(1234)
iphonesample_RC <- iphoneRC[sample(1:nrow(iphoneRC),
                                  1000, 
                                  replace = FALSE), ]
intrain_RC <- createDataPartition(y = iphonesample_RC$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain <- iphonesample[intrain1,]
iphonetest <- iphonesample[-intrain1,]

########## train and test  model for the iphone ###################


ctrl <- trainControl( method = "repeatedcv",   
                      repeats = 5,
                      sampling="smote")
a <- c("ranger", "C5.0", "kknn", "svmLinear")
compare_model <- c()
for(i in a) {
  model <- train(iphonesentiment ~., data = iphoneRC, method = i,trControl = ctrl)
  pred <- predict(model, newdata = iphonetest)
  pred_metric <- postResample(iphonetest$iphonesentiment, pred)
  compare_model <- cbind(compare_model , pred_metric)
}
colnames(compare_model) <- a
compare_model