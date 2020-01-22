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
                                   "SN" = "N", 
                                   "VP" = "P", 
                                   "P" = "P", 
                                   "SP" = "P") 

galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, 
                                   "VN" = "N", 
                                   "N" = "N", 
                                   "SN" = "N", 
                                   "VP" = "P", 
                                   "P" = "P", 
                                   "SP" = "P") 



str(iphoneRC)
str(galaxyRC)

# Check for iphone
iphoneRC %>% 
  group_by(iphonesentiment) %>% 
  (count(iphonesentiment))
# so we found there is a class imbalance 

### ---- Sampling  ----
## iphone undersampling
set.seed(1234)
iphonedata_under <- ovun.sample(iphonesentiment~., 
                                data = iphoneRC, 
                                p = 0.5, 
                                seed = 1, 
                                method = "under")$data

iphonedata_under %>% 
  group_by(iphonesentiment) %>% 
  count(iphonesentiment)

# Check for galaxy 
galaxyRC %>% 
  group_by(galaxysentiment) %>% 
  count(galaxysentiment)

## galaxy oversampling(just consider oversampling )
set.seed(2345)
galaxydata_over <- ovun.sample(galaxysentiment~., 
                               data = galaxyRC, 
                               p = 0.5, 
                               seed = 1, 
                               method = "over")$data

galaxydata_over %>% 
  group_by(galaxysentiment) %>% 
  count(galaxysentiment)

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

