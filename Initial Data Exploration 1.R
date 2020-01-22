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




# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

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

#plot_ly(iphone_sm, x= ~iphone_sm$iphonesentiment, type='histogram')

## Detect NA's
any(is.na(iphone_sm)) # Result = 0
any(is.na(galaxy_sm)) # Result = 0
sum(any(is.na(iphone_sm)))

## Increase max print 
options(max.print=1000000)

#Examine Correlation

## Check correlations for iphone
cor(iphone_sm)
ggcorr(iphone_sm)

## Check correlations for galaxy
cor(galaxy_sm)
ggcorr(galaxy_sm)


#Examine Feature Variance

#Features with no variance can be said to hold little to no information

iphone_metrics <- nearZeroVar(iphone_sm, saveMetrics = TRUE)
iphone_metrics

galaxy_metrics<-nearZeroVar(galaxy_sm, saveMetrics = TRUE)
galaxy_metrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
i_nzv <- nearZeroVar(iphone_sm, saveMetrics = FALSE) 
i_nzv

g_nzv<-nearZeroVar(galaxy_sm, saveMetrics = FALSE) 
g_nzv

# create a new data set and remove near zero variance features
iphone_NZV <- iphone_sm[,-i_nzv]
str(iphone_NZV)#After using zero variance feature reduced from 59 to 12 

galaxy_NZV<-galaxy_sm[,-g_nzv]
str(galaxy_NZV)

#Recursive Feature Elimination

# Let's sample the data before using RFE
set.seed(123)
iphone_Sample <- iphone_sm[sample(1:nrow(iphone_sm), 1000, replace=FALSE),]
galaxy_Sample <- galaxy_sm[sample(1:nrow(galaxy_sm), 1000, replace=FALSE),]
names(galaxy_Sample)
# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
# Use rfe and omit the response variable (attribute 59 iphonesentiment) 

i_rfe_Results <- rfe(iphone_Sample[,1:58], 
                     iphone_Sample$iphonesentiment, 
                     sizes=(1:58), 
                     rfeControl=ctrl)
# Get results
i_rfe_Results
# Plot results
plot(i_rfe_Results, type=c("g", "o"))
# create new data set with rfe recommended features
iphoneRFE <- iphone_sm[,predictors(i_rfe_Results)]
head(iphoneRFE)
# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone_sm$iphonesentiment
predictors(i_rfe_Results)
# review outcome
str(iphoneRFE)

g_rfe_Results<-rfe(galaxy_Sample[,1:58], 
                   galaxy_Sample$galaxysentiment, 
                   sizes=(1:58), 
                   rfeControl=ctrl)

# Get results
g_rfe_Results
# Plot results
plot(g_rfe_Results, type=c("g", "o"))

galaxyRFE <- galaxy_sm[,predictors(g_rfe_Results)]
head(galaxyRFE)
galaxyRFE$galaxysentiment <- galaxy_sm$galaxysentiment
str(galaxyRFE)
predictors(g_rfe_Results)

#Change target variable data type numeric to factor

iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)

galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)


### ---- Rename Levels of Factor ----
## 6 level factor
iphoneRFE %>%
  mutate(
    iphone_sentiment = 
      case_when(iphonesentiment %in% "0" ~ "VN",
                iphonesentiment %in% "1" ~ "N",
                iphonesentiment %in% "2" ~ "SN",
                iphonesentiment %in% "3" ~ "SP",
                iphonesentiment %in% "4" ~ "P",
                iphonesentiment %in% "5" ~ "VP")) -> iphoneRFE_ch_f
names(iphoneRFE_ch_f)

iphoneRFE_ch_f$iphonesentiment <- NULL #deleted feature which  contain with sentiment numeric value 

names(iphoneRFE_ch_f)[names(iphoneRFE_ch_f) == "iphone_sentiment"] <- "iphonesentiment"

galaxyRFE %>%
  mutate(
    galaxy_sentiment = 
      case_when(galaxysentiment %in% "0" ~ "VN",
                galaxysentiment %in% "1" ~ "N",
                galaxysentiment %in% "2" ~ "SN",
                galaxysentiment %in% "3" ~ "SP",
                galaxysentiment %in% "4" ~ "P",
                galaxysentiment %in% "5" ~ "VP")) -> galaxy_ch_f

galaxy_ch_f$galaxysentiment <- NULL

names(galaxy_ch_f)[names(galaxy_ch_f) == "galaxy_sentiment"] <- "galaxysentiment"
names(galaxy_ch_f)

### ---- Save Datasets for Modelization ----
saveRDS(iphoneRFE_ch_f, file = "iphone_dataframe.rds")

saveRDS(galaxy_ch_f, file = "galaxy_dataframe.rds")

## Stop Cluster
stopCluster(cl)
