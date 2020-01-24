


# load the model for iphone
iphone_model_RF <- readRDS("final_model_RF_iphone.rds")
print(iphone_model_RF)


# load the model for iphone
galaxy_model_RF <- readRDS("final_model_RF_galaxy.rds")
print(galaxy_model_RF)

# load large Dataset
largematrix<-read.csv("sentiment analysis/LargeMatrix.csv")
