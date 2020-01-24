

iphoneRC 

galaxyRC 

# load the model for iphone
iphone_model_RF <- readRDS("final_model_RF_iphone.rds")
print(iphone_model_RF)


# load the model for iphone
galaxy_model_RF <- readRDS("final_model_RF_galaxy.rds")
print(galaxy_model_RF)

# load large Dataset
largematrix<-read.csv("sentiment analysis/LargeMatrix.csv")
summary(largematrix)
str(largematrix)
dim(largematrix)

#iphone prediction
pred_iphone_final <- predict(iphone_model_RF, newdata = largematrix,type = "raw")
summary(pred_iphone_final)

# probability comparison
prop.table(table(pred_iphone_final))
prop.table(table(iphoneRC$iphonesentiment))

#galaxy prediction
pred_galaxy_final <- predict(galaxy_model_RF, newdata = largematrix,type = "raw")
summary(pred_galaxy_final)

# probability comparison
prop.table(table(pred_galaxy_final))
prop.table(table(galaxyRC$galaxysentiment))
