# load the preprocessed data
load("Homesite.RData")

# convert NAs to 0
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

# select features for the model
features <- names(train)[c(3:301)]

# transform all the text variables to numeric 
for (f in features) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

# select data for model training
train<-train[,features]

# determine the loss function
RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

# use xgboost package to build the model
library(xgboost)
# select the number of samples to do validation
h <- sample(nrow(train), 2000)
# seperate the train and validation sample and transform data to XGBoost Matrix
validate <-xgb.DMatrix(data=data.matrix(train[h,]),label=logSales[h])
train <-xgb.DMatrix(data=data.matrix(train[-h,]),label=loglogSales[-h])
# set up parameters for the model
watchlist<-list(val=validate,train=train)
param1 <- list(  objective           = "reg:linear", 
                 booster = "gbtree",
                 eta                 = 0.02, 
                 max_depth           = 10, 
                 subsample           = 0.9, 
                 colsample_bytree    = 0.7, 
                 num_parallel_tree   = 2,
                 alpha = 0.0001, 
                 lambda = 1)
# model training
xgb <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3000, 
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE)
# predicting in test data
predictions <- predict(xgb, data.matrix(test[,features]))
# generate submission data
submission <- data.frame(Id=test$Id, Sales=pred1)
# save predicted data
write.csv(submission, "xgb_param1_3000.csv",row.names=F)
# save model
save(rf_hex, file="xgb_param1_3000.RData")