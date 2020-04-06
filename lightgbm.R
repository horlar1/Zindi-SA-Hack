library(lightgbm)
library(caret)

devresult = rep(0,nrow(df_train))
predte = rep(0,nrow(df_test))
cvscore = c()
int.seed = c(500)
label2= label

for (i in 1:length(int.seed)) {
  cat("model training",i,"\n")
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)

  param = list(objective = "regression",
               metric = "rmse",
               boost_from_average = "false",
               #tree_learner = "serial",
               feature_fraction = 0.8,#0.7844718,#0.8,
               bagging_freq = 1,
               bagging_fraction = 0.8#0.7992016#
               #lambda = 1,
               #alpha = 1
              # max_bin = 63
              # min_data_in_leaf = 90,
              # min_sum_hessian_in_leaf = 10
              #min_split_gain = 0.1,
               )
  
  for (this.round in 1:length(folds)) {
    cat("model training",i," ","fold ",this.round,"\n")
    valid = c(1:length(label))[unlist(folds[this.round])]
    dev = c(1:length(label))[unlist(folds[1:length(folds)!= this.round])]
    
    dtrain = lgb.Dataset(data = as.matrix(df_train[dev,]),
      label = label2[dev], free_raw_data = F)
    dvalid = lgb.Dataset(data = as.matrix(df_train[valid,]),
      label = label2[valid],free_raw_data= F)
    
    model = lgb.train(data = dtrain,
                      params = param,
                      nrounds = 3000,
                      valids = list(val1 = dvalid, val2 = dtrain),
                      boosting_type = "gbdt",
                      learning_rate = 0.01,
                      max_depth = -1,
                      num_leaves = 20,
                      num_threads = 8,
                      eval_freq = 500,
                      seed = 1235,
                      verbose = 1,
                      early_stopping_rounds = 100
          )
    
pred = predict(model,as.matrix(df_train[valid,]))
devresult[valid] = pred
pred_test = predict(model, as.matrix(df_test[,colnames(df_train)]))
predte = predte + pred_test

cat("model cv rmse score:", model$best_score,"\n")
cvscore = c(cvscore, model$best_score)
cat("model cv rmse mean score:",mean(cvscore), "\n")
  }
}






