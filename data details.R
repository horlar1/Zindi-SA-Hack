options(warn = -1)

library(lightgbm)
library(reticulate)
library(plyr)
library(tidyverse)
library(caret)
library(SOAR)
library(data.table)
library(lubridate)
library(irlba)

#####
path.dir = getwd()
data.dir = paste0(path.dir,"/Data")

#dir.create(paste0(path.dir,"/subm"))
save.files.dir = paste0(path.dir,"/tmp")
subm.dir = paste0(path.dir,"/subm")
#cv.dir = paste0(path.dir,"/cv")
####
source("utils.R")



#####
df = fread(paste0(data.dir,"/Train_maskedv2.csv")) %>% as.data.frame()
test = fread(paste0(data.dir,"/Test_maskedv2.csv")) %>% as.data.frame()
############3
label = df$target_pct_vunerable
train.id = df$ward
test.id = test$ward

Store(train.id)
Store(test.id)
Store(label)

df = df %>% within(rm("ward","target_pct_vunerable"))
df = rbind(df,test[,colnames(df)])


#### FEATURE ENG
feat1 = round(df[,3:ncol(df)],2)
### CREATE SVD FEATURES
s = ssvd(as.matrix(cbind(df[,1:2],feat1)), k = 5,n =10,maxit = 1000)
colnames(s$u) = paste0("svd1_",1:ncol(s$u))
svd1 = data.frame(s$u)
save(svd1, file = paste0(save.files.dir,"/svdfeatures1.RData"))

#####
feat2 = round(df[,3:ncol(df)],1)
colnames(feat2) = paste0("feat2",1:ncol(feat2))
#### CREATE SVD FEATURES
set.seed(1234)
s = ssvd(as.matrix(feat2), k = 4,n =10,maxit = 1000)
colnames(s$u) = paste0("svd2_",1:ncol(s$u))
svd2 = data.frame(s$u)
save(svd2, file = paste0(save.files.dir,"/svdfeatures2.RData"))


#######
feat3 = df$total_households * df[,3:ncol(df)]
colnames(feat3) = paste0("feat3",1:ncol(feat3))
### CREATE SVD FEATURES
set.seed(1234)
s = ssvd(as.matrix(feat3), k = 4,n =10,maxit = 1000)
colnames(s$u) = paste0("svd3_",1:ncol(s$u))
svd3 = data.frame(s$u)
save(svd3, file = paste0(save.files.dir,"/svdfeatures3.RData"))


### FINAL DATA
df = data.frame(df[,1:2],feat1,svd1,feat2,svd2,feat3,svd3,
                label =c(label,rep(NA,nrow(test))))

### FEAT 4
### lln_01,dw_01, rnd24,psa_00,dw7,dw8
df = df %>% group_by(dw_08) %>% 
    mutate(dw_08_mean = mean(label,na.rm = T)) %>% ungroup() %>% 
    group_by(dw_07) %>% 
    mutate(dw_07_mean = mean(label,na.rm = T)) %>% ungroup() %>% 
    group_by(dw_01) %>% 
    mutate(dw_01_mean = mean(label,na.rm = T)) %>% ungroup() %>% 
    group_by(lln_01) %>% 
    mutate(lln_01_mean = mean(label,na.rm = T)) %>% ungroup() %>% 
    group_by(psa_00) %>% 
    mutate(psa_00_mean = mean(label,na.rm = T)) %>% ungroup()  
   
   



##### CHECK DATA PROPERTIES
ftrs = data.frame(
  type = unlist(lapply(df[1:length(train.id),],class)),
  n.unique = unlist(lapply(df[1:length(train.id),],function(x)length(unique(x)))),
  f.missing = unlist(lapply(df[1:length(train.id),],function(x)mean(is.na(x)))),
  spear.cor = unlist(lapply(df[1:length(train.id),],function(x){idx = !is.na(x);
  if(is.factor(x)) x = as.numeric(x);
  if(is.character(x)) x = as.numeric(as.factor(x));
  if(is.integer(x)) x = as.numeric(x);
  if(is.logical(x)) x = as.numeric(x);
  cor(x[idx],y = label[idx], method = "spearman")
  }))
)

## DROP NO VARIANCE
ftrs$name = rownames(ftrs)
ftrs =ftrs %>% drop_na()
df = df[,names(df) %in% ftrs$name]

########
## SPLIT DATA INTO TRAINING AND TEST DATA
#######
df_train = df[1:length(train.id),]
df_test = df[(length(train.id)+1):nrow(df),]

rm(df,test)

gc();gc()



## MODEL
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
    feature_fraction = 0.8,
    bagging_freq = 1,
    bagging_fraction = 0.8
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












