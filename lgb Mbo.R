library(mlrMBO)
library(rBayesianOptimization)



int.seed = 1235

dtrain = lgb.Dataset(data.matrix(df_train), label = label, free_raw_data = F)

obj.fun = smoof::makeSingleObjectiveFunction(
  name = "lgb_cv_log",
  fn = function(x){
    set.seed(int.seed)
    cv = lgb.cv(
      params = list(
        objective = "regression",
        boost = "gbdt",
        metric = "rmse",
        boost_from_average = "false",
        num_leaves = x["num_leaves"],
        max_depth = x["max_depth"],
       # tree_learner = "serial",
        feature_fraction = x["feature_fraction"],
        bagging_freq = x["bagging_freq"],
        bagging_fraction = x["bagging_fraction"],
        #min_data_in_leaf = x["min_data_in_leaf"],
        #numclass = 4,
        #min_sum_hessian_in_leaf = x["min_sum_hessian_in_leaf"],
        verbose = 1,
        learning_rate = x["learning_rate"]
        ),
      data = dtrain,
      nrounds = 3000,
      folds = folds,
     #nfold = 5,
     # prediction = F,
      eval_freq = 1000,
      showsd = T,
      num_thread = 8,
      early_stopping_rounds = 100)
    cv$best_score
  },
  par.set = makeParamSet(
    makeDiscreteParam("learning_rate", 0.01),
    makeDiscreteParam("max_depth", -1),
    makeIntegerParam("num_leaves", lower = 5, upper = 20L),
    #makeIntegerParam("min_sum_hessian_in_leaf", lower = 5, upper = 10L),
    #makeIntegerParam("min_data_in_leaf", lower = 50,upper = 100L),
    makeIntegerParam("bagging_freq", lower = 1, upper = 5L),
    makeNumericParam("bagging_fraction", lower = 0.7, upper = 0.9),
    makeNumericParam("feature_fraction", lower = 0.4, upper = 0.9)
  ),
  minimize = TRUE
)

des = generateDesign(n = 10, par.set = getParamSet(obj.fun),
                     fun = lhs::randomLHS)

ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters = 10L)

print("Runing Bayesioan OPtimization on LightGBM")
run = mbo(fun = obj.fun,
          design = des,
          control = ctrl,
          show.info = T)