#Load the data:
library(readr)
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime) 
library(tidyverse)
library(leaps)
library(MASS)
library(stargazer)

for (package in c('igraph', 'readr', 'statnet', 'countrycode', 'reporttools', 'ggplot2', 'stargazer', 'plm')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package, character.only=T)
  }
}

data <- read.csv("v5-sensitivity-NetLogo Model_v15_short_Dec 17.csv")

# for Exp 3:
pairs(~ initial.number.ethnos 
+initial.number.egos+initial.number.cosmos+initial.number.altruists+
  chance.coop.with.same + chance.coop.with.diff,   
      data=data, 
      main="Scatterplot test 5-10")

pairs(~ new.business + industry.wealth + total.mkt + max.mkt + min.mkt + mean.mkt + 
industry.growth.rate + total.merging + coop.merging + defect.merging, 
     data=data, 
      main="Scatterplot test 11-20")

pairs(~ average.mkt.growth 
      +mkt.egos+mkt.cosmos+mkt.altruists+ mkt.ethnos +
        merge.success + merge.failure,   
      data=data, 
      main="Scatterplot test 21-27")

pairs(~ ego.merge.success +ethno.merge.failure + ethno.merge.success
      +ego.merge.failure+cosmo.merge.success+cosmo.merge.failure+
        altruist.merge.success + altruist.merge.failure,   
      data=data, 
      main="Scatterplot test 28-35")



##Start with stepwise Regression:
# Fit the full model 
data_stepwise <- data[c(1,4:11, 16:17, 19:33, 35:42)]
full.model.merge.success <- lm(merge.success ~., data = data_stepwise)

#overall stats:
vars <- data[c(1,4:11, 16:17, 19:33, 35:42)]

tableContinuous(vars = vars, stats = c("n", "min", "mean", "median", 
                                       "max", "iqr", "na"), print.pval = "kruskal", 
                cap = "Table of continuous variables.", lab = "tab: descr stat")
# Stepwise regression model
step.model.success <- stepAIC(full.model.merge.success, direction = "both", 
                              trace = FALSE)
summary(step.model.success)

##Then do Gradient Boost Regression:

set.seed(123)
data_split <- initial_split(data_stepwise, prop = .7)
data_train <- training(data_split)
dat_test  <- testing(data_split)
# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = merge.failure ~ .,
  distribution = "gaussian",
  data = data_stepwise,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# print results
print(gbm.fit)
# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))
## this means on average our model is about 872.1289 off from the actual merge success, but error 
# is dcreasing over 10,000 iterations

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = merge.success ~ .,
  distribution = "gaussian",
  data = data_train,
  n.trees = 483,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.minobsinnode = 5,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

par(mar = c(4, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)


##Follow this to create regression tables in Stargazzer: 
library(arm)
library(car)

M1_general <- lm(formula = merge.success ~ X.run.number. + count.turtles + industry.wealth +
                   ethno.merge.success + ego.merge.success + cosmo.merge.success + 
                    + altruist.merge.success, 
                 data = data_stepwise)


M_ego.success <- lm(formula = ego.merge.success ~initial.number.ethnos + initial.number.egos + 
                      initial.number.cosmos + 
                      initial.number.altruists + 
                      penalty + boost + industry.wealth + firm.exit + ethno.merge.success + 
                      cosmo.merge.success + altruist.merge.success + X.run.number. + count.turtles, 
                    data = data_stepwise)

M_ethno.success <- lm(formula = ethno.merge.success ~initial.number.ethnos + initial.number.egos + 
                      initial.number.cosmos + 
                      initial.number.altruists + 
                      penalty + boost + industry.wealth + firm.exit + ego.merge.success + 
                      cosmo.merge.success + altruist.merge.success + X.run.number. + count.turtles, 
                    data = data_stepwise)

M_cosmo.success <- lm(formula = cosmo.merge.success ~initial.number.ethnos + initial.number.egos + 
                        initial.number.cosmos + 
                        initial.number.altruists + 
                        penalty + boost + industry.wealth + firm.exit + ego.merge.success + 
                        ethno.merge.success + altruist.merge.success + X.run.number. + count.turtles, 
                      data = data_stepwise)

M_altruist.success <- lm(formula =  altruist.merge.success~initial.number.ethnos + initial.number.egos + 
                        initial.number.cosmos + 
                        initial.number.altruists + 
                        penalty + boost + industry.wealth + firm.exit + ego.merge.success + 
                        ethno.merge.success + cosmo.merge.success + X.run.number. + count.turtles, 
                      data = data_stepwise)




coefplot(M_ego.success, xlim=c(-10, 10), intercept=TRUE)
coefplot(M_ethno.success, add=TRUE, col.pts="blue",  intercept=TRUE)
coefplot(M_altruist.success, add=TRUE, col.pts="green",  intercept=TRUE)
coefplot(M_cosmo.success, add=TRUE, col.pts="red",  intercept=TRUE)



#Export Regression Results: 
summary(M1_general)
stargazer(M1_general)
coefplot(M1_general, xlim=c(-0.5, 0.5), intercept=TRUE)


stargazer(M_ego.success, M_ethno.success, M_altruist.success, M_cosmo.success, title="Regression Results", align=TRUE)


