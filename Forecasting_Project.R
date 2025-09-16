
install.packages("arrow")
install.packages("ggplot2") 
install.packages("caret")
library(caret)
library(arrow)
library(dplyr)
library(plm)
library(ggplot2)  
library(car)
library(whitestrap)
library(lmtest)
library(tseries)

#features were built in python
data<-read.csv("dataset.csv")

#remove stock 31
#trying different test stocks we noticed that the naive predictor obtains always an RMSPE between 0.2174 and 0.4519
#except for stock 31 RMSPE=1.039

RMSPE_vect<-numeric(126)
for (j in 1:126) {
  data_single_stock<-data[data$stock_id==j,]
  RMSPE=sqrt(mean(((data_single_stock$target-data_single_stock$sigma)/data_single_stock$target)^2, na.rm = TRUE))
    RMSPE_vect[j]=RMSPE
}
summary(RMSPE_vect)
RMSPE_naive=mean(RMSPE_vect ,na.rm=TRUE)
tail(sort(RMSPE_vect), 10)
which.max(RMSPE_vect)
data<-data[data$stock_id!=31,]
#RMSPE of the naive predictor is 0.3305
#take out a stock to test
set.seed(123)
test_stock <- sample(unique(data$stock_id), 1)
data_test<- data[data$stock_id %in% test_stock, ]

#the traning set is a random subset of 150 time ids for all the stocks (16650 data points)
#we will use the remaining time ids for testing and validation
all_tid <- unique(data$time_id)
training_tid <- sample(all_tid, 150)
remaining_tid <- setdiff(all_tid,training_tid)
half <- floor(length(remaining_tid) / 2)
validation_tid <- sample(remaining_tid, half)
test_tid <- setdiff(remaining_tid, validation_tid)
train_data <- data[data$time_id %in% training_tid, ]
data_test<- data[data$stock_id %in% test_stock & data$time_id %in% test_tid, ]


#Regressions loglog and standard

x1<-log(train_data$sigma)
model_log<-lm(log(train_data$target)~x1)
summary(model_log)
model<-lm(train_data$target~train_data$sigma)
train_predictions<-predict(model)
train_predictions_log<-exp(predict(model_log))
model_features<-lm(train_data$target~x1+train_data$wap+train_data$spread+train_data$log_return+train_data$volume_ob)
summary(model_features)
#besides from sigma, only wap resulted significant
model_features<-lm(train_data$target~x1+train_data$wap)
summary(model_features)


#testing hypothesis
res=model$residuals
fitted_val=model$fitted.value
res_log=model_log$residuals
fitted_val_log=model_log$fitted.values

white_test(model)
white_test(model_log)
bptest(model)
bptest(model_log)

qqnorm(res)
qqline(res)
jarque.bera.test(res)

qqnorm(res_log)
qqline(res_log)
jarque.bera.test(res_log)

#testing
sum<-0
count<-0
RMSPE_per_stock<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- model$coefficients[1]+data_testmod$sigma*model$coefficients[2]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock<- rbind(RMSPE_per_stock, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock)
#0.2960

#log
sum<-0
count<-0
RMSPE_per_stock_log<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(model_log$coefficients[1])*data_testmod$sigma^model_log$coefficients[2]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_log<- rbind(RMSPE_per_stock_log, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_log)

#0.2644
#hypotesis are not verified for both models, we try remooving outliers

#outliers
Q1 <- quantile( model$res, 0.25)
Q3 <- quantile( model$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(model$res<limite_inf|model$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_new <- train_data[-ids_to_remove, ]
model_no_outliers = lm(train_data_new$target~ train_data_new$sigma)
qqnorm( model_no_outliers$res, ylab = "Raw Residuals", pch = 16 )
qqline( model_no_outliers$res )
jarque.bera.test( model_no_outliers$res )
plot(model_no_outliers, which=1, main="Residual vs Fitted values ")
bptest(model_no_outliers)

#testing
sum<-0
count<-0
RMSPE_per_stock_no_outliers<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- model_no_outliers$coefficients[1]+data_testmod$sigma*model_no_outliers$coefficients[2]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_no_outliers<- rbind(RMSPE_per_stock_no_outliers, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_no_outliers)
#0.2775

#however, we are still not able to verify the hypothesis
#we try with log model

#outliers log
Q1 <- quantile( model_log$res, 0.25)
Q3 <- quantile( model_log$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(model_log$res<limite_inf|model_log$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_new_log <- train_data[-ids_to_remove, ]
model_no_outliers_log = lm(log(train_data_new_log$target)~ log(train_data_new_log$sigma))
qqnorm( model_no_outliers_log$res, ylab = "Raw Residuals", pch = 16 )
qqline( model_no_outliers_log$res )
jarque.bera.test( model_no_outliers_log$res )
plot(model_no_outliers_log, which=1, main="Residual vs Fitted values ")
bptest(model_no_outliers_log)
summary(model_no_outliers_log)
#hypothesis are satisfied

#testing
sum<-0
count<-0
RMSPE_per_stock_no_outliers_log<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(model_no_outliers_log$coefficients[1])*data_testmod$sigma^model_no_outliers_log$coefficients[2]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_no_outliers_log<- rbind(RMSPE_per_stock_no_outliers_log, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_no_outliers_log)
#0.2647


#from now on we will use "log" models without outliers (16361 data points out of 16650) train_data_new_log

#we try to build a fixed effect model on alpha
fixed_eff <- plm(log(target) ~ log(sigma), data = train_data_new_log, 
                 index = c("stock_id"), 
                 effect = "individual", model = "within")
pFtest(fixed_eff, model_no_outliers_log)
alpha_values <- fixef(fixed_eff)
random_eff<-plm(log(target) ~ log(sigma), data = train_data_new_log,
                index = c("stock_id"), 
                effect = "individual", model = "random")
#we test if also random effects are significant
phtest(fixed_eff, random_eff)

jarque.bera.test( fixed_eff$res )
summary(fixed_eff)
bptest(fixed_eff)
#hypothesis are verified

#testing 
sum<-0
count<-0
RMSPE_per_stock_fixef<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(alpha_values[as.character(i-1)])*data_testmod$sigma^fixed_eff$coefficients
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_fixef<- rbind(RMSPE_per_stock_fixef, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_fixef)
#0.2575        
 
#Weighted average of first and last five minutes sigma
#cross validation to establish the best weight parameter
 sum<-0
 count<-0
 vect=numeric(10)
 
for(j in 1:10) {
  data_cv<-data
  data_cv$sigma=j*0.1*data_cv$sigma_2.2+(1-j*0.1)*data_cv$sigma_1.2
  train_data_cv <- train_data_new_log
  train_data_cv$sigma=j*0.1*train_data_cv$sigma_2.2+(1-j*0.1)*train_data_cv$sigma_1.2
  fixed_eff_cv <- plm(log(target) ~ log(sigma), data = train_data_cv, 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
  alpha_cv <- fixef(fixed_eff_cv)
  
 for (i in 1:127) {
   validation_data_cv <-data_cv[data_cv$stock_id == (i-1) & data_cv$time_id %in% validation_tid,]
   prediction_cv <- exp(alpha_cv[as.character(i-1)])*validation_data_cv$sigma^fixed_eff_cv$coefficients
   RMSPE_cv=sqrt(mean((( validation_data_cv$target-prediction_cv)/validation_data_cv$target)^2, na.rm = TRUE))
   if (!is.nan(RMSPE_cv)) {
     sum <- sum + RMSPE_cv
     count<-count+1
   }}
  media=sum/count
  vect[j]=media
  media=0
  count=0
  sum=0
}
 vect
 which.min(vect)*0.1
 
 #the optimal value of j is 0.7
 data$sigma_weighted_2<-0.7*data$sigma_2.2+0.3*data$sigma_1.2
 train_data$sigma_weighted_2<-0.7*train_data$sigma_2.2+0.3*train_data$sigma_1.2
 
 fixed_eff_two <- plm(log(target) ~ log(sigma_weighted_2), data = train_data, 
                       index = c("stock_id"), 
                       effect = "individual", model = "within")
 alpha_values_two <- fixef(fixed_eff_two)
 jarque.bera.test(fixed_eff_two$residuals)
 bptest(fixed_eff_two)
 Q1 <- quantile( fixed_eff_two$res, 0.25)
 Q3 <- quantile( fixed_eff_two$res, 0.75)
 IQR_val <- Q3 - Q1
 limite_inf <- Q1 - 1.65* IQR_val
 limite_sup <- Q3 + 1.65 * IQR_val
 watchout_ids_out=which(fixed_eff_two$res<limite_inf|fixed_eff_two$res>limite_sup)
 ids_to_remove=unique(c(watchout_ids_out))
 train_data_two <- train_data[-ids_to_remove, ]
 fixed_eff_two<- plm(log(target)~ log(sigma_weighted_2), data=train_data_two , 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
 alpha_values_two <- fixef(fixed_eff_two)
 jarque.bera.test(fixed_eff_two$res)
 bptest(fixed_eff_two)
 summary(fixed_eff_two)
 
 
 #testing 
 sum<-0
 count<-0
 RMSPE_per_stock_fixef_two<- data.frame(stock = integer(), RMSPE = numeric())
 for (i in 1:127) {
   datamod<-data
   data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
   test_predictionsmod <- exp(alpha_values_two[as.character(i-1)])*data_testmod$sigma_weighted_2^fixed_eff_two$coefficients
   RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
   RMSPE_per_stock_fixef_two<- rbind(RMSPE_per_stock_fixef_two, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
   if (!is.nan(RMSPE_testmod)) {
     sum <- sum + RMSPE_testmod
     count<-count+1
   }}
 media=sum/count
 print(media)
 print(RMSPE_per_stock_fixef_two)
 #0.2522

#now we try a six steps exponential weighted average
#we use cross validation in order to estalish the best parameter
sum<-0
count<-0
vect=numeric(10)

for(j in 1:10) {
  data_cv<-data
  data_cv$sigma=((j*0.1)^4*data$sigma_2.6+(j*0.1)^5*data$sigma_1.6+(j*0.1)^3*data$sigma_3.6+(j*0.1)^2*data$sigma_4.6+(j*0.1)*data$sigma_5.6+data$sigma_6.6)/(j*0.1+(j*0.1)^3+(j*0.1)^4+(j*0.1)^5+(j*0.1)^2+1)
  train_data_cv<-train_data_new_log
  train_data_cv$sigma=((j*0.1)^4*train_data_cv$sigma_2.6+(j*0.1)^5*train_data_cv$sigma_1.6+(j*0.1)^3*train_data_cv$sigma_3.6+(j*0.1)^2*train_data_cv$sigma_4.6+(j*0.1)*train_data_cv$sigma_5.6+train_data_cv$sigma_6.6)/(j*0.1+(j*0.1)^3+(j*0.1)^4+(j*0.1)^5+(j*0.1)^2+1)
  fixed_eff_cv <- plm(log(target) ~ log(sigma), data = train_data_cv, 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
  alpha_cv <- fixef(fixed_eff_cv)
  for (i in 1:127) {
    validation_data_cv <-data_cv[data_cv$stock_id == (i-1) & data_cv$time_id %in% validation_tid,]
    prediction_cv <- exp(alpha_cv[as.character(i-1)])*validation_data_cv$sigma^fixed_eff_cv$coefficients
    RMSPE_cv=sqrt(mean((( validation_data_cv$target-prediction_cv)/validation_data_cv$target)^2, na.rm = TRUE))
    if (!is.nan(RMSPE_cv)) {
      sum <- sum + RMSPE_cv
      count<-count+1
    }}
  media=sum/count
  vect[j]=media
  media=0
  count=0
  sum=0
}
vect
which.min(vect)*0.1

data$sigma_weighted_6=(0.8^4*data$sigma_2.6+0.8^5*data$sigma_1.6+0.8^3*data$sigma_3.6+0.8^2*data$sigma_4.6+0.8*data$sigma_5.6+data$sigma_6.6)/(0.8+0.8^3+0.8^4+0.8^5+0.8^2+1)
train_data$sigma_weighted_6=(0.8^4*train_data$sigma_2.6+0.8^5*train_data$sigma_1.6+0.8^3*train_data$sigma_3.6+0.8^2*train_data$sigma_4.6+0.8*train_data$sigma_5.6+train_data$sigma_6.6)/(0.8+0.8^3+0.8^4+0.8^5+0.8^2+1)
#doing six steps exponential weighted average best parameter is 0.8

fixed_eff_six<- plm(log(target) ~ log(sigma_weighted_6), data = train_data, 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
alpha_values_six <- fixef(fixed_eff_six)
jarque.bera.test(fixed_eff_six$residuals)
bptest(fixed_eff_six)
Q1 <- quantile( fixed_eff_six$res, 0.25)
Q3 <- quantile( fixed_eff_six$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(fixed_eff_six$res<limite_inf|fixed_eff_six$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_six <- train_data[-ids_to_remove, ]
fixed_eff_six <- plm(log(target)~ log(sigma_weighted_6), data=train_data_six , 
                     index = c("stock_id"), 
                     effect = "individual", model = "within")
alpha_values_six <- fixef(fixed_eff_six)
jarque.bera.test(fixed_eff_six$res)
bptest(fixed_eff_six)
summary(fixed_eff_six)


#testing
sum<-0
count<-0
RMSPE_per_stock_fixef_six<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(alpha_values_six[as.character(i-1)])*data_testmod$sigma_weighted_6^fixed_eff_six$coefficients
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_fixef_six<- rbind(RMSPE_per_stock_fixef_six, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_fixef_six)
#0.2487 

 

#Adding Kurtosis
fixed_eff_kurt <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis), data=train_data , 
                 index = c("stock_id"), 
                 effect = "individual", model = "within")
summary(fixed_eff_kurt)
jarque.bera.test(fixed_eff_kurt$res)
Q1 <- quantile( fixed_eff_kurt$res, 0.25)
Q3 <- quantile( fixed_eff_kurt$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(fixed_eff_kurt$res<limite_inf|fixed_eff_kurt$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_kurt <- train_data[-ids_to_remove, ]
fixed_eff_kurt <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis), data=train_data_kurt , 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
alpha_values_kurt <- fixef(fixed_eff_kurt)
jarque.bera.test(fixed_eff_kurt$res)
summary(fixed_eff_kurt)


#testing
sum<-0
count<-0
RMSPE_per_stock_fixef_kurt<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(alpha_values_kurt[as.character(i-1)])*data_testmod$sigma_weighted_6^fixed_eff_kurt$coefficients[1]*data_testmod$kurtosis^fixed_eff_kurt$coefficients[2]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_fixef_kurt<- rbind(RMSPE_per_stock_fixef_kurt, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_fixef_kurt)
quantile(RMSPE_per_stock_fixef_kurt$RMSPE, na.rm=TRUE)
#0.24712


#Adding GJR prediction
fixed_eff_GJR <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis)+log(GJR), data=train_data , 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
summary(fixed_eff_GJR)
jarque.bera.test(fixed_eff_GJR$res)
Q1 <- quantile( fixed_eff_GJR$res, 0.25)
Q3 <- quantile( fixed_eff_GJR$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(fixed_eff_GJR$res<limite_inf|fixed_eff_GJR$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_GJR <- train_data[-ids_to_remove, ]
fixed_eff_GJR <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis)+log(GJR), data=train_data_GJR , 
                      index = c("stock_id"), 
                      effect = "individual", model = "within")
alpha_values_GJR <- fixef(fixed_eff_GJR)
jarque.bera.test(fixed_eff_GJR$res)
#normality is not veirfied
summary(fixed_eff_GJR)

#testing
sum<-0
count<-0
RMSPE_per_stock_fixef_GJR<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(alpha_values_GJR[as.character(i-1)])*data_testmod$sigma_weighted_6^fixed_eff_GJR$coefficients[1]*data_testmod$kurtosis^fixed_eff_GJR$coefficients[2]*data_testmod$GJR^fixed_eff_GJR$coefficients[3]
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_fixef_GJR<- rbind(RMSPE_per_stock_fixef_GJR, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_fixef_GJR)
#0.2478
#no improvements


#Adding WAP
fixed_eff_wap <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis)+wap, data=train_data , 
                     index = c("stock_id"), 
                     effect = "individual", model = "within")
summary(fixed_eff_wap)
jarque.bera.test(fixed_eff_wap$res)
Q1 <- quantile( fixed_eff_wap$res, 0.25)
Q3 <- quantile( fixed_eff_wap$res, 0.75)
IQR_val <- Q3 - Q1
limite_inf <- Q1 - 1.65* IQR_val
limite_sup <- Q3 + 1.65 * IQR_val
watchout_ids_out=which(fixed_eff_wap$res<limite_inf|fixed_eff_wap$res>limite_sup)
ids_to_remove=unique(c(watchout_ids_out))
train_data_wap <- train_data[-ids_to_remove, ]
fixed_eff_wap <- plm(log(target)~ log(sigma_weighted_6)+log(kurtosis)+wap, data=train_data_wap , 
                     index = c("stock_id"), 
                     effect = "individual", model = "within")
alpha_values_wap <- fixef(fixed_eff_wap)
jarque.bera.test(fixed_eff_wap$res)
summary(fixed_eff_wap)



sum<-0
count<-0
RMSPE_per_stock_fixef_wap<- data.frame(stock = integer(), RMSPE = numeric())
for (i in 1:127) {
  datamod<-data
  data_testmod <- datamod[datamod$stock_id == (i-1) & data$time_id %in% test_tid,]
  test_predictionsmod <- exp(alpha_values_wap[as.character(i-1)])*data_testmod$sigma_weighted_6^fixed_eff_wap$coefficients[1]*data_testmod$kurtosis^fixed_eff_wap$coefficients[2]*exp(data_testmod$wap*fixed_eff_wap$coefficients[3])
  RMSPE_testmod=sqrt(mean(((data_testmod$target-test_predictionsmod)/data_testmod$target)^2, na.rm = TRUE))
  RMSPE_per_stock_fixef_wap<- rbind(RMSPE_per_stock_fixef_wap, data.frame(stock = i-1, RMSPE = RMSPE_testmod))
  if (!is.nan(RMSPE_testmod)) {
    sum <- sum + RMSPE_testmod
    count<-count+1
  }}
media=sum/count
print(media)
print(RMSPE_per_stock_fixef_wap)
#0.24699
#very small improvement

