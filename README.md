The data can be found [here](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)

# Summary

## Item 1

* The WAPE of Exponential Smoothing and SARIMA models are extremely high and the models are not useful
* So I tried two simpler models: 1. last observed value and 2. average of last 12 months
* These last 12 month average model gave a lower WAPE than the 2 more complex models
* A seasonal effect was noticed, so I tested another model using last year's demand for a given month as the predictor
* This gave the best WAPE of 63% (still not great)

## Automation

Thousands of items exist, meaning that this will either be the longest analysis of all time, or something will have to become automated.

### Find best model

So, I wrote a function in `forecast_demand.py` to return WAPEs for each model warehouse-product combination using the following models:
  1. Last month only
  2. Average of last 12 months
  3. Last year's demand during the same month
  4. Exponential smoothing
  5. ARIMA without seasonality
  6. SARIMA with seasonality

This produces a table like

```
  Warehouse  Product_Code  Last_Month_WAPE  Last_12_WAPE  Seasonal_12_WAPE   \
0    Whse_A  Product_0001        88.501742     86.324042         62.891986   
1    Whse_A  Product_0020        59.602649     44.812362         92.715232   
2    Whse_A  Product_0031       132.894737     33.223684         61.184211   
3    Whse_A  Product_0075       100.000000    100.000000         44.444444   
4    Whse_A  Product_0078        89.385475     73.091248        158.100559   
5    Whse_A  Product_0081       100.000000     73.456790         85.802469   
6    Whse_A  Product_0091       100.000000     64.500000         93.000000   
7    Whse_A  Product_0097       125.587467    228.981723        287.467363   
8    Whse_A  Product_0119       100.000000     77.414075         73.322422   
9    Whse_A  Product_0120        74.161627     48.506505        117.262232 

     ETS_WAPE  ARIMA_WAPE  SARIMA_WAPE  
0   92.177241   94.555749    94.555749  
1   73.105632   63.038079    63.038079  
2   64.891516   68.585526    68.585526  
3  172.288048  180.299310   180.299310  
4  118.371294   90.513240    90.513240  
5   79.612212   59.953704    59.953704  
6  167.254572   65.374617    65.374617  
7  180.950958  287.467363    92.489306  
8   72.486596   81.044013    80.864817  
9   69.853059  117.133977    58.438703 
```

### Use best model

* The second function, `best_future_forecast`, uses the first to determine the best model, then selects and returns the model used + WAPE + forecast.
* The third function, `plot_product_demand_forecast`, plots the actual demand and next 12 months of forecasted demand

![Demand Forecast](https://github.com/atamalu123/demand_forecasting/blob/main/best_model_plot.png)
