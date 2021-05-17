# Linear-Regression

Linear Regression gives the linear relationship between the set of input variables (x) and the single output variable (y). More specifically, that the target or Output variable (y) can be calculated from a linear combination of the input variables (x).

## Applying Linear Regression on Ecommerce  Dataset:

Initially we import the necessary libraries as follows:

```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
 `pandas` is used to import data and perform alalysis, `numpy` for mathematical operations,`matplotlib` & `seaborn` for data visualization.
 
## Exploratory Data Analysis:
After successfully loading the data with `df = pd.read_csv('Ecommerce Customers')` command, we get overview of the data by displaying the first 5 rows using `df.head()` or `df.sample(5)` to select randomly 5 rows. 

`.describe()` method gives the discriptive statastics of dataset's distribution excluding NaN (Non Numerical) values.
Using `.info()` method, we learn the shape of object types of our data and we see that there are 500 rows of data with no null values where 5 columns are of float datatype and 3 are Object datatype.

`sns.heatmap(df.corr(),annot=True)` gives the correlation of columns. 

![image](https://user-images.githubusercontent.com/64710293/118481016-4f710280-b713-11eb-833b-b4c7d4fd8c74.png)

From the above correlation heatmap we can interpret that 'Length of Membership' and 'Time on App' is 81% & 50% correlated with 'Yearly Amount Spent' respectively.

`sns.pairplot(df)` plots multiple pairwise bivariate distributions in the dataset. 

![pairplot](https://user-images.githubusercontent.com/64710293/118483289-1f772e80-b716-11eb-8776-5cb94fc4d6dd.png)

We could see a linear relation between 'Length of Membership' and 'Yearly Amount Spent' also between 'Time on App' and 'Yearly Amount Spent'.

As there are no missing values we can now proceed to Build the model.

## Training LinearRegression Model:

Firstly we split Features (X) and Lables (y) as below.

```
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
```

Then we split into test and train datasets using the following code 
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

``` 

Well now we are ready to fit the Linear Model using LinearRegression 

```
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
```
The training for our linear regression model is completed. The complexity of using more than one independent variable to predict a dependent variable by fitting a best linear relationship is carried out by `LinearRegression.fit(x_train,y_train)` method.

## Prediction: 
We predict the values for our testing set (x_test) and save it in the variable `y_pred ` 
```
y_pred = lr.predict(x_test)
```

We now check the accuracy score to check how well out model has performed.

```
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```
