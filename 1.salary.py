import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def coef(x,y):
    if len(x)!=len(y):
        print("X and Y are not equal length")
    n=len(x)
    mean_x=np.mean(x)
    mean_y=np.mean(y)
    ss_xy = np.sum(y*x) - n*mean_x*mean_y
    ss_xx = np.sum(x*x) - n*mean_x*mean_x
    b1 = ss_xy / ss_xx
    b0 = mean_y - b1*mean_x
    return (b0, b1)
def plot_regression(x,y,x1,y_predict):
    plt.scatter(x, y,color="blue",marker="o",s=30)
    plt.scatter(x1, y_predict, color="red",marker="*",s=150 )
    y_pred=intercept+slope*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
df=pd.read_csv('Salary_dataset.csv')
print(df)
x = df['YearsExperience']
y = df['Salary']
intercept,slope=coef(x, y)
print("Intercept Value:",intercept)
print("Slope Value:",slope)
x1=int(input("Enter a x value : "))
intercept,slope=coef(x, y)
x=np.append(x,[x1])
y_predict=intercept+slope*x1
y=np.append(y,[y_predict])
print(y_predict)
plot_regression(x,y,x1,y_predict)
plt.show()

    
