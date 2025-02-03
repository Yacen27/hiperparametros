import pandas as pd


df = pd.read_csv("C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Busqueda_Hiperparametros/50_Startups.csv")

x = df.drop(["State", "Profit"], axis= 1).values
y = df["Profit"].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("RMSE: %.2f" % mean_squared_error(y_test, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print('R²: %.2f' % r2_score(y_test, y_pred))

# Ahora veremos los coeficientes del modelo
print(list(zip(df.columns, regressor.coef_)))



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

polynomial_regression = make_pipeline(
    PolynomialFeatures(),
    LinearRegression()
    )
############### CREACIÓN GRILLA ##################3

param_grid = {"polynomialfeatures__degree" : [2,3]} #### Grados a los que va a elevar el modelo

from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits = 10, shuffle=True, random_state= 77)

modelos_grid = GridSearchCV(polynomial_regression, param_grid, cv=kfold, n_jobs=1, scoring = "neg_root_mean_squared_error") 

modelos_grid.fit(x_train, y_train)

mejor_modelo = modelos_grid.best_params_
print(mejor_modelo)