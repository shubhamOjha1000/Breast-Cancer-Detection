from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import hyperparameter_tuning
import sklearn


def model1(criterion : str, max_depth : int ) -> sklearn.tree._classes.DecisionTreeClassifier :
    return DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)

def model2(criterion : str, max_depth : int ) -> sklearn.tree._classes.DecisionTreeClassifier :
    return DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)

def model3(n_estimators : int, random_state : int) -> sklearn.ensemble._forest.RandomForestClassifier :
    return RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

def model4() -> sklearn.naive_bayes.GaussianNB :
    return GaussianNB()

def model5(C : int, gamma : float) -> sklearn.svm._classes.SVC :
    return SVC(C = C, gamma = gamma)

def model6(n_neighbors : int) -> sklearn.neighbors._classification.KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors = n_neighbors)

def model7() -> sklearn.linear_model._logistic.LogisticRegression:
    return LogisticRegression()

def model8() -> sklearn.ensemble._weight_boosting.AdaBoostClassifier:
    return AdaBoostClassifier()

def model9(learning_rate : float, loss : str, n_estimators : int) -> sklearn.ensemble._gb.GradientBoostingClassifier :
    return GradientBoostingClassifier(learning_rate = learning_rate, loss = loss, n_estimators = n_estimators)

def model10(random_state : int, booster : str) -> xgb.sklearn.XGBClassifier:
    return xgb.XGBClassifier(random_state = random_state, booster = booster)
