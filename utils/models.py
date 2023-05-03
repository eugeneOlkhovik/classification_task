import pandas as pd
import plotly.figure_factory as ff
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaseModel:
    def __init__(self, model, params=None):
        """
        Initialize the model with its parameters
        """
        self.model = model(**params) if params else model()

    def fit(self, X, y):
        """
        Fit the model with the provided training data
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Use the trained model to predict the target variable
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates
        The returned estimates for all classes are
        ordered by the label of classes
        """
        return self.model.predict_proba(X)

    def score(self, X, y, scoring):
        """
        Evaluate the model with the selected scoring function
        """
        predictions = self.predict_proba(X)[:, 1]
        return scoring(y, predictions)


class LogisticRegressionModel(BaseModel):
    """
    LogisticRegressionModel is an extension of
    BaseModel specific to logistic regression
    """
    def __init__(self, params=None):
        super().__init__(LogisticRegression, params)


class RandomForestModel(BaseModel):
    """
    RandomForestModel is an extension
    of BaseModel specific to random forest
    """
    def __init__(self, params=None):
        super().__init__(RandomForestClassifier, params)

    def random_search(self, X, y, param_distributions,
                      n_iter=100,
                      cv=5,
                      scoring='roc_auc'):
        """
        Perform a randomized search for best hyperparameters
        """
        self.rs = RandomizedSearchCV(self.model,
                                     param_distributions=param_distributions,
                                     n_iter=n_iter,
                                     cv=cv,
                                     scoring=scoring)
        self.rs.fit(X, y)
        self.model = self.rs.best_estimator_


def compare_scores(score1, score2, model1_name, model2_name):
    """
    Function to compare the ROC-AUC scores of two models
    """
    df = pd.DataFrame({model1_name: [score1], model2_name: [score2]})
    fig = ff.create_annotated_heatmap(z=df.values,
                                      x=list(df.columns),
                                      y=['ROC-AUC'],
                                      colorscale='Viridis')
    fig.show()
