from sklearn.ensemble import GradientBoostingClassifier

def train_two_models(X, treat, y):
    model_t = GradientBoostingClassifier()
    model_c = GradientBoostingClassifier()
    model_t.fit(X[treat==1], y[treat==1])
    model_c.fit(X[treat==0], y[treat==0])
    return model_t, model_c

def predict_uplift(model_t, model_c, X):
    p_t = model_t.predict_proba(X)[:,1]
    p_c = model_c.predict_proba(X)[:,1]
    return p_t - p_c
