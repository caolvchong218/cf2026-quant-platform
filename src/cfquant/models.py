"""A fixed, small model comparison; no final-period parameter search."""
import numpy as np


def fit_predict(name, x_train, y_train, x_predict, seed=20260921):
    if name=='ridge':
        from sklearn.linear_model import Ridge
        model=Ridge(alpha=100.)
    elif name=='lightgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(n_estimators=250,learning_rate=.03,num_leaves=15,max_depth=5,
                            min_child_samples=500,colsample_bytree=.8,reg_lambda=10.,
                            random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
    elif name=='mlp':
        from sklearn.neural_network import MLPRegressor
        from threadpoolctl import threadpool_limits
        model=MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',solver='adam',
                           alpha=.01,batch_size=4096,learning_rate_init=.001,
                           max_iter=8,early_stopping=False,random_state=seed)
        with threadpool_limits(limits=4):
            model.fit(x_train,y_train)
            result=model.predict(x_predict)
        return result,{'seed':seed,'epochs':int(model.n_iter_),'training_loss':model.loss_curve_,
                       'architecture':[x_train.shape[1],32,16,1],
                       'note':'Fixed epoch budget; convergence is not assumed.'}
    else:raise ValueError(name)
    model.fit(x_train,y_train)
    predicted=model.predict(x_predict)
    importance=(model.feature_importances_ if hasattr(model,'feature_importances_') else model.coef_)
    return predicted,{'seed':seed,'parameters':model.get_params(),'importance':np.asarray(importance).tolist()}
