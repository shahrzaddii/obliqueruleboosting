import numpy as np
from sklearn.linear_model import LogisticRegression, ridge_regression, Ridge
from sklearn.base import BaseEstimator
from scipy.special import expit
from sklearn.metrics import log_loss, mean_squared_error, zero_one_loss
import pandas as pd
from dataset import BootstrapSplitter
from sklearn.model_selection import train_test_split, KFold, cross_validate
import copy
import time


def expit_bnd(x):
    x_clipped = np.clip(x, -36, 36)
    return expit(x_clipped)


class L1LogisticRegression(BaseEstimator):
    """ Logistic Regression with L1 penalty for desired sparsity by using binary search to find the optimal C value. """
    def __init__(self, max_num_nonzero_coef=100, C_range=(0, 1e4), max_iter=1000,
                 tol=1e-4, fit_intercept=True, max_binary_search_iter=10000, l2_reg=1e-30, no_refit=False):
        self.max_iter = max_iter
        self.tol = tol
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.C_range = C_range
        self.fit_intercept = fit_intercept
        self.max_binary_search_iter = max_binary_search_iter
        self.l2_reg = l2_reg
        self.no_refit = no_refit

    def upper_bound_binary_search(self, x, y, g):
        high_sparse = False
        bound = 1 
        while(high_sparse is False):
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                C=bound
            ).fit(x, y, sample_weight=g)
            sparsity = np.count_nonzero(model.coef_[0])
            if sparsity >= self.max_num_nonzero_coef:
                high_sparse = True
            else:
                bound = bound * 2
        return bound, sparsity, model
    
    def refit(self, x, y, g=None):
        model = LogisticRegression(
                    penalty='l2',
                    solver='newton-cg',
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    C=1/self.l2_reg
            ).fit(x, y, sample_weight=g)
        
        return model
    
    def fit(self, x, y, g=None):
        refit_model = None
        if g is None:
            g = np.ones_like(y)
        low = self.C_range[0]
        self.coef_ = np.zeros((x.shape[1]))
        high, sparsity, model = self.upper_bound_binary_search(x, y, g)
        coef_backup = np.zeros((x.shape[1]))
        if sparsity == self.max_num_nonzero_coef:
            nz_coef = np.nonzero(model.coef_[0])[0]
            x_sparse = x[:,nz_coef]
            refit_model = self.refit(x_sparse, y, g)
            self.coef_[nz_coef] = refit_model.coef_[0]
            self.intercept_ = refit_model.intercept_[0]
            self.C = high
            self.l1_coef_ = model.coef_[0]
            self.l1_intercept_ = model.intercept_[0]
        else:
            for _ in range(self.max_binary_search_iter):
                mid = (low + high) / 2
                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    C=mid
                ).fit(x, y, sample_weight=g)
                nonzero_count = np.count_nonzero(model.coef_[0])

                if nonzero_count == self.max_num_nonzero_coef:
                    nz_coef = np.nonzero(model.coef_[0])[0]
                    x_sparse = x[:,nz_coef]
                    refit_model = self.refit(x_sparse, y, g)
                    self.coef_[nz_coef] = refit_model.coef_[0]
                    self.intercept_ = refit_model.intercept_[0]
                    self.C = mid
                    self.l1_coef_ = model.coef_[0]
                    self.l1_intercept_ = model.intercept_[0]
                    break
                elif nonzero_count > self.max_num_nonzero_coef:
                    high = mid
                else:
                    low = mid

                if nonzero_count != 0:
                    nz_coef_backup = np.nonzero(model.coef_.reshape(1,-1)[0])[0]
                    x_sparse_backup = x[:,nz_coef_backup]
                    refit_model_backup = self.refit(x_sparse_backup, y, g)
                    coef_backup[nz_coef_backup] = refit_model_backup.coef_.reshape(1,-1)[0]
                    intercept_backup = np.array(refit_model_backup.intercept_).reshape(1,-1)[0]
                    C_backup = mid
                    l1_coef_backup = model.coef_[0]
                    l1_intercept_backup = model.intercept_[0]

        if refit_model is None:
            self.refit_model = refit_model_backup
            self.coef_ = coef_backup
            self.intercept_ = intercept_backup
            self.C = C_backup
            self.l1_coef_ = l1_coef_backup
            self.l1_intercept_ = l1_intercept_backup

        if self.no_refit is True:
            self.coef_ = self.l1_coef_
            self.intercept_ = self.l1_intercept_

        return self

class Proposition(BaseEstimator):
    def __init__(self, sparsity=500, no_refit=False, task='linreg'):
        self.sparsity = sparsity
        self.no_refit = no_refit
        self.no_improvement = False
        self.task = task
    
    def fit(self, x, y, g_class, g=None):
        if g is None:
            g = np.ones_like(g_class)
        self.prop = L1LogisticRegression(max_num_nonzero_coef=self.sparsity, no_refit=self.no_refit).fit(x=x,y=g_class,g=abs(g))
        return self

class LearnRule():
    def __init__(self, max_complexity=10, epsilon_r=0.1, weighted_logloss = True, no_refit=False, task = 'linreg'):
        self.epsilon_r = epsilon_r
        self.max_complexity = max_complexity
        self.rng = np.random.default_rng()
        self.weighted_logloss = weighted_logloss
        self.no_refit = no_refit
        self.task = task

    def gradient_sum_obj(self, g, ext):
        if len(ext)==0:
            return -np.inf
        return sum(g[ext])
    
    def fit(self, x, y, g, rulens_pred_fn=None):
        best_valid_risk = np.inf
        best_obj = -np.inf
        n, d = x.shape
        best_w = np.zeros((d, 0))
        best_t = np.zeros(0)
        best_z = 1
        for z in [1,-1]:
            w, t, obj, valid_risk = self.learn_conjunction(x, y, g, z, rulens_pred_fn)
            if valid_risk < best_valid_risk:
                best_w = w
                best_t = t
                best_obj = obj
                best_valid_risk = valid_risk
                best_z = z

        self.w_ = best_z*best_w
        self.t_ = best_z*best_t
        self.obj_ = best_obj
        self.best_valid_risk_ = best_valid_risk

        return self

    def learn_conjunction(self, x, y, g, z, rulens_pred_fn=None):
        g = g*z
        complexity_level = 0
        configurations = {0: []}
        best_risk = np.inf
        best_config = None
        prop_sparsity = 0
        _, d = x.shape
        while complexity_level < self.max_complexity:
            next_config = []
            w = np.zeros((d, self.max_complexity))
            t = np.zeros(self.max_complexity)
            prop = configurations[complexity_level]

            if len(prop) > 0:
                num_learned_props = len(prop[1])
            else:
                num_learned_props = 0

            if num_learned_props < self.max_complexity:
                num_to_learn_props = num_learned_props + 1

            # initialising the variables
            best_obj = -np.inf
            updated_prop_obj = self.gradient_sum_obj(g, self._predict(x, w, t).astype(bool))
            updated_prop_val_risk = np.inf
            for i in range(num_to_learn_props):
                if len(prop) > 0:
                    w[:, :num_learned_props] = prop[0]
                    t[:num_learned_props] = prop[1]
                prop_sparsity = np.count_nonzero(w[:,i])
                if prop_sparsity < d:
                    w_fixed = np.concatenate((w[:, :i], w[:, i+1:num_to_learn_props]), axis=1)
                    t_fixed = np.concatenate((t[:i], t[i+1:num_to_learn_props]), axis=0)
                    selected = self._predict(x, w_fixed, t_fixed).astype(bool)
                    if not (len(g[selected])==sum(g[selected]>=0) or len(g[selected])==sum(g[selected]<0)):
                        updated_prop = Proposition(sparsity=prop_sparsity+1, no_refit=self.no_refit, task=self.task).fit(x=x[selected], y=y[selected], g_class=g[selected]>=0, g=g[selected])
                        w[:,i] = updated_prop.prop.coef_
                        t[i] = updated_prop.prop.intercept_
                        new_selected = self._predict(x, w[:,:num_to_learn_props], t[:num_to_learn_props]).astype(bool)
                        updated_prop_obj = self.gradient_sum_obj(g, new_selected)

                        self.w_ = z*w[:,:num_to_learn_props]
                        self.t_ = z*t[:num_to_learn_props]
                        updated_prop_val_risk = rulens_pred_fn(self)

                    if updated_prop_obj > best_obj:
                        w_sel = np.copy(w[:,:num_to_learn_props])
                        t_sel = np.copy(t[:num_to_learn_props])
                        best_obj = updated_prop_obj
                        if np.count_nonzero(w_sel[:,-1]) == 0:
                            w_sel = w_sel[:,:-1]
                            t_sel = t_sel[:-1]
                        next_config = [w_sel, t_sel, updated_prop_obj, updated_prop_val_risk]

            configurations[complexity_level + 1] = next_config
            rule_complexity = np.count_nonzero(next_config[0])

            if next_config[3] < best_risk*(1-self.epsilon_r):
                best_risk = next_config[3]
                best_config = next_config
            complexity_level += 1

        if best_config is None:
            w_sel = np.zeros((d, 0))
            t_sel = np.zeros(0)
            obj = -np.inf
            valid_risk = np.inf
        else:
            w_sel, t_sel, obj, valid_risk = best_config
        
        return w_sel, t_sel, obj, valid_risk

    
    @staticmethod
    def _predict(x,w,t):
        l = x@w
        s = l>=-t
        return np.prod(s, axis=1).astype(bool)

    def predict(self, x):
        return self._predict(x, self.w_, self.t_)


class LinRepRuleEnsemble(BaseEstimator):

    def __init__(self, num_rules=3, epsilon_r=0.1, max_complexity=10, reg = 1e-10,
                task = 'linreg', weighted_logloss = True, no_refit=False):
        self.epsilon_r = epsilon_r
        self.num_rules = num_rules
        self.task = task
        self.weighted_logloss = weighted_logloss
        self.max_complexity = max_complexity
        self.conditions_ = []
        self.reg = reg
        self.no_refit = no_refit
        self.ensembles_ = []

    def gradient(self, x, y):
        return y - self.predict(x)
        
    def first_background_rule(self, y):
        if self.task == 'linreg':
            return np.array([y.mean()])
        else:
            y_bar = np.array([y.mean()])
            return np.log(y_bar/(1-y_bar))

    def fit(self, x,y):
        self.feature_name = x.columns
        x_original = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        y_original = y.to_numpy() if isinstance(y, pd.Series) else y

        ### Fixed validation set ###
        # splitter = BootstrapSplitter(reps=1, train_size=len(x_original), replace=True, random_state=42)
        # for train_idx, valid_idx in splitter.split(x_original, y_original):
        #     x_valid, y_valid = x_original[valid_idx], y_original[valid_idx]
        #     x, y = x_original[train_idx], y_original[train_idx]

        self.weights_ = self.first_background_rule(y)
        self.time_elapsed = 0
        for _ in range(self.num_rules):
            start_time_one_iteration = time.time()
            ### Variable validation set ###
            splitter = BootstrapSplitter(reps=1, train_size=len(x_original), replace=True, random_state=None)
            for train_idx, valid_idx in splitter.split(x_original, y_original):
                x_valid, y_valid = x_original[valid_idx], y_original[valid_idx]
                x, y = x_original[train_idx], y_original[train_idx]

            self.valid_idx_to_out = valid_idx
            self.train_idx_to_out = train_idx
                
            g = self.gradient(x, y)
            self.gradient_valid_to_out = self.gradient(x_valid, y_valid)
            self.gradient_train_to_out = g
            def predict_fn(temp_condition):
                self.conditions_ = self.conditions_ + [temp_condition]
                res = self.compute_C(x)
                self.estimate_weights(res,y,reg=self.reg)
                valid_pred = self.predict(x_valid)
                if self.task == 'linreg':
                    valid_loss = mean_squared_error(y_valid, valid_pred)
                else:
                    valid_loss = log_loss(y_valid, valid_pred, labels=[0,1])
                self.conditions_.pop()
                return valid_loss

            if _ == 0:
                self.g_to_out = g

            rule = LearnRule(epsilon_r= self.epsilon_r, weighted_logloss=self.weighted_logloss,
                                          max_complexity=self.max_complexity, no_refit=self.no_refit, task = self.task).fit(x,y,g,predict_fn)
            self.conditions_.append(rule)
            c = self.compute_C(x)
            self.estimate_weights(c,y, reg=self.reg)
            end_time_one_iteration = time.time()
            self.time_elapsed = self.time_elapsed + (end_time_one_iteration - start_time_one_iteration)

            copy_model = copy.deepcopy(self)
            self.ensembles_.append(copy_model)
        return self
    
    def estimate_weights(self, c, y, reg):
        if self.task == 'linreg':
            linr = Ridge(alpha=reg, fit_intercept=False).fit(c,y)
            self.weights_ = linr.coef_
        elif self.task == 'logreg':
            logr = LogisticRegression(fit_intercept=False, solver='newton-cg', penalty='l2', C=1/reg).fit(c,y)
            self.weights_ = logr.coef_[0]
        return self

    
    def compute_C(self, x):
        res = np.zeros(shape=(len(x), len(self.conditions_)+1))
        res[:,0] = 1
        for i, c in enumerate(self.conditions_):
            res[:, i+1] = c.predict(x)
        return res

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        pred = self.compute_C(x)
        if pred.shape[1] != len(self.weights_):
            print(f'pre shape is {pred.shape} weight vector shape is {self.weights_.shape} and x shape is {x.shape}')
        if self.task == 'linreg':
            out = pred.dot(self.weights_)
        else:
            out = expit_bnd(pred.dot(self.weights_))
        return out
    
    def num_of_rules(self):
        return len(self.conditions_)
    
    def model_complexity(self):
        return self.num_of_rules() + sum([c.w_.shape[1] for c in self.conditions_]) + sum([np.count_nonzero(c.w_) for c in self.conditions_])
    
    def rules_complexity(self):
        return sum([np.count_nonzero(c.w_) for c in self.conditions_])
    
    def rules(self, y_mu=None, y_sigma=None, x_mu=None, x_sigma=None):
        rule_w = self.weights_
        if y_mu is not None and y_sigma is not None:
            rule_w = np.round(np.hstack([np.array([self.weights_[0]*y_sigma + y_mu]), self.weights_[1:]*y_sigma]),2)
        rule_expression = [f"{rule_w[0]:.3f}"]
        for r_i, r in enumerate(self.conditions_):
            w = r.w_
            t = r.t_
            if x_mu is not None and x_sigma is not None:
                w = w.T / x_sigma
                t = t - np.dot(w, x_mu)
                w = w.T

            t = t/np.max(np.abs(w), axis=0)
            w = w/np.max(np.abs(w), axis=0)
            propos_expression = []
            for p_i, p in enumerate(range(w.shape[1])):
                terms = []
                for i in range(len(self.feature_name)):
                    weight = w[i,p_i]
                    if weight != 0:
                        sign = '+' if weight > 0 and len(terms) > 0 else ''
                        if weight == 1:
                            terms.append(f"{sign}{self.feature_name[i]}")
                        elif weight == -1:
                            terms.append(f"{'-'}{self.feature_name[i]}")
                        else:
                            terms.append(f"{sign}{weight:.5f} {self.feature_name[i]}")
                if w.shape[1] > 1 and p_i != w.shape[1]-1:
                    and_ = ' & '
                else:
                    and_ = ''
                propos_expression.append(f" ".join(terms) + f" ≥ {-t[p_i]:.3f}" + and_)

            rule_expression.append(f"{rule_w[r_i+1]:.3f} if " + " ".join(propos_expression))

        return rule_expression
    
