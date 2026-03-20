import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Tuple

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from obliquetree import Classifier as ObliqueClassifier, Regressor as ObliqueRegressor
from obliquetree.utils import export_tree
from sklearn.base import BaseEstimator
from scipy.special import expit
import pandas as pd
import math



def scale_combinations(d, max_s, threshold=1000, shrink=0.8):
    sel_max_s = []
    sel_d_frac = []

    for p in range(2,max_s+1):
        comb_val = math.comb(d, p)
        if comb_val < threshold:
            sel_max_s.append(p)
            sel_d_frac.append(1.0)
            continue

        new_d = d
        while comb_val >= threshold and new_d > 1:
            new_d = max(1, int(new_d * shrink))
            comb_val = math.comb(new_d, p)

        sel_max_s.append(p)
        sel_d_frac.append(new_d / d)

    return sel_max_s, sel_d_frac


class L1Optimization(BaseEstimator):
    """ Logistic Regression with L1 penalty for desired sparsity by using binary search to find the optimal C value. """
    def __init__(self, task='classification' , max_num_nonzero_coef=100, C_range=(0, 1e4), max_iter=500,
                 tol=1e-4, fit_intercept=True, seed=None, max_binary_search_iter=1000, l2_reg=1e-10):
        self.max_iter = max_iter
        self.tol = tol
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.C_range = C_range
        self.fit_intercept = fit_intercept
        self.max_binary_search_iter = max_binary_search_iter
        self.task = task
        self.random_state = seed
        self.refit_model = None
        self.l2_reg = l2_reg

    def base_func(self, C):
        return LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                C=C) if self.task == 'classification' else Lasso(alpha=C, max_iter=self.max_iter, random_state=self.random_state, fit_intercept=self.fit_intercept)
    
    def upper_bound_binary_search(self, x, y):
        high_sparse = False
        bound = 1
        while(high_sparse is False):
            model = self.base_func(C=bound).fit(x, y)
            sparsity = np.count_nonzero(model.coef_)
            if self.task == 'classification':
                if sparsity >= self.max_num_nonzero_coef:
                    high_sparse = True
                else:
                    bound = bound * 2
            else:
                if sparsity <= self.max_num_nonzero_coef:
                    high_sparse = True
                else:
                    bound = bound * 0.5

        return bound, sparsity, model

    
    def refit(self, x, y):
        if self.task == 'classification':
            model = LogisticRegression(
                        penalty='l2',
                        solver='newton-cg',
                        max_iter=self.max_iter,
                        tol=self.tol,
                        fit_intercept=self.fit_intercept,
                        C=1/self.l2_reg
                    ).fit(x, y)
        else:
            model = Ridge(alpha=self.l2_reg, fit_intercept=self.fit_intercept).fit(x,y)
        return model
    
    def fit(self, x, y):
        low = self.C_range[0]
        self.coef_ = np.zeros((x.shape[1]))
        coef_backup = np.zeros((x.shape[1]))
        high, sparsity, model = self.upper_bound_binary_search(x, y)
        if sparsity == self.max_num_nonzero_coef:
            nz_coef = np.nonzero(model.coef_.reshape(1,-1)[0])[0]
            x_sparse = x[:,nz_coef]
            self.refit_model = self.refit(x_sparse, y)
            self.coef_[nz_coef] = self.refit_model.coef_.reshape(1,-1)[0]
            self.intercept_ = np.array(self.refit_model.intercept_).reshape(1,-1)[0]
            self.C = high
        else:
            for iter in range(self.max_binary_search_iter):
                mid = (low + high) / 2
                model = self.base_func(C=mid).fit(x, y)
                nonzero_count = np.count_nonzero(model.coef_)

                if nonzero_count == self.max_num_nonzero_coef:
                    nz_coef = np.nonzero(model.coef_.reshape(1,-1)[0])[0]
                    x_sparse = x[:,nz_coef]
                    self.refit_model = self.refit(x_sparse, y)
                    self.coef_[nz_coef] = self.refit_model.coef_.reshape(1,-1)[0]
                    self.intercept_ = np.array(self.refit_model.intercept_).reshape(1,-1)[0]
                    self.C = mid
                    break
                elif nonzero_count > self.max_num_nonzero_coef:
                    if self.task == 'classification':
                        high = mid
                    else:
                        low = mid
                else:
                    if self.task == 'classification':
                        low = mid
                    else:
                        high = mid

                if nonzero_count != 0:
                    nz_coef_backup = np.nonzero(model.coef_.reshape(1,-1)[0])[0]
                    x_sparse_backup = x[:,nz_coef_backup]
                    refit_model_backup = self.refit(x_sparse_backup, y)
                    coef_backup[nz_coef_backup] = refit_model_backup.coef_.reshape(1,-1)[0]
                    intercept_backup = np.array(refit_model_backup.intercept_).reshape(1,-1)[0]
                    C_backup = mid

            if (self.refit_model is None) and (iter == self.max_binary_search_iter-1):
                self.refit_model = refit_model_backup
                self.coef_ = coef_backup
                self.intercept_ = intercept_backup
                self.C = C_backup
                
                # raise ValueError("Binary search did not find a valid model with non-zero coefficients.")

        return self
    
    def predict_proba(self, x):
        return expit(x.dot(self.coef_.reshape(-1,1)) + self.intercept_)
    
    def predict(self, x):
        return np.where(self.predict_proba(x)>=0.5, 1, 0) if self.task == 'classification' else x.dot(self.coef_.reshape(-1,1)) + self.intercept_


# ----- rule/term structures -----
@dataclass(frozen=True)
class Term:
    feats: np.ndarray
    weights: np.ndarray
    thr: float
    op: str  # "<=" or ">"

@dataclass
class Rule:
    terms: List[Term]


# ===== 1) train oblique trees (forest) + extract rules =====
def _fit_one_oblique_tree(Xz, y, task, max_depth, n_pair, random_state):
    if task == "classification":
        t = ObliqueClassifier(use_oblique=True, max_depth=max_depth, n_pair=n_pair, random_state=random_state)
    else:
        t = ObliqueRegressor(use_oblique=True, max_depth=max_depth, n_pair=n_pair, random_state=random_state)
    t.fit(Xz, y)
    return t

def extract_rules(tree_dict: Dict[str, Any], feat_map: Optional[np.ndarray], include_internal_nodes: bool) -> List[Rule]:
    rules: List[Rule] = []

    def add_rule(path):
        if path: rules.append(Rule(list(path)))

    def walk(node, path):
        if include_internal_nodes and path:
            add_rule(path)
        if node.get("is_leaf", False):
            if not include_internal_nodes: add_rule(path)
            return

        if node.get("is_oblique", False):
            feats = np.array(node["features"], int)
            if feat_map is not None: feats = feat_map[feats]
            w = np.array(node["weights"], float)
            t = float(node["threshold"])
            path.append(Term(feats, w, t, "<=")); walk(node["left"], path); path.pop()
            path.append(Term(feats, w, t,  ">")); walk(node["right"], path); path.pop()
        else:
            fi = int(node["feature_idx"]); 
            if feat_map is not None: fi = int(feat_map[fi])
            t = float(node["threshold"])
            path.append(Term(np.array([fi]), np.array([1.0]), t, "<=")); walk(node["left"], path); path.pop()
            path.append(Term(np.array([fi]), np.array([1.0]), t,  ">")); walk(node["right"], path); path.pop()

    walk(tree_dict["tree"], [])
    return rules

def fit_oblique_forest_rules(
    X: np.ndarray, y: np.ndarray, task: Literal["classification","regression"],
    n_trees: int = 25, max_depth: int = 3, n_pair: int = 2,
    sample_frac: float = 0.7, feature_frac: float = 1.0,
    random_state: int = 0, include_internal_nodes: bool = True
) -> Tuple[StandardScaler, List[Rule]]:
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    all_rules: List[Rule] = []
    seen = set()

    def rkey(r: Rule, dec=6):
        def tkey(t: Term):
            return (tuple(t.feats.tolist()), tuple(np.round(t.weights, dec)), round(float(t.thr), dec), t.op)
        return tuple(sorted(tkey(t) for t in r.terms))
    
    max_s, d_frac = scale_combinations(d, n_pair, threshold=1000, shrink=0.8)

    for k in range(n_trees):
        for s_ind, s in enumerate(max_s):
            feature_frac = d_frac[s_ind]
            for m_d in range(2, max_depth + 1):
            
                m = max(2, int(sample_frac * n))
                idx = rng.integers(0, n, size=m)
                Xb, yb = Xz[idx], y[idx]

                if feature_frac < 1.0:
                    p = max(1, int(feature_frac * d))
                    feat_map = np.sort(rng.choice(d, size=p, replace=False))
                    Xt = Xb[:, feat_map]
                    tree = _fit_one_oblique_tree(Xt, yb, task, m_d, s, random_state + k)
                    rules = extract_rules(export_tree(tree), feat_map, include_internal_nodes)
                else:
                    tree = _fit_one_oblique_tree(Xb, yb, task, m_d, s, random_state + k)
                    rules = extract_rules(export_tree(tree), None, include_internal_nodes)

                for r in rules:
                    k_ = rkey(r)
                    if k_ not in seen:
                        seen.add(k_)
                        all_rules.append(r)

    return scaler, all_rules


# ===== 2) evaluate rules and scale per RuleFit (center + std ≈ 0.4) =====
def eval_rules_matrix(Xz: np.ndarray, rules: List[Rule]) -> np.ndarray:
    n, K = Xz.shape[0], len(rules)
    R = np.zeros((n, K), dtype=np.uint8)
    for j, rule in enumerate(rules):
        sat = np.ones(n, dtype=bool)
        for t in rule.terms:
            s = (Xz[:, t.feats] * t.weights).sum(axis=1)
            sat &= (s <= t.thr) if t.op == "<=" else (s > t.thr)
            if not sat.any(): break
        R[:, j] = sat
    return R

def scale_rules_rulefit(R: np.ndarray, target_std: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = R.mean(axis=0)
    Rc = R - p  # center
    s = np.sqrt(np.clip(p * (1.0 - p), 1e-6, None))  # std of centered Bernoulli
    R_scaled = Rc / s * target_std
    return R_scaled, p, s


# ===== 3) L1 fit (rules only) =====
@dataclass
class RuleEnsembleModel:
    task: Literal["classification","regression"]
    model: any
    rules: List[Rule]
    scaler: StandardScaler
    p_support: np.ndarray
    s_std: np.ndarray


class ObliqueTreeEnsembles(BaseEstimator):
    def __init__(self, task: Literal["classification","regression"], n_trees: int = 25, max_depth: int = 3, n_pair: int = 2, 
                 sample_frac: float = 0.7, feature_frac: float = 1.0, target_rules: Optional[int] = 30, random_state: int = 0, l2_reg: float = 1e-10):
        self.task = task
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_pair = n_pair
        self.sample_frac = sample_frac
        self.feature_frac = feature_frac
        self.target_rules = target_rules
        self.random_state = random_state
        self.l2_reg = l2_reg
        

    def fit(self, X, y):
        self.feature_name = X.columns
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        self.scaler, self.all_rules = fit_oblique_forest_rules(
            X, y, task=self.task, n_trees=self.n_trees, max_depth=self.max_depth, n_pair=min(self.n_pair, X.shape[1]),
            sample_frac=self.sample_frac, feature_frac=self.feature_frac,
            random_state=self.random_state, include_internal_nodes=True)
        Xz = self.scaler.transform(X)
        q = eval_rules_matrix(Xz, self.all_rules).astype(float)
        # R_scaled, p_support, s_std = scale_rules_rulefit(R, target_std=0.4)

        self.lr = L1Optimization(task=self.task , max_num_nonzero_coef=min(self.target_rules,len(self.all_rules)) , fit_intercept=True, l2_reg=self.l2_reg).fit(q, y)
        self.selected_rules_ind = np.nonzero(self.lr.coef_)[0]
        self.selected_rules = [self.all_rules[i] for i in self.selected_rules_ind]
        self.weights_ = self.lr.coef_[self.selected_rules_ind]
        self.l2_reg_ = self.lr.best_reg_ if hasattr(self.lr, 'best_reg_') else self.l2_reg
        self.intercept_ = self.lr.intercept_
        self.rules = self._rules()

        return self
    
    def predict_proba(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        Xz = self.scaler.transform(X)
        q = eval_rules_matrix(Xz, self.all_rules).astype(float)
        return self.lr.predict_proba(q)

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        Xz = self.scaler.transform(X)
        q = eval_rules_matrix(Xz, self.all_rules).astype(float)
        if self.task == "classification":
            return self.lr.predict(q)
        else:
            return self.lr.predict(q)

    @staticmethod
    def format_rule(coef, rule: Rule, feature_names: Optional[List[str]] = None, decimals: int = 3, x_mu=None, x_sigma=None, y_mu=None, y_sigma=None):
        def term_text(t: Term):
            parts = []
            added_to_thr = 0.0
            thr = t.thr
            for f, w in zip(t.feats, t.weights):
                if x_mu is not None and x_sigma is not None:
                    w = w / x_sigma[f]
                    added_to_thr = added_to_thr + w * x_mu[f]
                name = feature_names[f] if feature_names is not None else f"x{f}"
                parts.append(f"{w:+.{decimals}f}*{name}")
            thr = thr + added_to_thr
            return f"({' '.join(parts).lstrip('+')}) {t.op} {thr:.{decimals}f}"
        conditions = " AND ".join(term_text(t) for t in rule.terms)
        if y_mu is not None and y_sigma is not None and coef != 0:
            coef = coef * y_sigma
        return f"{coef:+8.2f} if {conditions}"
    
    def _rules(self,x_mu=None, x_sigma=None, y_mu=None, y_sigma=None):
        rules = [f"{self.intercept_[0]:+8.2f} if True"]
        if y_mu is not None and y_sigma is not None and self.task == 'regression':
            rules = [f"{self.intercept_[0]*y_sigma+y_mu:+8.2f} if True"]
        for w, r in zip(self.weights_, self.selected_rules):
            rules.append(self.format_rule(coef=w, rule=r, feature_names=self.feature_name, x_mu=x_mu, x_sigma=x_sigma, y_mu=y_mu, y_sigma=y_sigma))
        return rules

    def num_of_rules(self):
        return np.count_nonzero(self.lr.coef_)
    

    def rules_complexity(self):
        c = 0
        for r in self.selected_rules:
            for t in r.terms:
                c += np.count_nonzero(t.weights)
        return c
    
    def num_propositions(self):
        c = 0
        for r in self.selected_rules:
            c += len(r.terms)
        return c


    def model_complexity(self):
        return self.num_of_rules() + self.rules_complexity() + self.num_propositions()
