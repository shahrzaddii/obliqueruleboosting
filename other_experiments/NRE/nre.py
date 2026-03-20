import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier, _tree, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class NeuralRule(nn.Module):
    def __init__(self, weights, biases, c, input_indices):
        super(NeuralRule, self).__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.biases = nn.Parameter(torch.tensor(biases, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32), requires_grad=True)
        self.input_indices = input_indices

    def forward(self, x):
        x_subset = x[:, self.input_indices]  # Only the features this rule uses
        out = torch.relu(torch.matmul(x_subset, self.weights.T) + self.biases)
        min_out, _ = torch.min(out, dim=1, keepdim=True)
        return self.c * min_out

class NeuralRuleEnsemble(nn.Module):
    def __init__(self, rules, task='logreg'):
        super(NeuralRuleEnsemble, self).__init__()
        self.task = task
        self.rules = nn.ModuleList([NeuralRule(w, b, c, indices) for w, b, c, indices in rules])

    def forward(self, x):
        rule_outputs = [rule(x) for rule in self.rules]
        return torch.sum(torch.cat(rule_outputs, dim=1), dim=1, keepdim=True)
    
    # def parameters(self):
    #     return super().parameters()  # This line ensures all submodules' parameters are registered

    def predict_prob(self, X):
        if self.task != 'logreg':
            raise ValueError("predict_prob is only valid for classification.")
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            return torch.sigmoid(logits).numpy()

    def predict(self, X, threshold=0.5):
        if self.task == 'logreg':
            pred = (self.predict_prob(X) >= threshold).astype(int)
        else:
            pred = self.predict_regression(X)
        return pred

    def predict_regression(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return self.forward(X_tensor).numpy()

    def describe_rules(self, feature_names, threshold=1e-6, x_mu=None, x_sigma=None, y_mu=None, y_sigma=None):
        # print("\nFinal Neural Rules (with non-zero output weights):")
        rule_count = 0
        rules_des = []
        if y_mu is not None and y_sigma is not None and self.task == 'linreg':
            rules_des = [f'{y_mu:.3f} if True']
        for idx, rule in enumerate(self.rules):
            props = []
            c_val = rule.c.item()
            if abs(c_val) < threshold:
                continue  # Skip rules with near-zero output weights
            
            if y_mu is not None and y_sigma is not None and self.task == 'linreg':
                c_val = c_val*y_sigma
                
            rule_count += 1

            weights = rule.weights.detach().numpy()  # shape: (num_propositions, T)
            biases = rule.biases.detach().numpy()    # shape: (num_propositions,)
            input_indices = rule.input_indices       # features used in this rule

            for j, (w, b) in enumerate(zip(weights, biases)):
                if x_mu is not None and x_sigma is not None:
                    w = w / x_sigma[input_indices]
                    b = b - np.dot(w, x_mu[input_indices])

                terms = []
                for k, coef in enumerate(w):
                    if abs(coef) > threshold:
                        feat_name = feature_names[input_indices[k]].replace(" ", "_")
                        terms.append(f"{coef:+.3f} {feat_name}")
                if terms:
                    expr = " ".join(terms)
                    props.append(f"  ReLU({expr} {b:+.3f})")
                else:
                    props.append(f"  Skipped (no active features)")
            rule_body = " & ".join(props)
            rules_des.append(f"{c_val:+.3f} If: " + rule_body)
            # print(f"  Then Rule Weight c = {c_val:+.3f}\n")

        if rule_count == 0:
            print("  (No rules with non-zero output weights.)")
        return rules_des


    def num_of_rules(self):
        return len(self.rules)
    
    def rules_complexity(self):
        rules_complexity = 0
        for rule in self.rules:
            rule_weights = rule.weights.detach().numpy()  
            # num_propositions = rule.weights.shape[0]
            num_nonzero_weights = np.count_nonzero(rule_weights)
            rules_complexity = rules_complexity + num_nonzero_weights

        return rules_complexity
            

    def model_complexity(self):
        num_rules = len(self.rules)
        num_propositions = sum(rule.weights.shape[0] for rule in self.rules)
        num_nonzero_weights = sum((rule.weights.detach().abs() > 1e-6).sum().item() for rule in self.rules)
        complexity = num_rules + num_propositions + num_nonzero_weights
        return complexity


def train_nre(X, y, rules, task='logreg', epochs=200, lr=0.01):
    model = NeuralRuleEnsemble(rules, task)
    cs = [rule.c.item() for rule in model.rules]
    ws = [rule.weights.data for rule in model.rules]
    # print(f"At initialisation, c values: {cs}")
    # print(f"At initialisation, weight values: {ws}")
    if task == 'logreg':
        criterion = nn.BCEWithLogitsLoss()
    elif task == 'linreg':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Unknown task type: should be 'classification' or 'regression'")
    
    params = list()
    for r in model.rules:
        params = params + list(r.parameters())

    optimizer = optim.Adam(params, lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            # cs = [rule.c.item() for rule in model.rules]
            # print(f"Epoch {epoch+1}, c values: {cs}")
            # ws = [rule.weights.data for rule in model.rules]
            # print(f"weight values: {ws}")

    kept_rules = []
    for rule in model.rules:
        if abs(rule.c.item()) > 1e-6:
            kept_rules.append(rule)

    print(f"Retaining {len(kept_rules)} / {len(model.rules)} rules with non-zero output weight.")

    # Rebuild the model with only the retained rules
    pruned_rules = []
    for rule in kept_rules:
        w = rule.weights.detach().numpy()
        b = rule.biases.detach().numpy()
        c = rule.c.item()
        indices = rule.input_indices
        pruned_rules.append((w, b, c, indices))

    final_model = NeuralRuleEnsemble(pruned_rules, task)
    final_model.eval()
    return final_model


def best_split_margin_gain(X, y, feature_indices=None):
    """
    Implements Eq. (15) split for binary y in {+1, -1}.
    Returns (best_feat, best_thresh, best_I); best_feat=None if no valid split.
    """
    n = y.shape[0]
    if n <= 1:
        return None, None, -np.inf

    # parent term
    n_pos = np.sum(y == 1)
    n_neg = n - n_pos
    parent_term = ((n_pos - n_neg) ** 2) / n

    if feature_indices is None:
        feature_indices = range(X.shape[1])

    best_I = -np.inf
    best_feat = None
    best_thresh = None

    for j in feature_indices:
        xj = X[:, j]
        order = np.argsort(xj, kind="mergesort")  # stable
        xj_sorted = xj[order]
        y_sorted = y[order]

        # candidates only between distinct values
        diffs = np.diff(xj_sorted)
        split_positions = np.where(diffs != 0)[0]
        if split_positions.size == 0:
            continue

        # cumulative counts for left side
        y_pos = (y_sorted == 1).astype(np.int32)
        cum_pos = np.cumsum(y_pos)                         # left positives up to index i
        cum_total = np.arange(1, n + 1, dtype=np.int32)    # left total up to index i
        cum_neg = cum_total - cum_pos

        total_pos = cum_pos[-1]
        total_neg = n - total_pos

        # stats at candidate splits (left = first k+1 points)
        k = split_positions
        nl = cum_total[k]
        nr = n - nl

        nl_pos = cum_pos[k]
        nl_neg = nl - nl_pos

        nr_pos = total_pos - nl_pos
        nr_neg = total_neg - nl_neg

        # Eq. (15): left + right - parent
        with np.errstate(divide='ignore', invalid='ignore'):
            left_term  = ((nl_pos - nl_neg) ** 2) / np.maximum(nl, 1)
            right_term = ((nr_pos - nr_neg) ** 2) / np.maximum(nr, 1)
            I_vals = left_term + right_term - parent_term

        i_local = np.argmax(I_vals)
        I_local = I_vals[i_local]
        if I_local > best_I:
            best_I = I_local
            # threshold is midpoint between x[k] and x[k+1]
            t = (xj_sorted[k[i_local]] + xj_sorted[k[i_local] + 1]) / 2.0
            best_feat = j
            best_thresh = t

    return best_feat, best_thresh, best_I


class MarginNode:
    __slots__ = ("feat", "thr", "left", "right", "is_leaf", "leaf_value", "indices")

    def __init__(self, indices):
        self.indices = indices     # data indices at this node
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.leaf_value = None     # e.g., majority sign for classification

class MarginTree:
    def __init__(self, max_depth=5, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        # y should be in {+1, -1}
        self.X = X
        self.y = y
        self.root = self._grow(np.arange(X.shape[0]), depth=0)
        return self

    def _grow(self, idx, depth):
        node = MarginNode(idx)
        X, y = self.X[idx], self.y[idx]
        n = len(idx)

        # stopping conditions
        if depth >= self.max_depth or n <= 2 * self.min_samples_leaf or np.all(y == y[0]):
            node.is_leaf = True
            # leaf value (classification): sign of majority
            node.leaf_value = 1.0 if (np.sum(y == 1) >= np.sum(y == -1)) else -1.0
            return node

        feat, thr, I = best_split_margin_gain(X, y)
        if feat is None or not np.isfinite(I):
            node.is_leaf = True
            node.leaf_value = 1.0 if (np.sum(y == 1) >= np.sum(y == -1)) else -1.0
            return node

        # partition
        left_mask  = X[:, feat] <= thr
        right_mask = ~left_mask
        if (left_mask.sum() < self.min_samples_leaf) or (right_mask.sum() < self.min_samples_leaf):
            node.is_leaf = True
            node.leaf_value = 1.0 if (np.sum(y == 1) >= np.sum(y == -1)) else -1.0
            return node

        node.is_leaf = False
        node.feat = feat
        node.thr = thr
        node.left  = self._grow(idx[left_mask],  depth + 1)
        node.right = self._grow(idx[right_mask], depth + 1)
        return node

def extract_rules_from_margin_tree(tree: MarginTree, n_features: int):
    rules = []

    def dfs(node, conds):
        if node.is_leaf:
            # initialize c=0.0 (classification) and learn it; or use node.leaf_value if you prefer
            c_init = 0.0
            # compact per-rule feature set
            feats = sorted(set(f for f, _, _ in conds))
            idx_map = {f:i for i, f in enumerate(feats)}
            T = len(feats)
            weights, biases = [], []
            for f, sgn, thr in conds:
                w = np.zeros(T, dtype=np.float32)
                w[idx_map[f]] = sgn
                weights.append(w)
                biases.append(thr)
            weights = np.stack(weights) if weights else np.zeros((0, T), dtype=np.float32)
            biases  = np.array(biases, dtype=np.float32)
            rules.append((weights, biases, float(c_init), feats))
            return

        # internal node
        f, t = node.feat, node.thr
        dfs(node.left,  conds + [(f, -1, -t)])  # left: x_f <= t  => w=-1, b=-t
        dfs(node.right, conds + [(f, +1, +t)])  # right: x_f > t  => w=+1, b=+t

    dfs(tree.root, [])
    return rules


def extract_rules_from_tree(tree, task='logreg'):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    leaves = []  # list of (conditions, c_init)

    def recurse(node, conds):
        if feature[node] != _tree.TREE_UNDEFINED:
            feat = feature[node]
            thr = threshold[node]
            recurse(tree_.children_left[node],  conds + [(feat, -1, -thr)])
            recurse(tree_.children_right[node], conds + [(feat, +1, +thr)])
        else:
            # Leaf value: mean(y) for regression; for classification you can keep 0.0
            if task == 'regression':
                c_init = float(tree_.value[node][0][0])  # scalar mean at leaf
            else:
                c_init = 0.0
            leaves.append((conds, c_init))

    recurse(0, [])

    rules = []
    for conds, c_init in leaves:
        feats = sorted(set(f for f, _, _ in conds))          # original column indices used by this rule
        idx_map = {f:i for i, f in enumerate(feats)}
        T = len(feats)

        weights = []
        biases = []
        for f, sgn, thr in conds:
            w = np.zeros(T, dtype=np.float32)
            w[idx_map[f]] = sgn
            weights.append(w)
            biases.append(thr)

        weights = np.stack(weights) if len(weights) else np.zeros((0, T), dtype=np.float32)
        biases  = np.array(biases, dtype=np.float32)
        rules.append((weights, biases, float(c_init), feats))  # include per-rule input feature indices

    return rules

class NREModel:
    def __init__(self, max_tree_depth=None, max_leaf_nodes=None, min_samples_split = 2, task='logreg'):
       self.max_tree_depth = max_tree_depth
       self.max_leaf_nodes = max_leaf_nodes
       self.task = task
       self.num_rules = None
       self.min_samples_split = min_samples_split 

    def fit(self, X, y):
        self.feature_names = X.columns 
        X = X.to_numpy()
        y = y.to_numpy()
        if self.task == 'logreg':
            y_pm = np.where(y.ravel()>0, 1, -1)
            tree = MarginTree(max_depth=5, min_samples_leaf=5).fit(X, y_pm)
            self.tree_rules = extract_rules_from_margin_tree(tree, n_features=X.shape[1])
            # tree = DecisionTreeClassifier(max_depth=self.max_tree_depth).fit(X, y)
            # self.tree_rules = extract_rules_from_tree(tree, task=self.task)
        else:
            tree = DecisionTreeRegressor(max_depth=self.max_tree_depth).fit(X, y)
            self.tree_rules = extract_rules_from_tree(tree, task=self.task)
            

        self.tree_to_out = tree
        self.num_rules = len(self.tree_rules)
        self.model = train_nre(X, y, self.tree_rules, task=self.task, epochs=100, lr=0.01)
        return self
   
    def predict(self, x):
        if isinstance(x, pd.DataFrame):
                    x = x.to_numpy()
        return self.model.predict(x)
    
    def predict_prob(self, x):
        if isinstance(x, pd.DataFrame):
                    x = x.to_numpy()
        return self.model.predict_prob(x)
    
    def rules(self, x_mu=None, x_sigma=None, y_mu=None, y_sigma=None):
        return self.model.describe_rules(self.feature_names, x_mu=x_mu, x_sigma=x_sigma, y_mu=y_mu, y_sigma=y_sigma)

    def model_complexity(self):
        return self.model.model_complexity()
    
    def rules_complexity(self):
        return self.model.rules_complexity()
    def num_of_rules(self):
        return self.model.num_of_rules()
