import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_genes=50):
        self.target_genes = target_genes
        self.selected_indices_ = None
        
    def fit(self, X, y):
        initial_k = self.target_genes * 6
        
        # 1. ANOVA
        f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, 0)
        top_anova = np.argsort(f_scores)[-initial_k:]
        
        # 2. LASSO
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', random_state=42)
        lasso.fit(X, y)
        lasso_coefs = np.abs(lasso.coef_).max(axis=0)
        top_lasso = np.argsort(lasso_coefs)[-initial_k:]
        
        # 3. RF
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X, y)
        top_rf = np.argsort(rf.feature_importances_)[-initial_k:]
        
        # Consensus
        from collections import Counter
        votes = Counter(list(top_anova) + list(top_lasso) + list(top_rf))
        candidates = [idx for idx, count in votes.items() if count >= 2]
        
        rank_anova = {idx: r for r, idx in enumerate(np.argsort(f_scores))}
        rank_lasso = {idx: r for r, idx in enumerate(np.argsort(lasso_coefs))}
        rank_rf = {idx: r for r, idx in enumerate(np.argsort(rf.feature_importances_))}
        
        def combined_score(idx):
            return rank_anova.get(idx, 0) + rank_lasso.get(idx, 0) + rank_rf.get(idx, 0)
            
        candidates = sorted(candidates, key=combined_score, reverse=True)
        
        if len(candidates) < self.target_genes:
            needed = self.target_genes - len(candidates)
            extra = [idx for idx in np.argsort(f_scores)[::-1] if idx not in candidates]
            candidates.extend(extra[:needed])
            
        self.selected_indices_ = np.array(candidates[:self.target_genes])
        return self
        
    def transform(self, X):
        return X[:, self.selected_indices_]
