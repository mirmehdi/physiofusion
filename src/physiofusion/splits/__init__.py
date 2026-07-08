"""Subject-grouped, leakage-proof data splitting (the P0 keystone)."""
"""The keystone: split windows by SUBJECT so no person is ever in
both the training pile and the test pile.

We don't hand-roll the dealing logic — scikit-learn's GroupKFold already
does 'split into folds, never let a group span two folds.' We just wrap
it in one small, honest function that speaks our 'groups' language.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def subject_grouped_split(groups, n_splits=5, fold=0):
    """Deal window positions into train/test, keeping each subject whole.

    groups   : array of length N, the owner sticker for each window
               (this is the 'groups' array build_dataset gave you)
    n_splits : how many folds to divide the subjects into. 5 folds means
               each test pile holds ~1/5 of the subjects (~3 of your 15).
    fold     : which of the folds to use as the TEST pile (0 to n_splits-1).
               The other folds become the training pile.

    returns  : train_idx, test_idx — arrays of ROW POSITIONS into your
               X / y / groups. Not subject ids — actual row numbers you can
               use like  X[train_idx],  y[test_idx],  etc.
    """
    groups = np.asarray(groups)

    # GroupKFold looks at 'groups' and produces n_splits ways to divide the
    # data such that no group (subject) is ever split across the divide.
    gkf = GroupKFold(n_splits=n_splits)

    # gkf.split(...) yields one (train_positions, test_positions) pair per
    # fold. We only need X's length and the groups, so we pass a dummy X.
    # We walk the folds and stop at the one the caller asked for.
    for i, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
        if i == fold:
            return train_idx, test_idx

    # If we get here, 'fold' was out of range (e.g. asked for fold 9 of 5).
    raise ValueError(f"fold {fold} is out of range for n_splits={n_splits}")