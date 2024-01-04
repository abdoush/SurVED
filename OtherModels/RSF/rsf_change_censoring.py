from rsfhelper import rsf_change_size_only, rsf_change_censoring_only, rsf_change_censoring_and_size


n_estimators = 200
max_depth = 5
min_samples_split = 10
min_samples_leaf = 10
max_features = 'log2'

print('change_size_only', [0.60, 0.51, 0.36, 'full'])
cis = rsf_change_size_only(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, final_test=True)
print('change_size_only:', cis)

print('change_censoring_only', [0.20, 0.35, 0.50, 'full'])
cis = rsf_change_censoring_only(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, final_test=True)
print('change_censoring_only:', cis)

print('change_censoring_and_size', [0.20, 0.35, 0.50, 'full'])
cis = rsf_change_censoring_and_size(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, final_test=True)
print('change_censoring_and_size:', cis)
