from vsihelper import vsi_change_size_only, vsi_change_censoring_only, vsi_change_censoring_and_size

nc = 100
epochs=10000000
patience=1000

print('change_size_only', [0.60, 0.51, 0.36, 'full'])
cis = vsi_change_size_only(nc=nc, epochs=epochs, patience=patience, final_test=True)
print('change_size_only:', cis)

print('change_censoring_only', [0.20, 0.35, 0.50, 'full'])
cis = vsi_change_censoring_only(nc=nc, epochs=epochs, patience=patience, final_test=True)
print('change_censoring_only:', cis)

print('change_censoring_and_size', [0.20, 0.35, 0.50, 'full'])
cis = vsi_change_censoring_and_size(nc=nc, epochs=epochs, patience=patience, final_test=True)
print('change_censoring_and_size:', cis)
