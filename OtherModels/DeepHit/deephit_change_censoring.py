from deephithelper import deephit_change_size_only, deephit_change_censoring_only, deephit_change_censoring_and_size

nc = 400
alpha = 0.05
sigma = 0.8
epochs = 10000000
patience = 1000
batch_size = 256

print('change_size_only', [0.60, 0.51, 0.36, 'full'])
cis = deephit_change_size_only(nc=nc, alpha=alpha, sigma=sigma, epochs=epochs, patience=patience, batch_size=batch_size, final_test=True)
print('change_size_only:', cis)

print('change_censoring_only', [0.20, 0.35, 0.50, 'full'])
cis = deephit_change_censoring_only(nc=nc, alpha=alpha, sigma=sigma, epochs=epochs, patience=patience, batch_size=batch_size, final_test=True)
print('change_censoring_only:', cis)

print('change_censoring_and_size', [0.20, 0.35, 0.50, 'full'])
cis = deephit_change_censoring_and_size(nc=nc, alpha=alpha, sigma=sigma, epochs=epochs, patience=patience, batch_size=batch_size, final_test=True)
print('change_censoring_and_size:', cis)
