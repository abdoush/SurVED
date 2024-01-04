from survedhelper import surved_change_size_only, surved_change_censoring_only, surved_change_censoring_and_size

epochs= 10000000
patience= 1000

events_weight = 0.001
censored_weight = 0.8
c_index_lb_weight = 0.8
kl_loss_weight = 0.0005

print('change_size_only', [0.60, 0.51, 0.36, 'full'])
cis = surved_change_size_only(events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=256)
print('change_size_only:', cis)

print('change_censoring_only', [0.20, 0.35, 0.50, 'full'])
cis = surved_change_censoring_only(events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=256)
print('change_censoring_only:', cis)

print('change_censoring_and_size', [0.20, 0.35, 0.50, 'full'])
cis = surved_change_censoring_and_size(events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=256)
print('change_censoring_and_size:', cis)


