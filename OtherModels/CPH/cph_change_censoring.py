from cphhelper import cph_change_size_only, cph_change_censoring_only, cph_change_censoring_and_size
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

print('cph_change_size_only', [0.60, 0.51, 0.36, 'full'])
cis = cph_change_size_only(reg=0.1, final_test=True)
print('cph_change_size_only:', cis)

print('cph_change_censoring_only', [0.20, 0.35, 0.50, 'full'])
cis = cph_change_censoring_only(reg=0.1, final_test=True)
print('cph_change_censoring_only:', cis)

print('cph_change_censoring_and_size', [0.20, 0.35, 0.50, 'full'])
cis = cph_change_censoring_and_size(reg=0.1, final_test=True)
print('cph_change_censoring_and_size:', cis)
