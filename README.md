# The Concordance Index Decomposition
C-index (CI) is a weighted harmonic average of the C-indices defined for the subsets ee (events vs. events) and ec (events vs. censored cases).


## $\frac{1}{CI} = \alpha \frac{1}{CI_{ee}} + (1 - \alpha) \frac{1}{CI_{ec}}$

To use the C-index decompostion, download the file Utils/metrics.py and use the function c_index_decomposition. The function will return the following terms:
 * <b>Cee</b>: The C-index of the ee pairs.
 * <b>Cec</b>: The C-index of the ec pairs.
 * <b>alpha</b>: The weight alpha.
 * <b>alpha_deviation</b>: The deviation from the optimal alpha.
 * <b>C</b>: The total C-index.

For more details, see the full paper [The Concordance Index decomposition: A measure for a deeper understanding of survival prediction models](https://doi.org/10.1016/j.artmed.2024.102781)

## BibTeX Citation
```
@article{ALABDALLAH2024102781,
   title = {The Concordance Index decomposition: A measure for a deeper understanding of survival prediction models},
   journal = {Artificial Intelligence in Medicine},
   volume = {148},
   pages = {102781},
   year = {2024},
   issn = {0933-3657},
   doi = {https://doi.org/10.1016/j.artmed.2024.102781},
   url = {https://www.sciencedirect.com/science/article/pii/S093336572400023X},
   author = {Abdallah Alabdallah and Mattias Ohlsson and Sepideh Pashami and Thorsteinn Rögnvaldsson},
   keywords = {Survival analysis, Evaluation metric, Concordance Index, Variational encoder–decoder}
}
```

## SurVED
Survival Analysis with Variational Encoder Decoder.

## Reproducing the results
SurVED 
* Run the file *surved_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *surved_change_censoring.py* to reproduce the SurVED model results on the SUPPORT dataset with changing censoring levels.

DeepHit:
* Run the file *OtherModels\DeepHit\deephit_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\DeepHit\deephit_change_censoring.py* to reproduce the DeepHit model results on the SUPPORT dataset with changing censoring levels.

DeepSurv:
* Run the file *OtherModels\DeepSurv\deepsurv_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\DeepSurv\deepsurv_change_censoring.py* to reproduce the DeepSurv model results on the SUPPORT dataset with changing censoring levels.

RSF (Random Survival Forest)
* Run the file *OtherModels\RSF\rsf_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\RSF\rsf_change_censoring.py* to reproduce the RSF model results on the SUPPORT dataset with changing censoring levels.

CPH (Cox Proportional Hazard)
* Run the file *OtherModels\CPH\cph_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\CPH\cph_change_censoring.py* to reproduce the CPH model results on the SUPPORT dataset with changing censoring levels.

DATE:
* Run the file *OtherModels\DATE\date_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\DATE\date_change_censoring.py* to reproduce the DATE model results on the SUPPORT dataset with changing censoring levels.

   _Note: DATE repository should be downloaded from [DATE_Repo](https://github.com/paidamoyo/adversarial_time_to_event) and placed in the same folder_

VSI:
* Run the file *OtherModels\VSI\vsi_final_test.py* to reproduce the 100-fold test results of the four datasets.
* Run the file *OtherModels\VSI\vsi_change_censoring.py* to reproduce the VSI model results on the SUPPORT dataset with changing censoring levels.

   _Note: VSI repository should be downloaded from [VSI_Repo](https://github.com/ZidiXiu/VSI) and placed in the same folder_


_Copies of the four datasets are provided in the Data folder for convenience._
