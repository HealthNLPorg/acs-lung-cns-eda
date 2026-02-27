# acs-lung-cns-eda
Example command
```
uv run main.py \
--casenum_ade_date_table ~/Documents/cns_eda_workspace/tables/tox_bitterman_8dec25.xlsx \
--casenum_dfci_mrn_table ~/Documents/cns_eda_workspace/tables/pt_name_dfci_mrn_case_number.csv \
--inpatient_json_path ~/Documents/cns_eda_workspace/Inpatient\ Progress.json \
--outpatient_json_path ~/Documents/cns_eda_workspace/Outpatient\ Progress.json \
--output_dir .
 ```
 Updated
```
 uv run main.py \
--casenum_ade_date_table ~/Documents/cns_eda_workspace/tables/tox_bitterman_8dec25.xlsx \
--casenum_dfci_mrn_table ~/Documents/cns_eda_workspace/tables/pt_name_dfci_mrn_case_number.csv \
--inpatient_json_path ~/Documents/cns_eda_workspace/Inpatient\ Progress.json \
--outpatient_json_path ~/Documents/cns_eda_workspace/Outpatient\ Progress.json \
--filter_by_word_count --output_dir .
 ```

 Even more updated
```
 uv run main.py \
--casenum_ade_date_table ~/Documents/cns_eda_workspace/tables/tox_bitterman_8dec25.xlsx \
--casenum_dfci_mrn_table ~/Documents/cns_eda_workspace/tables/pt_name_dfci_mrn_case_number.csv \
--inpatient_json_path ~/Documents/cns_eda_workspace/Inpatient\ Progress.json \
--outpatient_json_path ~/Documents/cns_eda_workspace/Outpatient\ Progress.json \
--inpatient_provider_departments_path  ~/Partners\ HealthCare\ Dropbox/Eli\ Goldner/PHI-ASTRO-ACS/cns/provider_type_filtered_outpatient.csv \
--outpatient_provider_departments_path  ~/Partners\ HealthCare\ Dropbox/Eli\ Goldner/PHI-ASTRO-ACS/cns/provider_type_filtered_outpatient.csv \
--filter_by_word_count --output_dir .
 ```

Even even more updated
```
uv run main.py \
--casenum_ade_date_table ~/Documents/cns_eda_workspace/tables/tox_bitterman_8dec25.xlsx \
--casenum_dfci_mrn_table ~/Documents/cns_eda_workspace/tables/pt_name_dfci_mrn_case_number.csv \
--inpatient_json_path ~/Documents/cns_eda_workspace/Inpatient\ Progress.json \
--outpatient_json_path ~/Documents/cns_eda_workspace/Outpatient\ Progress.json \
--inpatient_provider_departments_path  ~/Partners\ HealthCare\ Dropbox/Eli\ Goldner/PHI-ASTRO-ACS/cns/provider_type_filtered_outpatient.csv \
--outpatient_provider_departments_path  ~/Partners\ HealthCare\ Dropbox/Eli\ Goldner/PHI-ASTRO-ACS/cns/provider_type_filtered_outpatient.csv \
--filter_by_word_count --output_dir . \
--stratify_beginning
```
