# Dataset Placement and Structure (local-only)

The dataset is not stored in Git. Place files under `data/` locally as shown below.

Recommended layout (per the IEEE DataPort mmWave MIMO dataset):

```text
mmwave_MIMO_project/
└─ data/
   ├─ day0/
   │  ├─ train_data/
   │  │  ├─ mmwave_data_train_1.mat
   │  │  ├─ mmwave_data_train_2.mat
   │  │  └─ ... (up to _9.mat)
   │  └─ test_data/
   │     └─ mmwave_data_test.mat
   ├─ day1/
   │  ├─ train_data/
   │  │  ├─ mmwave_data_train_1.mat
   │  │  └─ ... (up to _9.mat)
   │  └─ test_data/
   │     └─ mmwave_data_test.mat
   └─ day2/
      ├─ train_data/
      │  ├─ mmwave_data_train_1.mat
      │  └─ ... (up to _9.mat)
      └─ test_data/
         └─ mmwave_data_test.mat
```

Inside each training file (per device `k`):

- `mmwave_data_train_k`: approx. shape `N_k x 256 x 63` (range–azimuth maps)
- `label_train_k`: integer labels in `{0..9}`

Test file fields:

- `mmwave_data_test`: approx. `N_test x 256 x 63`
- `label_test`: integer labels in `{0..9}`

Notes:

- `.gitignore` excludes `data/` and `*.mat` to keep large files out of Git.
- If you use different folder names, update your configs/CLI accordingly when running loaders.
- Source dataset: IEEE DataPort — "Federated Learning: mmWave MIMO Radar Dataset for Testing" (Savazzi et al.). Official FL reference code: <https://github.com/labRadioVision/federated>

Quick check (PowerShell) to verify counts:

```powershell
Get-ChildItem data -Recurse -Filter *.mat | Group-Object Directory | Select-Object Count, Name | Sort-Object Name
```
