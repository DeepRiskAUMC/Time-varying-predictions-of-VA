#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import biosppy.signals.ecg
import neurokit2 as nk
log = pd.read_csv(path_log)
ecg = np.load(path_np)

df = ecg
heartbeats_all = np.empty((0, 251))
for i in range(df.shape[0]):
    ecg_pat = df[i, :, :]
    try:
        ecg_data_id = ecg_pat[:, 3]
        signals, info = nk.ecg_process(ecg_data_id, sampling_rate=250)
        rpeaks = info["ECG_R_Peaks"]
        cleaned_ecg = signals["ECG_Clean"]
        heartbeats = biosppy.signals.ecg.extract_heartbeats(cleaned_ecg, rpeaks=rpeaks, sampling_rate=250, before=0.4, after=0.6)
        heartbeats_temp = heartbeats['templates']
        heartbeats_array = np.concatenate((heartbeats_temp, np.full((heartbeats_temp.shape[0], 1), i)), axis=1)
        heartbeats_all = np.concatenate((heartbeats_all, heartbeats_array))

    except:
        # Add a row of NaN to the array for the current ID
        heartbeats_array = np.full((1, 251), np.nan)
        heartbeats_array[0, -1] = i
        heartbeats_all = np.concatenate((heartbeats_all, heartbeats_array))

heartbeats_all = pd.DataFrame(heartbeats_all)

