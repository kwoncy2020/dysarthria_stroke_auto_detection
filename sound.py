import glob, librosa
import numpy as np
from load_dysarthria_datas import Patients

BASE_PATH = "C:\kwoncy\\nlp\dysarthria\data"
hospital = "HS"
sentence_type = "SCO"
patient_number = "0001"
file_type = "wav" ## "wav" | "json" | "txt" | "*"

# iglob_path = f"{BASE_PATH}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.{file_type}"

# a = glob.iglob(iglob_path, recursive=True)

# print(list(a))

# ci1 = Patients(BASE_PATH,"HS","SCO","0001")
# ci2 = Patients(BASE_PATH,"HS","SCO","0002")

# print(len(a.load_wavs()))

hl98 = Patients(BASE_PATH,"HL","SCO","0098")
print(hl98.loaded_wavs.stat)
print(hl98.loaded_wavs.value[0].shape)
print(hl98.loaded_wavs.value[0].dtype)

hl98.loaded_wavs.get_padded_nparray()
print(hl98.loaded_wavs.padded_nparrays.shape)  ## 43



ci3 = Patients(BASE_PATH,"HS","SCO","0003")
npa = ci3.loaded_wavs.get_padded_shuffled_n_nparray(43)
ci3.save_nparray(npa,"ci3_loaded_wavs_43.npy")

ci4 = Patients(BASE_PATH,"HS","SCO","0004")
ci4.save_nparray(ci4.loaded_wavs.get_padded_shuffled_n_nparray(43),"ci4_loaded_wavs_43")
ci5 = Patients(BASE_PATH,"HS","SCO","0005")
ci5.save_nparray(ci5.loaded_wavs.get_padded_shuffled_n_nparray(43),"ci5_loaded_wavs_43")
