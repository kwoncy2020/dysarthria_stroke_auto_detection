import glob, librosa, json, pickle
import numpy as np
import pandas as pd

BASE_PATH = "C:\kwoncy\\nlp\dysarthria2\data"
BASE_PATH = "C:\kwoncy\\nlp\dysarthria2\data\\0923"
# hospital = "HS"
# sentence_type = "SCO"
# patient_number = "0001"
file_type = "wav" ## "wav" | "json" | "txt" | "*"
## C:\kwoncy\nlp\dysarthria2\data\0919\HL\SCO\A
jsons = glob.glob(f'{BASE_PATH}\**\*.json', recursive=True)
jsons = glob.glob(f'{BASE_PATH}\**\*.json', recursive=True)

# print(jsons)

def process_json(json_file:str):
    with open(json_file,'r',encoding='utf-8') as f:
        print(json_file)
        jsn = json.load(f, strict=False)
    
    dataset_info = jsn["dataset"]
    id = dataset_info["speakerID"]
    speaker_info = jsn["speaker"]
    hospital = speaker_info["hospital"]
    classification = speaker_info["classification"]
    degree = speaker_info["degree"]
    education = speaker_info["education"]
    intelligibility = speaker_info["intelligibility"]
    gender = speaker_info["gender"]
    age = speaker_info["age"]
    new_json_obj = {"speakerID": f"{id}",**speaker_info}

    return new_json_obj

a = pd.DataFrame(list(map(process_json,jsons)))
a = a.sort_values('speakerID')
id_series = a.value_counts('speakerID')
id_frame = id_series.to_frame(name='count')
print(id_frame)
print('frame:', id_frame.index)
print('frame:', id_frame.values)
print(id_frame)
b = a.drop_duplicates(['speakerID'])
b = b.set_index('speakerID')

c = pd.concat([b,id_frame],axis=1)
print(c.iloc[:,5:])

c.to_excel("patients_info.xlsx")
c.to_pickle("patients_info.pkl")


# with open('patients_info_0919.pkl', 'rb') as f:
#     # a = pickle.load(f, encoding='utf-8')
#     a = pickle.load(f)
    
