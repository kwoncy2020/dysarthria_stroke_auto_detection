import glob, librosa, json, pickle
import numpy as np
import pandas as pd

BASE_PATH = "C:\kwoncy\\nlp\dysarthria2\data"

# hospital = "HS"
# sentence_type = "SCO"
# patient_number = "0001"
file_type = "wav" ## "wav" | "json" | "txt" | "*"
## C:\kwoncy\nlp\dysarthria2\data\0919\HL\SCO\A



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

def save_statistics_from_jsons(folder_path:str)->None:
    jsons = glob.glob(f'{folder_path}\**\*.json', recursive=True)

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


def save_total_statistics(base_path:'str', folder_names:'list[str]')->None:
    dataframes = []

    for name in folder_names:
        with open(f'{base_path}/{name}/patients_info_{name}.pkl', 'rb') as f:
            # a = pickle.load(f, encoding='utf-8')
            dataframe = pickle.load(f)
        dataframes.append(dataframe)
    
    df = pd.concat(dataframes, axis=0)
    
    count_sum = df.groupby('speakerID')['count'].sum()
    count_sum_frame = count_sum.to_frame(name='count')
    df.drop(columns=['count'], inplace=True)
    df = df.drop_duplicates()
    df = df.sort_values('speakerID')
    count_sum_frame = count_sum_frame.sort_values('speakerID')

    ## even if call drop_duplicates(), there's same speakerID because of miss data.
    ## so concat cause error. do another way to add count column
    # df2 = pd.concat([df,count_sum_frame],axis=1)
    df['count'] = count_sum_frame.loc[df.index,'count']

    df.to_excel("patients_info_total.xlsx")
    df.to_pickle("patients_info_total.pkl")



# json_path = "C:\kwoncy\\nlp\dysarthria2\data\\0930_2"
# save_statistics_from_jsons(json_path)

save_total_statistics('C:\kwoncy\\nlp\dysarthria2\data',['0919','0922','0923','0930_1','0930_2'])