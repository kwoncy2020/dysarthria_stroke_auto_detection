import glob, librosa, pickle, random
import numpy as np
import pandas as pd

BASE_PATH = "C:\kwoncy\\nlp\dysarthria\data"
hospital = "HS"
sentence_type = "SCO"
patient_number = "0001"
# file_type = "wav" ## "wav" | "json" | "txt" | "*"

# iglob_path = f"{BASE_PATH}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.*"

# a = glob.iglob(iglob_path, recursive=True)

# print(list(a))

class InterfaceWavsStat:
    def __init__(self,wavs:'list[np.ndarray]'):
        self.wavs:'list[np.ndarray]' = wavs
        self.lengths:'list[int]' = list(map(lambda x: len(x), self.wavs))
        self.stat:str = self._statistics()
        self.SAMPLE_RATE:int = 16000
        self.WINDOW_SIZE:int = int(self.SAMPLE_RATE * 4)
        self.SLIDE_STRIDES:int = 16000
        self.padded_nparray:np.ndarray = None

    def _statistics(self):
        if len(self.lengths) == 0:
            return f"len: 0, min: 0, max: 0, average: 0, q1: 0, q3: 0"
        else:
            q1, q3 = np.percentile(self.lengths,[25,75])
            return f"len: {len(self.lengths)}, min: {min(self.lengths)}, max: {max(self.lengths)}, average: {round(sum(self.lengths)/len(self.lengths))}, q1: {q1}, q3: {q3}"

    def get_padded_nparray(self)->np.ndarray:
        stacks = []
        for wav in self.wavs:
            wav_length = len(wav)
            if wav_length <= self.WINDOW_SIZE:
                pad_n = self.WINDOW_SIZE - wav_length
                pad = np.zeros((pad_n,), dtype=wav.dtype)
                wav_padded = np.append(wav, pad)
                stacks.append(wav_padded)
            else:
                step = (wav_length-self.WINDOW_SIZE)//self.SLIDE_STRIDES
                for j in range(0,step+1):
                    start = j*self.SLIDE_STRIDES
                    stacks.append(wav[start: start+self.WINDOW_SIZE])
        
        self.padded_nparray = np.stack(stacks, axis=0)
        return self.padded_nparray

    def get_padded_shuffled_n_nparray(self,n:int=None) -> np.ndarray:
        if self.padded_nparray == None:
            self.get_padded_nparray()
        if n == None:
            n = self.padded_nparray.shape[0]
        # indices = list(self.padded_nparrays.shape[0])
        # np.random.shuffle(indices)
        
        indices = np.random.choice(self.padded_nparray.shape[0],size=n, replace=False)

        return self.padded_nparray[indices]


class Patient:
    def __init__(self, BASE_PATH:str, patient_id:str, patients_info_path:str) -> None:
        self.patient_id = patient_id
        self.loaded_wavs:InterfaceWavsStat = InterfaceWavsStat([])

        self.wav_paths = f"{BASE_PATH}\{patient_id}*.wav"
        # self.txt_paths = f"{BASE_PATH}\{patient_id}*.txt"
        # self.json_paths = f"{BASE_PATH}\{patient_id}*.json"
        self.wav_files:'list[str]' = glob.glob(self.wav_paths,recursive=True)
        self.wav_files_length = len(self.wav_files)
        print("self.wav_files's length: ", len(self.wav_files))
        # self.txt_files:'list[str]' = glob.glob(self.txt_paths,recursive=True)
        # self.json_files:'list[str]' = glob.glob(self.json_paths)
        self.patient_info_file_path = glob.glob(f"{patients_info_path}\patients_info*.pkl")[0]
        with open(self.patient_info_file_path, 'rb') as f :
            self.patient_info_df = pickle.load(f)
        # print(self.patient_info_df.head(5))
        
        self.patient_info_df = self.patient_info_df.reset_index()
        ## pkl dataframe index is speakerID. index reset require so that it can be using below .loc[ ??]
         
        # print(self.patient_info_df.head(5))
        if self.patient_id not in self.patient_info_df["speakerID"].to_list():
            raise Exception(f"from Patient init: ({self.patient_id}) not in the dataframe from ({self.patient_info_file_path})" )
        self.classification = self.patient_info_df.loc[self.patient_info_df["speakerID"] == self.patient_id,'classification'].values[0]
        self.intelligibility = self.patient_info_df.loc[self.patient_info_df["speakerID"] == self.patient_id,'intelligibility'].values[0]
        self.age = self.patient_info_df.loc[self.patient_info_df["speakerID"] == self.patient_id,'age'].values[0]
        self.gender = self.patient_info_df.loc[self.patient_info_df["speakerID"] == self.patient_id,'gender'].values[0]
        self.is_wavs_loaded:bool = False

        # self.load_wavs()

    def load_wav(self, wav_file:str) -> np.ndarray:
        arr, sr =  librosa.load(wav_file, sr = 16000)
        return arr


    def load_wavs(self, wav_files:'list[str]'=None, n:int=None) -> 'Patient':
        ## load wav files. if input is None, then loads self.wav_files which is indicate the files made at the time this class being instance.
        ## this function will set self.loaded_wavs using input list of wav_files
        if wav_files == None:
            wav_files = self.wav_files
        self.is_wavs_loaded = True  
        
        if n == None :
            n = self.wav_files_length
        
        length_wav_files = len(wav_files)
        select_length = min(n,length_wav_files)
        sampled_wavs=random.sample(wav_files,k=select_length)
        self.loaded_wavs = InterfaceWavsStat(list(map(self.load_wav, sampled_wavs)))
        print("sampled_loaded_wavs: ", len(sampled_wavs))
        
        return self

    def save_nparray(self, nparray:np.ndarray=None, save_name:str=None) -> 'Patient':
        if not isinstance(nparray, np.ndarray):
            nparray = self.loaded_wavs.get_padded_shuffled_n_nparray()
        if save_name == None:
            save_name = f"{self.patient_id}_{self.classification}_{self.intelligibility}_{self.age}_{self.gender}_h{nparray.shape[0]}_w{nparray.shape[1]}.npy"

        np.save(save_name, nparray)
        return self

    # def load_wavs(self, wav_files:'list[str]' = None) -> 'list[np.ndarray]':
    #     if wav_files == None:
    #         wav_files = self.wav_files
    #     self.loaded_wavs = list(map(self.load_wav, wav_files))
    #     return self.loaded_wavs

    # def wavs_statistics(self, loaded_wavs:'list[np.ndarray]'= None ) -> str:
    #     if loaded_wavs == None:
    #         if self.loaded_wavs == None:
    #             return 'no wavs'
    #         else:
    #             loaded_wavs = self.loaded_wavs
        
    #     lengths = list(map(lambda x:len(x),loaded_wavs))
        
    #     return f"len: {len(lengths)}, min: {min(lengths)}, max: {max(lengths)} ,average: {sum(lengths)/len(lengths)}"



if __name__ == "__main__":
    BASE_PATH = "C:\kwoncy\\nlp\dysarthria2\data\**"
    ID = "HL0111"
    PATIENT_INFO_PATH = "C:\kwoncy\\nlp\dysarthria2\data\\0923"

    
    # patient.save_nparray()
    # print(len(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\**\**\**\\",recursive=True)))  ## 3050?
    # print(len(set(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\**\**\**\\",recursive=True)))) ## 101
    # print(len(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\\",recursive=True)))               ## 101
    # print(len(set(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\\",recursive=True))))          ## 101
    # print(len(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\\")))                              ## 3
    # print(len(set(glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\\"))))                         ## 3

    # print(list(map(lambda x: x[-20:],glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\**\**\**\\",recursive=True))))
    # print(set(list(map(lambda x: x[-20:],glob.glob("C:\kwoncy\\nlp\dysarthria2\data\**\**\**\**\\",recursive=True)))))

    patient = Patient(BASE_PATH,ID,PATIENT_INFO_PATH).load_wavs().save_nparray()

    # BASE_PATH2 = "C:\kwoncy\\nlp\dysarthria2\CI"
    # a = np.load(glob.glob(f'{BASE_PATH2}\{ID}*.npy')[0])
    # l_ = []
    # for file_path in glob.glob(f'{BASE_PATH2}\*.npy'):
    #     temp = np.load(file_path)
    #     l_.append(temp)
    #     print(file_path)



    
