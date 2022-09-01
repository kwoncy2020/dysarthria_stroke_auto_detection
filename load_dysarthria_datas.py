import glob, librosa
import numpy as np

BASE_PATH = "C:\kwoncy\\nlp\dysarthria\data"
hospital = "HS"
sentence_type = "SCO"
patient_number = "0001"
# file_type = "wav" ## "wav" | "json" | "txt" | "*"

# iglob_path = f"{BASE_PATH}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.*"

# a = glob.iglob(iglob_path, recursive=True)

# print(list(a))

class InterfaceValueStat:
    def __init__(self,value:'list[np.ndarray]'):
        self.value:'list[np.ndarray]' = value
        self.lengths:'list[int]' = list(map(lambda x: len(x), self.value))
        self.stat:str = self._statistics()
        self.SAMPLE_RATE:int = 16000
        self.WINDOW_SIZE:int = int(self.SAMPLE_RATE * 4)
        self.SLIDE_STRIDES:int = 16000
        self.padded_nparrays:np.ndarray = None

    def _statistics(self):
        if len(self.lengths) == 0:
            return f"len: 0, min: 0, max: 0, average: 0, q1: 0, q3: 0"
        else:
            q1, q3 = np.percentile(self.lengths,[25,75])
            return f"len: {len(self.lengths)}, min: {min(self.lengths)}, max: {max(self.lengths)}, average: {round(sum(self.lengths)/len(self.lengths))}, q1: {q1}, q3: {q3}"

    def get_padded_nparray(self)->np.ndarray:
        stacks = []
        for i in self.value:
            item_length = len(i)
            if item_length <= self.WINDOW_SIZE:
                pad_n = self.WINDOW_SIZE - item_length
                pad = np.zeros((pad_n,), dtype=i.dtype)
                i2 = np.append(i, pad)
                stacks.append(i2)
            else:
                step = (item_length-self.WINDOW_SIZE)//self.SLIDE_STRIDES
                for j in range(0,step+1):
                    start = j*self.SLIDE_STRIDES
                    stacks.append(i[start: start+self.WINDOW_SIZE])
        
        self.padded_nparrays = np.stack(stacks, axis=0)
        return self.padded_nparrays

    def get_padded_shuffled_n_nparray(self,n:int=None) -> np.ndarray:
        if self.padded_nparrays == None:
            self.get_padded_nparray()
        if n == None:
            n = self.padded_nparrays.shape[0]
        # indices = list(self.padded_nparrays.shape[0])
        # np.random.shuffle(indices)
        
        indices = np.random.choice(self.padded_nparrays.shape[0],size=n, replace=False)

        return self.padded_nparrays[indices]


class Patients:
    def __init__(self, base_path:str, hospital:str, sentence_type:str, patient_number:str) -> None:
        iglob_wav_path:str = f"{base_path}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.wav"
        iglob_txt_path:str = f"{base_path}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.txt"
        iglob_json_path:str = f"{base_path}\{hospital}\{sentence_type}\**\{hospital}{patient_number}*.json"

        self.patient_info = f"{base_path}\{hospital}\{sentence_type}\{hospital}{patient_number}"
        self.wav_files:'list[str]' = list(glob.iglob(iglob_wav_path))
        self.txt_files:'list[str]' = list(glob.iglob(iglob_txt_path))
        self.json_files:'list[str]' = list(glob.iglob(iglob_json_path))
        self.loaded_wavs:InterfaceValueStat = InterfaceValueStat([])
        self.is_wavs_loaded:bool = False

        self.load_wavs()

    def load_wav(self, wav_file:str) -> np.ndarray:
        arr, sr =  librosa.load(wav_file, sr = 16000)
        return arr


    def load_wavs(self, wav_files:'list[str]' = None) -> 'dict':
        if wav_files == None:
            wav_files = self.wav_files
        self.is_wavs_loaded = True  
        self.loaded_wavs = InterfaceValueStat(list(map(self.load_wav, wav_files)))
        return self.loaded_wavs

    def save_nparray(self, nparray:np.ndarray, save_name:str=None):
        if save_name == None:
            save_name = self.patient_info + ".npy"

        np.save(save_name, nparray)

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
