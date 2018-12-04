import librosa
from IPython.display import HTML, display, Javascript, clear_output
from dtw import dtw
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
import os
import glob
import pickle
from traitlets import traitlets
from ipywidgets import *

class LoadedText(widgets.Text):
    """A button that can holds a value as a attribute."""

    def __init__(self, archive_value=None, location_value=None, voice_value=None, *args, **kwargs):
        super(LoadedText, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(archive_value=traitlets.Any(archive_value))
#        self.add_traits(location_value=traitlets.Any(location_value))
        self.add_traits(test_value=traitlets.Any(voice_value))

class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))
    
def load_archive(ex):
    with open('voiceDataArchive.pickle','rb') as handle:
        voiceDataArchive = pickle.load(handle)
    ex.value = voiceDataArchive
    return ex.value

def archive_deets():
    with open('voiceDataArchive.pickle','rb') as handle:
        voiceDataArchive = pickle.load(handle)
    print('Number of entries in archive is:',len(voiceDataArchive))
    print('')
    print('Names of people in the archive:')
    count = 0
    for key in voiceDataArchive.keys():
        count+=1
        print(str(count)+'.',key)

def audio_preprocessing(audio):
    #create Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(audio)
    #standardize data - remove mean and scale to unit variance
    scaler = StandardScaler()
    scaledAudio = scaler.fit_transform(mfcc)
    return scaledAudio

def find_single_difference(target_audio, input_audio):
    min_distance, cost_matrix, accum_cost_matrix, wrap_path = dtw(target_audio.T,
                                                              input_audio.T, 
                                                              dist=lambda x, y: norm(x-y, ord=1)
                                                             )
    return min_distance, cost_matrix, accum_cost_matrix, wrap_path

def compute_average_distance(target_audio, input_audio):
    '''
    target_audio: audio data to compare the input_audio to
    input_audio: the audio data that is to be matched
    '''
    dist1 = find_single_difference(target_audio[0], input_audio)[0]
    dist2 = find_single_difference(target_audio[1], input_audio)[0]
    dist3 = find_single_difference(target_audio[2], input_audio)[0]
    avg_dist = (dist1+dist2+dist3)/3
    return avg_dist

def closest_match(audio_archive, input_audio):
    similarity_dict = {}
    for key in audio_archive:
        target_data = audio_archive[key]
        dist = compute_average_distance(target_data, input_audio)
        similarity_dict[key] = dist
        match = min(similarity_dict, key=similarity_dict.get)
    testFiles = glob.glob(r'C:\Users\212313601\Documents\Innovation Station\AI Hackathon 0427\Team Becky.. Kang\voice_files\*test.wav')
    [os.remove(file) for file in testFiles]
    return match

def file_transfer_and_test(ex):
    clear_output(wait=True)
    name=ex.value
    file = glob.glob('C:\Temp\Temp_Chrome\*.wav')
    os.rename(file[0],'voice_files/'+name+'test.wav')
    voice = librosa.load(r'voice_files\\'+name+'test.wav')[0]
    location = str(r'voice_files\\'+name+'test'+'.wav')
    ex.test_value = [voice, location]
    print('Received new audio sample')
    return ex.test_value 
    
def file_transfer(name):
    files = glob.glob('C:\Temp\Temp_Chrome\*.wav')
    count = 0
    for file in files:
        os.rename(file,
                  r'C:\Users\212313601\Documents\Innovation Station\AI Hackathon 0427\Team Becky.. Kang\voice_files\\'+name+str(count)+'.wav')
        count+=1
    return 

def cleanup(ex):
    clear_output(wait=True)
    testFiles = glob.glob(r'C:\Users\212313601\Documents\Innovation Station\AI Hackathon 0427\Team Becky.. Kang\voice_files\*{0}*.wav'.format(ex.value))
    print('Cleaned Files')
    [os.remove(file) for file in testFiles]
    return

def append_data_archive(ex):
    clear_output(wait=True)
    p_identifier = ex.value
    print(p_identifier, 'is being added to the Voice Archive')
    with open('voiceDataArchive.pickle','rb') as handle:
        ex.dict_Value = pickle.load(handle)
    handle.close()
    file_transfer(p_identifier)
    files = glob.glob(r'voice_files\\'+p_identifier+'*.wav')
    new_data = []
    for file in files:
        file_i = librosa.load(file)[0]
        file_i = audio_preprocessing(file_i)
        new_data.append(file_i)
    ex.dict_Value[p_identifier] = new_data
    with open('voiceDataArchive.pickle', 'wb') as handle:
        pickle.dump(ex.dict_Value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print(p_identifier, 'added to the voice archive')
    
def del_key(ex):
    clear_output(wait=True)
    with open('voiceDataArchive.pickle','rb') as handle:
        archive = pickle.load(handle)
    handle.close()
    if ex.value in archive: 
        del archive[ex.value]
        with open('voiceDataArchive.pickle', 'wb') as handle:
            pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
            return print(ex.value, 'has been deleted from the Voice Archive')
    else:
        print(ex.value, "is not enrolled in Voice Authentication")