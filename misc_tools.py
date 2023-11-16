from IPython.display import Audio, display

class MyAudio: 
    @staticmethod
    def play_audio_torch(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=sample_rate))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported.")
        
    @staticmethod
    def play_audio_np(waveform, sample_rate):
        display(Audio(waveform[0], rate=sample_rate))