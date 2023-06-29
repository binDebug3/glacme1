import matplotlib.pyplot as plt
import numpy as np
import wave

from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import os


def visualize_audio(audio_data):
    # Save the audio data to a temporary WAV file
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as file:
        file.write(audio_data.getbuffer())

    # Read the temporary WAV file using soundfile
    audio_array, sample_rate = sf.read(temp_file)

    # Calculate time axis
    duration = len(audio_array) / sample_rate
    time = np.linspace(0, duration, num=len(audio_array))

    # Determine the number of channels
    num_channels = audio_array.shape[1] if len(audio_array.shape) > 1 else 1

    # Plot the audio waveform for each channel
    plt.figure(figsize=(10, 4))
    if num_channels == 1:
        plt.plot(time, audio_array, label='Mono')
    else:
        for channel in range(num_channels):
            plt.plot(time, audio_array[:, channel], label=f'Channel {channel + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Remove the temporary WAV file
    os.remove(temp_file)


def get_audio_duration(audio_data):
    # Open the WAV file using the BytesIO object
    audio_file = wave.open(audio_data, 'rb')

    # Get the number of frames and the frame rate
    num_frames = audio_file.getnframes()
    frame_rate = audio_file.getframerate()

    # Calculate the duration in seconds
    duration = num_frames / frame_rate

    # Close the audio file
    audio_file.close()

    return duration

def play_audio(wav_data):
    audio_segment = AudioSegment.from_file(wav_data, format="wav")
    print("Playing audio...")
    play(audio_segment)
    print("Audio finished")