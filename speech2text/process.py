import numpy as np
import soundfile as sf
import wave
import shutil
import os

def remove_silence(audio_file, energy_threshold):
    # Read the audio file
    audio_data, sample_rate = sf.read(audio_file)

    # remove the silence at the beginning
    start_index = np.where(audio_data > energy_threshold)[0][0]
    audio_data = audio_data[start_index:]

    # remove the silence at the end
    cons = 1600
    end_index = len(audio_data)
    for i in range(end_index - cons + 1):
        if np.all(audio_data[i:i + cons] < energy_threshold):
            audio_data = audio_data[:i]
            end_index = i
            break

    # Write the modified audio data back to the file
    sf.write(audio_file, audio_data, sample_rate, format='WAV')

    # Calculate the duration of removed portions
    removed_start_duration = start_index / sample_rate
    removed_end_duration = (len(audio_data) - end_index) / sample_rate

    return removed_start_duration, removed_end_duration


def append_to_wav(existing_file, temp_file):
    valid = False
    if os.path.exists(existing_file):
        if os.path.getsize(existing_file) == 0:
            os.remove(existing_file)
        else:
            valid = True
            # Open the existing .wav file in read mode
            existing_audio = wave.open(existing_file, 'rb')

            # Create a new .wav file to append the data
            params = existing_audio.getparams()
            appended_file = wave.open('appended.wav', 'wb')
            appended_file.setparams(params)

            # Read the existing audio data and write it to the appended file
            appended_file.writeframes(existing_audio.readframes(existing_audio.getnframes()))
            existing_audio.close()
    if not valid:
        # Create a new .wav file with the specified file name
        # TODO doesn't work
        appended_file = wave.open(existing_file, 'wb')

    # Open the temporary .wav file in read mode
    temp_audio = wave.open(temp_file, 'rb')
    appended_file.setnchannels(temp_audio.getnchannels())
    appended_file.writeframes(temp_audio.readframes(temp_audio.getnframes()))

    # Close the audio files
    temp_audio.close()
    appended_file.close()

    # Move the appended file to replace the existing file if it existed
    if os.path.exists(existing_file):
        shutil.move('rolling_audio.wav', existing_file)