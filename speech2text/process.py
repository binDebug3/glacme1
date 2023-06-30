import numpy as np
import soundfile as sf
import wave
from pydub import AudioSegment

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
    infiles = [existing_file, temp_file]
    outfile = "rolling_audio.wav"

    if not os.path.exists(existing_file):
        shutil.copy2(temp_file, outfile)
        return

    audio_segments = []
    sample_width = None
    sample_rate = None
    num_channels = None

    for infile in infiles:
        with wave.open(infile, 'rb') as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())

            if sample_width is None:
                sample_width = params.sampwidth
            if sample_rate is None:
                sample_rate = params.framerate
            if num_channels is None:
                num_channels = params.nchannels

            # Check if sample rate and number of channels match
            if (
                sample_width != params.sampwidth
                or sample_rate != params.framerate
                or num_channels != params.nchannels
            ):
                print("Sample rate, number of channels, or sample width mismatch. Skipping file:", infile)
                continue

            audio_segments.append(AudioSegment(
                data=frames,
                sample_width=params.sampwidth,
                frame_rate=params.framerate,
                channels=params.nchannels
            ))

    output = audio_segments[0]
    for i in range(1, len(audio_segments)):
        output += audio_segments[i]

    output.export(outfile, format="wav")


def cut_wav(audiofile, size):
    if size is None:
        size = 7  # Default size in seconds

        # Open the input audio file
    with wave.open(audiofile, 'rb') as audio_file:
        # Get the sample width, number of channels, and sample rate
        sample_width = audio_file.getsampwidth()
        num_channels = audio_file.getnchannels()
        sample_rate = audio_file.getframerate()

        # Calculate the total number of frames in the audio
        total_frames = audio_file.getnframes()

        # Calculate the number of frames to keep based on the desired size
        num_frames = int(size * sample_rate)

        # Calculate the starting frame index for the desired duration
        start_frame = max(total_frames - num_frames, 0)

        # Set the position in the audio file to the starting frame
        audio_file.setpos(start_frame)

        # Read the audio data from the starting frame until the end
        audio_data = audio_file.readframes(total_frames - start_frame)

        # Create a new temporary file with the trimmed audio
    temp_file = "temp_audio.wav"
    with wave.open(temp_file, 'wb') as temp_audio:
        temp_audio.setnchannels(num_channels)
        temp_audio.setsampwidth(sample_width)
        temp_audio.setframerate(sample_rate)
        temp_audio.writeframes(audio_data)

    # Replace the original file with the trimmed audio
    shutil.move(temp_file, audiofile)