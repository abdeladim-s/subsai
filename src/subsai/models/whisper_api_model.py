#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper API Model

See [openai/whisper](https://platform.openai.com/docs/guides/speech-to-text)
"""

import os
import ffmpeg
import tempfile
from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config
from openai import OpenAI
from pysubs2 import SSAFile
from pydub import AudioSegment

TMPDIR = tempfile.gettempdir()

def split_filename(filepath):
    path, full_filename = os.path.split(filepath)
    filename, ext = os.path.splitext(full_filename)
    return path,filename,ext

path,filename,ext = split_filename('/Users/luka/Desktop/y2mate.is - AGI Inches Closer 5 Key Quotes Altman Huang and The Most Interesting Year -fPzp_sdCf2Y-1080pp-1711573970.mp3')

def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    # Construct the output file name
    path,filename,ext = split_filename(video_file)
    output_file = os.path.join(TMPDIR,f"{filename}.{output_ext}")
    

    print('Saving audio to {} with ffmpeg...'.format(output_file))
    # Execute the ffmpeg conversion
    (
        ffmpeg
        .input(video_file)
        .output(output_file)
        .overwrite_output()
        .run(quiet=True)
    )

    return output_file


class WhisperAPIModel(AbstractModel):
    model_name = 'openai/whisper'
    config_schema = {
            # load model config
            'model_type': {
                'type': list,
                'description': "OpenAI Whisper API, currently only supports large-v2 which is named as whisper-1/ \
                                There is a 25mb upload limit so audio is chunked locally, this may lead to lower performance.",
                'options': ['whisper-1'],
                'default': 'whisper-1'
            },
            'api_key': {
                'type': str,
                'description': "Your OpenAI API key",
                'options': None,
                'default': None
            },
        }

    def __init__(self, model_config):
        # config
        self.model_type = _load_config('model_type', model_config, self.config_schema)
        self.api_key = _load_config('api_key', model_config, self.config_schema)

        self.client = OpenAI(api_key=self.api_key)

    def chunk_audio(self,audio_file_path) -> list:
        # Load the audio file
        audio = AudioSegment.from_mp3(audio_file_path)

        # Desired chunk size in megabytes (MB)
        chunk_size_mb = 5
        chunk_size_bits = chunk_size_mb * 1024 * 1024 * 8
        bitrate = audio.frame_rate * audio.frame_width
        chunk_duration_ms = ((chunk_size_bits) / bitrate) * 1000

        chunks = []

        # Split the audio into chunks
        current_ms = 0
        while current_ms < len(audio):
            # Calculate the end of the current chunk
            end_ms = current_ms + chunk_duration_ms
            # Create a chunk from the current position to the end position
            chunk = audio[current_ms:int(end_ms)]
            # Add the chunk to the list of chunks
            chunks.append((chunk,current_ms))
            # Update the current position
            current_ms = end_ms

        return chunks
        

    def transcribe(self, media_file) -> str:

        audio_file_path = convert_video_to_audio_ffmpeg(media_file)

        chunks = self.chunk_audio(audio_file_path)

        # Export each chunk as needed
        results = ''

        for i, (chunk,offset) in enumerate(chunks):
            chunk_path = os.path.join(TMPDIR,f'chunk_{i}.mp3')
            print(chunk_path)
            chunk.export(chunk_path, format='mp3')

            audio_file = open(chunk_path, "rb")
            result = self.client.audio.transcriptions.create(
                model=self.model_type, 
                file=audio_file,
                response_format="srt"
            )
            # shift subtitles by offset
            result = SSAFile.from_string(result)
            print('SHIFTING {}'.format(offset))
            result.shift(ms=offset)
            results += result.to_string('srt')

        results = ''.join(results)

        return SSAFile.from_string(results)

