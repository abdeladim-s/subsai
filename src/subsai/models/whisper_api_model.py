#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper API Model

See [openai/whisper](https://platform.openai.com/docs/guides/speech-to-text)
"""

import os
import ffmpeg
from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config
from openai import OpenAI
from pysubs2 import SSAFile

def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    # Construct the output file name
    filename, ext = os.path.splitext(video_file)
    output_file = f"{filename}.{output_ext}"
    
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
                'description': "One of the official model names listed by `whisper.available_models()`, or "
                               "path to a model checkpoint containing the model dimensions and the model "
                               "state_dict.",
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
        

    def transcribe(self, media_file) -> str:
        audio_file_path = convert_video_to_audio_ffmpeg(media_file)
        audio_file = open(audio_file_path, "rb")
        result = self.client.audio.transcriptions.create(
            model=self.model_type, 
            file=audio_file,
            response_format="srt"
        )
        return SSAFile.from_string(result)

