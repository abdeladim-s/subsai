#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hugging Face Model

See [automatic-speech-recognition](https://huggingface.co/tasks/automatic-speech-recognition)
"""

import pysubs2
from pysubs2 import SSAFile, SSAEvent
from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config, get_available_devices

from transformers import pipeline


devices = get_available_devices()

class HuggingFaceModel(AbstractModel):
    model_name = 'HuggingFaceModel'
    config_schema = {
        # load model config
        'model_id': {
            'type': str,
            'description': 'The model id from the Hugging Face Hub.',
            'options': None,
            'default': 'openai/whisper-tiny'
        },
        'device': {
            'type': list,
            'description': 'Pytorch device',
            'options': devices,
            'default': devices[0]
        },
        'segment_type': {
            'type': list,
            'description': "Sentence-level or word-level timestamps",
            'options': ['sentence', 'word'],
            'default': 'sentence'
        },
        'chunk_length_s': {
            'type': float,
            'description': '(`float`, *optional*, defaults to 0):'
                           'The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default).',
            'options': None,
            'default': 30
        }
    }

    def __init__(self, model_config):
        super(HuggingFaceModel, self).__init__(model_config=model_config,
                                               model_name=self.model_name)
        # config
        self._model_id = _load_config('model_id', model_config, self.config_schema)
        self._device = _load_config('device', model_config, self.config_schema)
        self.segment_type = _load_config('segment_type', model_config, self.config_schema)
        self._chunk_length_s = _load_config('chunk_length_s', model_config, self.config_schema)


        self.model = pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            device=self._device,
        )

    def transcribe(self, media_file):
        results = self.model(
            media_file,
            chunk_length_s=self._chunk_length_s,
            return_timestamps=True if self.segment_type == 'sentence' else 'word',
        )
        subs = SSAFile()
        for chunk in results['chunks']:
            event = SSAEvent(start=pysubs2.make_time(s=chunk['timestamp'][0]),
                             end=pysubs2.make_time(s=chunk['timestamp'][1]))
            event.plaintext = chunk['text']
            subs.append(event)
        return subs
