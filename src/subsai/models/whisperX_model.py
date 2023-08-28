#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WhisperX Model

See [m-bain/whisperX](https://github.com/m-bain/whisperX)
"""
import logging
from typing import Tuple
import pysubs2
import torch

from subsai.models.abstract_model import AbstractModel
import whisper
import whisperx
from subsai.utils import _load_config, get_available_devices
import gc
from pysubs2 import SSAFile, SSAEvent


class WhisperXModel(AbstractModel):
    model_name = 'm-bain/whisperX'
    config_schema = {
        # load model config
        'model_type': {
            'type': list,
            'description': "One of the official model names listed by `whisper.available_models()`, or "
                           "path to a model checkpoint containing the model dimensions and the model "
                           "state_dict.",
            'options': whisper.available_models(),
            'default': 'base'
        },
        'device': {
            'type': list,
            'description': "The PyTorch device to put the model into",
            'options': [None, *get_available_devices()],
            'default': None
        },
        'compute_type': {
            'type': list,
            'description': "change to 'int8' if low on GPU mem (may reduce accuracy)",
            'options': ["default", "float16", 'int8'],
            'default': "default"
        },
        'download_root': {
            'type': str,
            'description': "Path to download the model files; by default, it uses '~/.cache/whisper'",
            'options': None,
            'default': None
        },
        'language': {
            'type': str,
            'description': "language that the audio is in; uses detected language if None",
            'options': None,
            'default': None
        },
        'segment_type': {
            'type': list,
            'description': "Word-level timestamps, "
                           "Choose here between sentence-level and word-level",
            'options': ['sentence', 'word'],
            'default': 'sentence'
        },
        # transcribe config
        'batch_size': {
            'type': int,
            'description': "reduce if low on GPU mem",
            'options': None,
            'default': 16
        },
        'return_char_alignments': {
            'type': bool,
            'description': "Whether to return char alignments",
            'options': None,
            'default': False
        },
        'speaker_labels': {
            'type': bool,
            'description': "Run Diarization Pipeline",
            'options': None,
            'default': False
        },
        'HF_TOKEN': {
            'type': str,
            'description': "if speaker labels is True, you will need Hugging Face access token to use the diarization "
                           "models, https://github.com/m-bain/whisperX#speaker-diarization",
            'options': None,
            'default': None
        },
        'min_speakers': {
            'type': int,
            'description': "min speakers",
            'options': None,
            'default': None
        },
        'max_speakers': {
            'type': int,
            'description': "max speakers",
            'options': None,
            'default': None
        }
    }

    def __init__(self, model_config):
        super(WhisperXModel, self).__init__(model_config=model_config,
                                            model_name=self.model_name)
        # config
        self.model_type = _load_config('model_type', model_config, self.config_schema)
        self.device = _load_config('device', model_config, self.config_schema)
        self.compute_type = _load_config('compute_type', model_config, self.config_schema)
        self.download_root = _load_config('download_root', model_config, self.config_schema)
        self.language = _load_config('language', model_config, self.config_schema)
        self.segment_type = _load_config('segment_type', model_config, self.config_schema)
        # transcribe config
        self.batch_size = _load_config('batch_size', model_config, self.config_schema)
        self.return_char_alignments = _load_config('return_char_alignments', model_config, self.config_schema)
        self.speaker_labels = _load_config('speaker_labels', model_config, self.config_schema)
        self.HF_TOKEN = _load_config('HF_TOKEN', model_config, self.config_schema)
        self.min_speakers = _load_config('min_speakers', model_config, self.config_schema)
        self.max_speakers = _load_config('max_speakers', model_config, self.config_schema)

        self.model = whisperx.load_model(self.model_type,
                                         device=self.device,
                                         compute_type=self.compute_type,
                                         download_root=self.download_root,
                                         language=self.language)

    def transcribe(self, media_file) -> str:
        audio = whisperx.load_audio(media_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device,
                                return_char_alignments=self.return_char_alignments)
        self._clear_gpu()
        del model_a
        if self.speaker_labels:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.HF_TOKEN, device=self.device)
            diarize_segments = diarize_model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            self._clear_gpu()
            del diarize_model

        subs = SSAFile()

        if self.segment_type == 'word':  # word level timestamps
            for segment in result['segments']:
                for word in segment['words']:
                    try:
                        event = SSAEvent(start=pysubs2.make_time(s=word["start"]), end=pysubs2.make_time(s=word["end"]),
                                         name=segment["speaker"] if self.speaker_labels else "")
                        event.plaintext = word["word"].strip()
                        subs.append(event)
                    except Exception as e:
                        logging.warning(f"Something wrong with {word}")
                        logging.warning(e)

        elif self.segment_type == 'sentence':
            for segment in result['segments']:
                event = SSAEvent(start=pysubs2.make_time(s=segment["start"]), end=pysubs2.make_time(s=segment["end"]),
                                 name=segment["speaker"] if self.speaker_labels else "")
                event.plaintext = segment["text"].strip()
                subs.append(event)
        else:
            raise Exception(f'Unknown `segment_type` value, it should be one of the following: '
                            f' {self.config_schema["segment_type"]["options"]}')
        return subs

    def _clear_gpu(self):
        gc.collect()
        torch.cuda.empty_cache()
