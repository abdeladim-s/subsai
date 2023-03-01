#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper Model

See [openai/whisper](https://github.com/openai/whisper)
"""

from typing import Tuple
import pysubs2
from subsai.models.abstract_model import AbstractModel
import whisper
from subsai.utils import _load_config, get_available_devices


class WhisperModel(AbstractModel):
    model_name = 'openai/whisper'
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
                'type':  list,
                'description': "The PyTorch device to put the model into",
                'options': [None, *get_available_devices()],
                'default': None
            },
            'download_root': {
                'type': str,
                'description': "Path to download the model files; by default, it uses '~/.cache/whisper'",
                'options': None,
                'default': None
            },
            'in_memory': {
                'type': bool,
                'description': "whether to preload the model weights into host memory",
                'options': None,
                'default': False
            },
            # transcribe config
            'verbose': {
                'type': bool,
                'description': "Whether to display the text being decoded to the console. "
                               "If True, displays all the details,"
                               "If False, displays minimal details. If None, does not display anything",
                'options': None,
                'default': None
            },
            'temperature': {
                'type': Tuple,
                'description': "Temperature for sampling. It can be a tuple of temperatures, which will be "
                               "successively used upon failures according to either `compression_ratio_threshold` "
                               "or `logprob_threshold`.",
                'options': None,
                'default': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            },
            'compression_ratio_threshold': {
                'type': float,
                'description': "If the gzip compression ratio is above this value, treat as failed",
                'options': None,
                'default': 2.4
            },
            'logprob_threshold': {
                'type': float,
                'description': "If the average log probability over sampled tokens is below this value, treat as failed",
                'options': None,
                'default': -1.0
            },
            'no_speech_threshold': {
                'type': float,
                'description': "If the no_speech probability is higher than this value AND the average log probability "
                               "over sampled tokens is below `logprob_threshold`, consider the segment as silent",
                'options': None,
                'default': 0.6
            },
            'condition_on_previous_text': {
                'type': bool,
                'description': "if True, the previous output of the model is provided as a prompt for the next window; "
                               "disabling may make the text inconsistent across windows, but the model becomes less "
                               "prone to getting stuck in a failure loop, such as repetition looping or timestamps "
                               "going out of sync.",
                'options': None,
                'default': True
            },
            # decode options
            'task': {
                'type': list,
                'description': "whether to perform X->X 'transcribe' or X->English 'translate'",
                'options': ['transcribe', 'translate'],
                'default': 'transcribe'
            },
            'language': {
                'type': str,
                'description': "language that the audio is in; uses detected language if None",
                'options': None,
                'default': None
            },
            'sample_len': {
                'type': int,
                'description': "maximum number of tokens to sample",
                'options': None,
                'default': None
            },
            'best_of': {
                'type': int,
                'description': "number of independent samples to collect, when t > 0",
                'options': None,
                'default': None
            },
            'beam_size': {
                'type': int,
                'description': "number of beams in beam search, when t == 0",
                'options': None,
                'default': None
            },
            'patience': {
                'type': float,
                'description': "patience in beam search (https://arxiv.org/abs/2204.05424)",
                'options': None,
                'default': None
            },
            'length_penalty': {
                'type': float,
                'description': "'alpha' in Google NMT, None defaults to length norm",
                'options': None,
                'default': None
            },
            'prompt': {
                'type': str,
                'description': "text or tokens for the previous context",
                'options': None,
                'default': None
            },
            'prefix': {
                'type': str,
                'description': "text or tokens to prefix the current context",
                'options': None,
                'default': None
            },
            'suppress_blank': {
                'type': bool,
                'description': "this will suppress blank outputs",
                'options': None,
                'default': True
            },
            'suppress_tokens': {
                'type': str,
                'description': 'list of tokens ids (or comma-separated token ids) to suppress "-1" will suppress '
                               'a set of symbols as defined in `tokenizer.non_speech_tokens()`',
                'options': None,
                'default': "-1"
            },
            'without_timestamps': {
                'type': bool,
                'description': 'use <|notimestamps|> to sample text tokens only',
                'options': None,
                'default': False
            },
            'max_initial_timestamp': {
                'type': float,
                'description': 'the initial timestamp cannot be later than this',
                'options': None,
                'default': 1.0
            },
            'fp16': {
                'type': bool,
                'description': 'use fp16 for most of the calculation',
                'options': None,
                'default': True
            },

        }

    def __init__(self, model_config):
        super(WhisperModel, self).__init__(model_config=model_config,
                                           model_name=self.model_name)
        # config
        self.model_type = _load_config('model_type', model_config, self.config_schema)
        self.device = _load_config('device', model_config, self.config_schema)
        self.download_root = _load_config('download_root', model_config, self.config_schema)
        self.in_memory = _load_config('in_memory', model_config, self.config_schema)

        self.verbose = _load_config('verbose', model_config, self.config_schema)
        self.temperature = _load_config('temperature', model_config, self.config_schema)
        self.compression_ratio_threshold = _load_config('compression_ratio_threshold', model_config, self.config_schema)
        self.logprob_threshold = _load_config('logprob_threshold', model_config, self.config_schema)
        self.no_speech_threshold = _load_config('no_speech_threshold', model_config, self.config_schema)
        self.condition_on_previous_text = _load_config('condition_on_previous_text', model_config, self.config_schema)

        self.decode_options = \
            {config: _load_config(config, model_config, self.config_schema)
             for config in self.config_schema if not hasattr(self, config)}

        self.model = whisper.load_model(name=self.model_type,
                                        device=self.device,
                                        download_root=self.download_root,
                                        in_memory=self.in_memory)

    def transcribe(self, media_file) -> str:
        audio = whisper.load_audio(media_file)
        result = self.model.transcribe(audio, verbose=self.verbose,
                                       temperature=self.temperature,
                                       compression_ratio_threshold=self.compression_ratio_threshold,
                                       logprob_threshold=self.logprob_threshold,
                                       no_speech_threshold=self.no_speech_threshold,
                                       condition_on_previous_text=self.condition_on_previous_text,
                                       **self.decode_options)
        subs = pysubs2.load_from_whisper(result)
        return subs

