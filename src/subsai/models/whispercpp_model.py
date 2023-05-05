#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper.cpp Model

See [whisper.cpp](https://github.com/ggerganov/whisper.cpp),
See [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp/)
"""

from typing import Tuple
import pysubs2
from pysubs2 import SSAFile, SSAEvent

from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config, get_available_devices
from pywhispercpp.model import Model
from pywhispercpp.constants import AVAILABLE_MODELS
from _pywhispercpp import WHISPER_SAMPLING_GREEDY, WHISPER_SAMPLING_BEAM_SEARCH


class WhisperCppModel(AbstractModel):
    model_name = 'ggerganov/whisper.cpp'
    config_schema = {
            # load model config
            'model_type': {
                'type': list,
                'description': "Available whisper.cpp models",
                'options': AVAILABLE_MODELS,
                'default': 'base'
            },
            'n_threads': {
                'type': int,
                'description': "Number of threads to allocate for the inference"
                               "default to min(4, available hardware_concurrency)",
                'options': None,
                'default': 4
            },
            'n_max_text_ctx': {
                'type': int,
                'description': "max tokens to use from past text as prompt for the decoder",
                'options': None,
                'default': 16384
            },
            'offset_ms': {
                'type': int,
                'description': "start offset in ms",
                'options': None,
                'default': 0
            },
            'duration_ms': {
                'type': int,
                'description': "audio duration to process in ms",
                'options': None,
                'default': 0
            },
            'translate': {
                'type': bool,
                'description': "whether to translate the audio to English",
                'options': None,
                'default': False
            },
            'no_context': {
                'type': bool,
                'description': "do not use past transcription (if any) as initial prompt for the decoder",
                'options': None,
                'default': False
            },
            'single_segment': {
                'type': bool,
                'description': "force single segment output (useful for streaming)",
                'options': None,
                'default': False
            },
            'print_special': {
                'type': bool,
                'description': "print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)",
                'options': None,
                'default': False
            },
            'print_progress': {
                'type': bool,
                'description': "print progress information",
                'options': None,
                'default': True
            },
            'print_realtime': {
                'type': bool,
                'description': "print results from within whisper.cpp (avoid it, use callback instead)",
                'options': None,
                'default': False
            },
            'print_timestamps': {
                'type': bool,
                'description': "print timestamps for each text segment when printing realtime",
                'options': None,
                'default': True
            },
            # [EXPERIMENTAL] token-level timestamps
            'token_timestamps': {
                'type': bool,
                'description': "enable token-level timestamps",
                'options': None,
                'default': False
            },
            'thold_pt': {
                'type': float,
                'description': "timestamp token probability threshold (~0.01)",
                'options': None,
                'default': 0.01
            },
            'thold_ptsum': {
                'type': float,
                'description': "timestamp token sum probability threshold (~0.01)",
                'options': None,
                'default': 0.01
            },
            'max_len': {
                'type': int,
                'description': "max segment length in characters",
                'options': None,
                'default': 0
            },
            'split_on_word': {
                'type': bool,
                'description': "split on word rather than on token (when used with max_len)",
                'options': None,
                'default': False
            },
            'max_tokens': {
                'type': int,
                'description': "max tokens per segment (0 = no limit)",
                'options': None,
                'default': 0
            },
            # [EXPERIMENTAL] speed-up techniques
            # note: these can significantly reduce the quality of the output
            'speed_up': {
                'type': bool,
                'description': "speed-up the audio by 2x using Phase Vocoder",
                'options': None,
                'default': False
            },
            'audio_ctx': {
                'type': int,
                'description': "overwrite the audio context size (0 = use default)",
                'options': None,
                'default': 0
            },
            'prompt_n_tokens': {
                'type': int,
                'description': "tokens to provide to the whisper decoder as initial prompt",
                'options': None,
                'default': 0
            },
            'language': {
                'type': str,
                'description': 'for auto-detection, set to None, "" or "auto"',
                'options': None,
                'default': 'en'
            },
            'suppress_blank': {
                'type': bool,
                'description': 'common decoding parameters',
                'options': None,
                'default': True
            },
            'suppress_non_speech_tokens': {
                'type': bool,
                'description': 'common decoding parameters',
                'options': None,
                'default': False
            },
            'temperature': {
                'type': float,
                'description': 'initial decoding temperature',
                'options': None,
                'default': 0.0
            },
            'max_initial_ts': {
                'type': float,
                'description': 'max_initial_ts',
                'options': None,
                'default': 1.0
            },
            'length_penalty': {
                'type': float,
                'description': 'length_penalty',
                'options': None,
                'default': -1.0
            },
            'temperature_inc': {
                'type': float,
                'description': 'temperature_inc',
                'options': None,
                'default': 0.2
            },
            'entropy_thold': {
                'type': float,
                'description': 'similar to OpenAI\'s "compression_ratio_threshold"',
                'options': None,
                'default': 2.4
            },
            'logprob_thold': {
                'type': float,
                'description': 'logprob_thold',
                'options': None,
                'default': -1.0
            },
            'no_speech_thold': {  # not implemented
                'type': float,
                'description': 'no_speech_thold',
                'options': None,
                'default': 0.6
            },
            'greedy': {
                'type': dict,
                'description': 'greedy',
                'options': None,
                'default': {"best_of": -1}
            },
            'beam_search': {
                'type': dict,
                'description': 'beam_search',
                'options': None,
                'default': {"beam_size": -1, "patience": -1.0}
            }
        }

    def __init__(self, model_config):
        super(WhisperCppModel, self).__init__(model_config=model_config,
                                           model_name=self.model_name)
        # config
        self.model_type = _load_config('model_type', model_config, self.config_schema)

        self.params = {}
        for config in self.config_schema:
            if not hasattr(self, config):
                config_value = _load_config(config, model_config, self.config_schema)
                if config_value is None:
                    continue
                self.params[config] = config_value

        self.model = Model(model=self.model_type, **self.params)

    def transcribe(self, media_file) -> str:
        segments = self.model.transcribe(media=media_file)
        subs = SSAFile()
        for seg in segments:
            event = SSAEvent(start=seg.t0*10, end=seg.t1*10)
            event.plaintext = seg.text.strip()
            subs.append(event)
        return subs

