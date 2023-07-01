#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Faster Whisper Model

See [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
"""

from typing import Tuple
import pysubs2
import whisper
from pysubs2 import SSAFile, SSAEvent
from tqdm import tqdm

from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config, get_available_devices
from faster_whisper import WhisperModel


class FasterWhisperModel(AbstractModel):
    model_name = 'guillaumekln/faster-whisper'
    config_schema = {
        # load model config
        'model_size_or_path': {
            'type': list,
            'description': 'Size of the model to use (e.g. "large-v2", "small", "tiny.en", etc.)'
                           'or a path to a converted model directory. When a size is configured, the converted'
                           'model is downloaded from the Hugging Face Hub.',
            'options': whisper.available_models(),
            'default': 'base'
        },
        'device': {
            'type': list,
            'description': 'Device to use for computation ("cpu", "cuda", "auto")',
            'options': ['auto', 'cpu', 'cuda'],
            'default': 'auto'
        },
        'device_index': {
            'type': int,
            'description': 'Device ID to use.'
                           'The model can also be loaded on multiple GPUs by passing a list of IDs'
                           '(e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel'
                           'when transcribe() is called from multiple Python threads (see also num_workers).',
            'options': None,
            'default': 0
        },
        'compute_type': {
            'type': str,
            'description': 'Type to use for computation.'
                           'See https://opennmt.net/CTranslate2/quantization.html.',
            'options': None,
            'default': "default"
        },
        'cpu_threads': {
            'type': int,
            'description': 'Number of threads to use when running on CPU (4 by default).'
                           'A non zero value overrides the OMP_NUM_THREADS environment variable.',
            'options': None,
            'default': 0
        },
        'num_workers': {
            'type': int,
            'description': 'When transcribe() is called from multiple Python threads,'
                           'having multiple workers enables true parallelism when running the model'
                           '(concurrent calls to self.model.generate() will run in parallel).'
                           'This can improve the global throughput at the cost of increased memory usage.',
            'options': None,
            'default': 1
        },
        # transcribe config
        'temperature': {
            'type': Tuple,
            'description': "Temperature for sampling. It can be a tuple of temperatures, which will be "
                           "successively used upon failures according to either `compression_ratio_threshold` "
                           "or `logprob_threshold`.",
            'options': None,
            'default': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        'compression_ratio_threshold': {
            'type': float,
            'description': "If the gzip compression ratio is above this value, treat as failed",
            'options': None,
            'default': 2.4
        },
        'log_prob_threshold': {
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
        # # decode options
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
        'best_of': {
            'type': int,
            'description': "number of independent samples to collect, when t > 0",
            'options': None,
            'default': 5
        },
        'beam_size': {
            'type': int,
            'description': "number of beams in beam search, when t == 0",
            'options': None,
            'default': 5
        },
        'patience': {
            'type': float,
            'description': "patience in beam search (https://arxiv.org/abs/2204.05424)",
            'options': None,
            'default': 1.
        },
        'length_penalty': {
            'type': float,
            'description': "'alpha' in Google NMT, None defaults to length norm",
            'options': None,
            'default': 1.
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
            'type': Tuple,
            'description': 'list of tokens ids (or comma-separated token ids) to suppress "-1" will suppress '
                           'a set of symbols as defined in `tokenizer.non_speech_tokens()`',
            'options': None,
            'default': [-1]
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
        # Faster-whisper configs
        'initial_prompt': {
            'type': str,
            'description': 'Optional text to provide as a prompt for the first window.',
            'options': None,
            'default': None
        },
        'word_timestamps': {
            'type': bool,
            'description': 'Extract word-level timestamps using the cross-attention pattern'
                           'and dynamic time warping, and include the timestamps for each word in each segment.',
            'options': None,
            'default': False
        },
        'prepend_punctuations': {
            'type': str,
            'description': 'If word_timestamps is True, merge these punctuation symbols'
                           'with the next word',
            'options': None,
            'default': "\"'“¿([{-"
        },
        'append_punctuations': {
            'type': str,
            'description': 'If word_timestamps is True, merge these punctuation symbols'
                           'with the previous word',
            'options': None,
            'default': "\"'.。,，!！?？:：”)]}、"
        },
        'vad_filter': {
            'type': bool,
            'description': 'If True, use the integrated Silero VAD model to filter out parts of the audio without speech.',
            'options': None,
            'default': False
        },
        'vad_parameters': {
            'type': dict,
            'description': 'Parameters for splitting long audios into speech chunks using silero VAD.',
            'options': None,
            'default': {
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'max_speech_duration_s': float('inf'),
                'min_silence_duration_ms': 2000,
                'window_size_samples': 1024,
                'speech_pad_ms': 400
            }
        },
    }

    def __init__(self, model_config):
        super(FasterWhisperModel, self).__init__(model_config=model_config,
                                           model_name=self.model_name)
        # config
        self._model_size_or_path = _load_config('model_size_or_path', model_config, self.config_schema)
        self._device = _load_config('device', model_config, self.config_schema)
        self._device_index = _load_config('device_index', model_config, self.config_schema)
        self._compute_type = _load_config('compute_type', model_config, self.config_schema)
        self._cpu_threads = _load_config('cpu_threads', model_config, self.config_schema)
        self._num_workers = _load_config('num_workers', model_config, self.config_schema)

        self.transcribe_configs = \
            {config: _load_config(config, model_config, self.config_schema)
             for config in self.config_schema if not hasattr(self, f"_{config}")}

        self.model = WhisperModel(model_size_or_path=self._model_size_or_path,
                                  device=self._device,
                                  device_index=self._device_index,
                                  compute_type=self._compute_type,
                                  cpu_threads=self._cpu_threads,
                                  num_workers=self._num_workers)


        # to show the progress
        import logging

        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    def transcribe(self, media_file) -> str:
        segments, info = self.model.transcribe(media_file, **self.transcribe_configs)
        subs = SSAFile()
        total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
        timestamps = 0.0  # to get the current segments
        with tqdm(total=total_duration, unit=" audio seconds") as pbar:
            if self.transcribe_configs['word_timestamps']:  # word level timestamps
                for segment in segments:
                    pbar.update(segment.end - timestamps)
                    timestamps = segment.end
                    if timestamps < info.duration:
                        pbar.update(info.duration - timestamps)
                    for word in segment.words:
                        event = SSAEvent(start=pysubs2.make_time(s=word.start), end=pysubs2.make_time(s=word.end))
                        event.plaintext = word.word.strip()
                        subs.append(event)
            else:
                for segment in segments:
                    pbar.update(segment.end - timestamps)
                    timestamps = segment.end
                    if timestamps < info.duration:
                        pbar.update(info.duration - timestamps)
                    event = SSAEvent(start=pysubs2.make_time(s=segment.start), end=pysubs2.make_time(s=segment.end))
                    event.plaintext = segment.text.strip()
                    subs.append(event)

        return subs
