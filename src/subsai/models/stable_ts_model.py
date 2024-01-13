#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable-ts Model

See [jianfch/stable-ts](https://github.com/jianfch/stable-ts)
"""
import logging
from typing import Tuple
import pysubs2
import whisper
from pysubs2 import SSAFile, SSAEvent

from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config, get_available_devices
from stable_whisper.whisper_word_level import transcribe_stable, load_model


class StableTsModel(AbstractModel):
    model_name = 'jianfch/stable-ts'
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
        'in_memory': {
            'type': bool,
            'description': "bool, default False, Whether to preload the model weights into host memory.",
            'options': None,
            'default': False
        },
        'cpu_preload': {
            'type': bool,
            'description': "Load model into CPU memory first then move model to specified device to reduce GPU memory usage when loading model",
            'options': None,
            'default': True
        },
        'dq': {
            'type': bool,
            'description': "Whether to apply Dynamic Quantization to model to reduced memory usage and increase "
                           "inference speed but at the cost of a slight decrease in accuracy. Only for CPU.",
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
            'default': True
        },
        'regroup': {
            'type': bool,
            'description': "default True, meaning the default regroup algorithm"
                           "String for customizing the regrouping algorithm. False disables regrouping."
                           "Ignored if ``word_timestamps = False``.",
            'options': None,
            'default': True
        },
        'ts_num': {
            'type': int,
            'description': "meaning disable this option"
                           "Number of extra timestamp inferences to perform then use average of these extra timestamps."
                           "An experimental option that might hurt performance.",
            'options': None,
            'default': 0
        },
        'ts_noise': {
            'type': float,
            'description': "Percentage of noise to add to audio_features to perform inferences for ``ts_num``.",
            'options': None,
            'default': 0.1
        },
        'suppress_silence': {
            'type': bool,
            'description': "Whether to enable timestamps adjustments based on the detected silence.",
            'options': None,
            'default': True
        },
        'suppress_word_ts': {
            'type': bool,
            'description': "Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.",
            'options': None,
            'default': True
        },
        'q_levels': {
            'type': int,
            'description': "Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``."
                           "Acts as a threshold to marking sound as silent."
                           "Fewer levels will increase the threshold of volume at which to mark a sound as silent.",
            'options': None,
            'default': 20
        },
        'k_size': {
            'type': int,
            'description': "Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``."
                           "Recommend 5 or 3; higher sizes will reduce detection of silence.",
            'options': None,
            'default': 5
        },
        'time_scale': {
            'type': float,
            'description': "Factor for scaling audio duration for inference."
                           "Greater than 1.0 'slows down' the audio, and less than 1.0 'speeds up' the audio. None is same as 1.0."
                           "A factor of 1.5 will stretch 10s audio to 15s for inference. This increases the effective resolution"
                           "of the model but can increase word error rate.",
            'options': None,
            'default': None
        },
        'demucs': {
            'type': bool,
            'description': "Whether to preprocess ``audio`` with Demucs to isolate vocals / remove noise. Set ``demucs`` to an instance of"
                           "a Demucs model to avoid reloading the model for each run."
                           "Demucs must be installed to use. Official repo. https://github.com/facebookresearch/demucs.",
            'options': None,
            'default': False
        },
        'demucs_output': {
            'type': str,
            'description': "Path to save the vocals isolated by Demucs as WAV file. Ignored if ``demucs = False``."
                           "Demucs must be installed to use. Official repo. https://github.com/facebookresearch/demucs.",
            'options': None,
            'default': None
        },
        'demucs_options': {
            'type': dict,
            'description': "Options to use for :func:`stable_whisper.audio.demucs_audio`.",
            'options': None,
            'default': None
        },
        'vad': {
            'type': bool,
            'description': "Whether to use Silero VAD to generate timestamp suppression mask."
                           "Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.",
            'options': None,
            'default': False
        },
        'vad_threshold': {
            'type': float,
            'description': "Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.",
            'options': None,
            'default': 0.35
        },
        'vad_onnx': {
            'type': bool,
            'description': "Whether to use ONNX for Silero VAD.",
            'options': None,
            'default': False
        },
        'min_word_dur': {
            'type': float,
            'description': "Shortest duration each word is allowed to reach for silence suppression.",
            'options': None,
            'default': 0.1
        },
        'only_voice_freq': {
            'type': bool,
            'description': "Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.",
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
        'mel_first': {
            'type': bool,
            'description': "Process entire audio track into log-Mel spectrogram first instead in chunks."
                           "Used if odd behavior seen in stable-ts but not in whisper, but use significantly more memory for long audio.",
            'options': None,
            'default': False
        },
        'suppress_ts_tokens': {
            'type': bool,
            'description': " Whether to suppress timestamp tokens during inference for timestamps are detected at silent."
                           "Reduces hallucinations in some cases, but also prone to ignore disfluencies and repetitions."
                           "This option is ignored if ``suppress_silence = False``.",
            'options': None,
            'default': False
        },
        'gap_padding': {
            'type': str,
            'description': "Padding prepend to each segments for word timing alignment."
                           "Used to reduce the probability of model predicting timestamps earlier than the first utterance.",
            'options': None,
            'default': '...'
        },
        'only_ffmpeg': {
            'type': bool,
            'description': "Whether to use only FFmpeg (instead of not yt-dlp) for URls",
            'options': None,
            'default': False
        },
        'max_instant_words': {
            'type': float,
            'description': "If percentage of instantaneous words in a segment exceed this amount, the segment is removed.",
            'options': None,
            'default': 0.5
        },
        'avg_prob_threshold': {
            'type': float,
            'description': "Transcribe the gap after the previous word and if the average word proababiliy of a segment falls below this"
                           "value, discard the segment. If ``None``, skip transcribing the gap to reduce chance of timestamps starting"
                           "before the next utterance.",
            'options': None,
            'default': None
        },
        'ignore_compatibility': {
            'type': bool,
            'description': "Whether to ignore warnings for compatibility issues with the detected Whisper version.",
            'options': None,
            'default': False
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
        # Faster-whisper configs
        # 'vad_filter': {
        #     'type': bool,
        #     'description': 'If True, use the integrated Silero VAD model to filter out parts of the audio without speech.',
        #     'options': None,
        #     'default': False
        # },
        # 'vad_parameters': {
        #     'type': dict,
        #     'description': 'Parameters for splitting long audios into speech chunks using silero VAD.',
        #     'options': None,
        #     'default': {
        #         'threshold': 0.5,
        #         'min_speech_duration_ms': 250,
        #         'max_speech_duration_s': float('inf'),
        #         'min_silence_duration_ms': 2000,
        #         'window_size_samples': 1024,
        #         'speech_pad_ms': 400
        #     }
        # },
    }

    def __init__(self, model_config):
        super(StableTsModel, self).__init__(model_config=model_config,
                                            model_name=self.model_name)
        # config
        self._model_type = _load_config('model_type', model_config, self.config_schema)
        self._device = _load_config('device', model_config, self.config_schema)
        self._in_memory = _load_config('in_memory', model_config, self.config_schema)
        self._cpu_preload = _load_config('cpu_preload', model_config, self.config_schema)
        self._dq = _load_config('dq', model_config, self.config_schema)

        self._verbose = _load_config('verbose', model_config, self.config_schema)
        self._temperature = _load_config('temperature', model_config, self.config_schema)
        self._compression_ratio_threshold = _load_config('compression_ratio_threshold', model_config, self.config_schema)
        self._logprob_threshold = _load_config('logprob_threshold', model_config, self.config_schema)
        self._no_speech_threshold = _load_config('no_speech_threshold', model_config, self.config_schema)
        self._condition_on_previous_text = _load_config('condition_on_previous_text', model_config, self.config_schema)
        self._initial_prompt = _load_config('initial_prompt', model_config, self.config_schema)
        self._word_timestamps = _load_config('word_timestamps', model_config, self.config_schema)
        self._regroup = _load_config('regroup', model_config, self.config_schema)
        self._ts_num = _load_config('ts_num', model_config, self.config_schema)
        self._ts_noise = _load_config('ts_noise', model_config, self.config_schema)
        self._suppress_silence = _load_config('suppress_silence', model_config, self.config_schema)
        self._suppress_word_ts = _load_config('suppress_word_ts', model_config, self.config_schema)
        self._q_levels = _load_config('q_levels', model_config, self.config_schema)
        self._k_size = _load_config('k_size', model_config, self.config_schema)
        self._time_scale = _load_config('time_scale', model_config, self.config_schema)
        self._demucs = _load_config('demucs', model_config, self.config_schema)
        self._demucs_output = _load_config('demucs_output', model_config, self.config_schema)
        self._demucs_options = _load_config('demucs_options', model_config, self.config_schema)
        self._vad = _load_config('vad', model_config, self.config_schema)
        self._vad_threshold = _load_config('vad_threshold', model_config, self.config_schema)
        self._vad_onnx = _load_config('vad_onnx', model_config, self.config_schema)
        self._min_word_dur = _load_config('min_word_dur', model_config, self.config_schema)
        self._only_voice_freq = _load_config('only_voice_freq', model_config, self.config_schema)
        self._prepend_punctuations = _load_config('prepend_punctuations', model_config, self.config_schema)
        self._append_punctuations = _load_config('append_punctuations', model_config, self.config_schema)
        self._mel_first = _load_config('mel_first', model_config, self.config_schema)
        self._suppress_ts_tokens = _load_config('suppress_ts_tokens', model_config, self.config_schema)
        self._gap_padding = _load_config('gap_padding', model_config, self.config_schema)
        self._only_ffmpeg = _load_config('only_ffmpeg', model_config, self.config_schema)
        self._max_instant_words = _load_config('max_instant_words', model_config, self.config_schema)
        self._avg_prob_threshold = _load_config('avg_prob_threshold', model_config, self.config_schema)
        self._ignore_compatibility = _load_config('ignore_compatibility', model_config, self.config_schema)

        self.transcribe_configs = \
            {config: _load_config(config, model_config, self.config_schema)
             for config in self.config_schema if not hasattr(self, f"_{config}")}

        self.model = load_model(name=self._model_type,
                                device=self._device,
                                in_memory=self._in_memory,
                                cpu_preload=self._cpu_preload,
                                dq=self._dq)

        # self.model = load_faster_whisper(model_size_or_path=self._model_type)

        # to show the progress
        # import logging
        #
        # logging.basicConfig()
        # logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    def transcribe(self, media_file) -> SSAFile:
        result = transcribe_stable(self.model,
                                   audio=media_file,
                                   verbose=self._verbose,
                                   temperature=self._temperature,
                                   compression_ratio_threshold=self._compression_ratio_threshold,
                                   logprob_threshold=self._logprob_threshold,
                                   no_speech_threshold=self._no_speech_threshold,
                                   condition_on_previous_text=self._condition_on_previous_text,
                                   initial_prompt=self._initial_prompt,
                                   word_timestamps=self._word_timestamps,
                                   regroup=self._regroup,
                                   ts_num=self._ts_num,
                                   ts_noise=self._ts_noise,
                                   suppress_silence=self._suppress_silence,
                                   suppress_word_ts=self._suppress_word_ts,
                                   q_levels=self._q_levels,
                                   k_size=self._k_size,
                                   time_scale=self._time_scale,
                                   demucs=self._demucs,
                                   demucs_output=self._demucs_output,
                                   demucs_options=self._demucs_options,
                                   vad=self._vad,
                                   vad_threshold=self._vad_threshold,
                                   vad_onnx=self._vad_onnx,
                                   min_word_dur=self._min_word_dur,
                                   only_voice_freq=self._only_voice_freq,
                                   prepend_punctuations=self._prepend_punctuations,
                                   append_punctuations=self._append_punctuations,
                                   mel_first=self._mel_first,
                                   suppress_ts_tokens=self._suppress_ts_tokens,
                                   gap_padding=self._gap_padding,
                                   only_ffmpeg=self._only_ffmpeg,
                                   max_instant_words=self._max_instant_words,
                                   avg_prob_threshold=self._avg_prob_threshold,
                                   ignore_compatibility=self._ignore_compatibility,
                                   **self.transcribe_configs,
                                   )

        subs = SSAFile()

        if self._word_timestamps:  # word level timestamps
            for segment in result.segments:
                for word in segment.words:
                    try:
                        event = SSAEvent(start=pysubs2.make_time(s=word.start), end=pysubs2.make_time(s=word.end))
                        event.plaintext = word.word.strip()
                        subs.append(event)
                    except Exception as e:
                        logging.warning(f"Something wrong with {word}")
                        logging.warning(e)

        else:
            for segment in result.segments:
                event = SSAEvent(start=pysubs2.make_time(s=segment.start), end=pysubs2.make_time(s=segment.end))
                event.plaintext = segment.text.strip()
                subs.append(event)


        return subs
