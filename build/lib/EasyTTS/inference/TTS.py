import numpy as np

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

class TTS(object):

    """
    🤗 Apply Text-To-Speech Inference and get Audio content
    """ 
    def __init__(self, lang="fr"):

        self.lang = lang.lower()

        self.languages = {
            "fr": {
                "processor": "tensorspeech/tts-tacotron2-synpaflex-fr",
                "model": "tensorspeech/tts-tacotron2-synpaflex-fr",
                "melgan": "tensorspeech/tts-mb_melgan-synpaflex-fr",
            },
            "ger": {
                "processor": "tensorspeech/tts-tacotron2-thorsten-ger",
                "model": "tensorspeech/tts-tacotron2-thorsten-ger",
                "melgan": "tensorspeech/tts-mb_melgan-thorsten-ger",
            },
            "en": {
                "processor": "tensorspeech/tts-tacotron2-ljspeech-en",
                "model": "tensorspeech/tts-tacotron2-ljspeech-en",
                "melgan": "tensorspeech/tts-mb_melgan-ljspeech-en",
            },
            "ch": {
                "processor": "tensorspeech/tts-tacotron2-baker-ch",
                "model": "tensorspeech/tts-tacotron2-baker-ch",
                "melgan": "tensorspeech/tts-mb_melgan-baker-ch",
            },
        }

        if self.lang not in self.languages:
            print("This language isn't supported yet!")
                
        self.processor = AutoProcessor.from_pretrained(self.languages[lang]["processor"])
        self.tacotron2 = TFAutoModel.from_pretrained(self.languages[lang]["model"])
        self.mb_melgan = TFAutoModel.from_pretrained(self.languages[lang]["melgan"])

    """
    ⚙️ Make a prediction
    """
    def predict(self, text="Bonjour le monde"):

        if self.lang in self.languages:
            print("Prediction started!")
            return self.predictTacotron2(text)
        else:            
            return None

    """
    TTS Model Tacotron2
    """
    def predictTacotron2(self,text):

        input_ids = self.processor.text_to_sequence(text)

        # tacotron2 inference (text-to-mel)
        decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )

        # melgan inference (mel-to-wav)
        audio = self.mb_melgan.inference(mel_outputs)[0, :, 0]

        return audio