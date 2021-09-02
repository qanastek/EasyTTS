import numpy as np

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

class TTS(object):

    """
    🤗 Apply Text-To-Speech Inference and get Audio content
    """ 
    def __init__(self, lang="fr"):

        self.lang = lang

        if lang not in ["fr"]:
            print("This language isn't supported yet!")

        if self.lang == "fr":                   
            self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-synpaflex-fr")
            self.tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-synpaflex-fr")
            self.mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-synpaflex-fr")

    """
    ⚙️ Make a prediction
    """
    def predict(self, text="Bonjour le monde"):

        if self.lang == "fr":
            return self.frTTS(text)
        else:            
            return None

    """
    French TTS Model
    """
    def frTTS(self,text):

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