import numpy as np
import soundfile as sf

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

from EasyTTS.utils import Toolbox

class TTS:

    """
    🤗 Constructor for the Text-To-Speech Inference Method
    """
    def __init__(self, lang="fr", text="Bonjour le monde"):

        if lang not in Toolbox.SUPPORTED_LANGUAGES:
            print("This language isn't supported yet!")

        if lang == "fr":
            self.frTTS(text)

        elif lang == "en":
            self.frTTS(text)

    """
    French TTS Model
    """
    def frTTS(text):
        
        processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-synpaflex-fr")
        tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-synpaflex-fr")
        mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-synpaflex-fr")

        text = "Comme le capitaine prononçait ces mots, un éclair illumina les ondes de l'Atlantique, puis une détonation se fit entendre et deux boulets ramés balayèrent le pont de l'Alcyon."

        input_ids = processor.text_to_sequence(text)

        # tacotron2 inference (text-to-mel)
        decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )

        # melgan inference (mel-to-wav)
        audio = mb_melgan.inference(mel_outputs)[0, :, 0]

        return audio