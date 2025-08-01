import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

# THIS CODE IS FOR EVALUATION SPECIALLY FOR GENERATING GRAPH
# import numpy as np
# import tensorflow as tf
# from hparams import hparams
# from models import create_model
# from text import text_to_sequence


# class Synthesizer:
#   def load(self, checkpoint_path, model_name='tacotron'):
#     print('Constructing model: %s' % model_name)
    
#     self.inputs = tf.placeholder(tf.int32, [1, None], name='inputs')
#     self.input_lengths = tf.placeholder(tf.int32, [1], name='input_lengths')

#     with tf.variable_scope('model'):
#       self.model = create_model(model_name, hparams)
#       self.model.initialize(self.inputs, self.input_lengths)

#     self.mel_outputs = self.model.mel_outputs  # [1, T, mel_dim]

#     print('Loading checkpoint: %s' % checkpoint_path)
#     self.session = tf.Session()
#     self.session.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(self.session, checkpoint_path)

#   def synthesize_mel(self, text):
#     cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
#     seq = text_to_sequence(text, cleaner_names)
    
#     feed_dict = {
#       self.inputs: [np.asarray(seq, dtype=np.int32)],
#       self.input_lengths: np.asarray([len(seq)], dtype=np.int32)
#     }

#     mel = self.session.run(self.mel_outputs, feed_dict=feed_dict)
#     return mel[0]  # [T, mel_dim]


#  python eval.py --checkpoint /path/to/checkpoints/model.ckpt-77700python -u "e:\newtacotron\tacotron\our_code\spectrogram_seperator.py"    