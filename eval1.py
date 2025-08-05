# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf
# from datetime import datetime
# import traceback

# from hparams import hparams, hparams_debug_string
# from models import create_model
# from datasets.datafeeder import TestFeeder
# from util import audio, infolog, plot
# from text import sequence_to_text

# log = infolog.log

# def time_string():
#     return datetime.now().strftime('%Y-%m-%d %H:%M')


# def evaluate(log_dir, args):
#     # tsv_path = os.path.join(args.base_dir, args.input)
#     tsv_path = r'E:\newtacotron\tacotron\nepali\formatted_test.tsv'
#     checkpoint_path = r'E:\newtacotron\tacotron\nepali\checkpoint'
#     test_output_dir = r'E:\newtacotron\tacotron\nepali\test'
#     os.makedirs(test_output_dir, exist_ok=True)

#     log(hparams_debug_string())

#     # Setup feeder
#     feeder = TestFeeder(tsv_path, hparams)
#     feeder.setup()

#     with tf.variable_scope('model') as scope:
#         model = create_model(args.model, hparams)
#         model.initialize(feeder_input=model.inputs,
#                         input_lengths=model.input_lengths,
#                         mel_targets=model.mel_targets,
#                         linear_targets=model.linear_targets)
#         model.add_loss()

#     with tf.Session() as sess:
#         try:
#             saver = tf.train.Saver()
#             saver.restore(sess, checkpoint_path)
#             log(f'✅ Restored model from checkpoint: {checkpoint_path}')

#             test_losses = []
#             total_loss = 0.0
#             step = args.checkpoint_step

#             for idx, input_seq, mel_target, linear_target, input_length, input_text in feeder.get_next():
#                 feed_dict = {
#                     model.inputs: [input_seq],
#                     model.input_lengths: [input_length],
#                     model.mel_targets: [mel_target],
#                     model.linear_targets: [linear_target]
#                 }

#                 linear_output, loss, alignment = sess.run(
#                     [model.linear_outputs, model.loss, model.alignments],
#                     feed_dict=feed_dict
#                 )

#                 linear_output = linear_output[0]
#                 alignment = alignment[0]
#                 total_loss += loss

#                 # Save audio
#                 waveform = audio.inv_spectrogram(linear_output.T)
#                 audio.save_wav(waveform, os.path.join(test_output_dir, f'eval-{idx}.wav'))

#                 # Save alignment
#                 plot.plot_alignment(
#                     alignment,
#                     os.path.join(test_output_dir, f'align-{idx}.png'),
#                     info=f'{args.model}, step={step}, loss={loss:.5f}'
#                 )

#                 # Log & record loss
#                 log(f'[Eval {idx}] loss={loss:.5f}, text="{input_text}"')
#                 test_losses.append((idx, loss, input_text))

#             # Save all losses
#             loss_csv = os.path.join(test_output_dir, 'losses.csv')
#             with open(loss_csv, 'w', encoding='utf-8') as f:
#                 f.write('index,loss,text\n')
#                 for idx, loss_val, text in test_losses:
#                     f.write(f'{idx},{loss_val:.5f},"{text}"\n')

#             avg_loss = total_loss / feeder.num_examples
#             log(f'\n✅ Average Loss over {feeder.num_examples} samples: {avg_loss:.5f}')

#         except Exception as e:
#             log(f'❌ Evaluation failed: {e}')
#             traceback.print_exc()


# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--base_dir', default='E:/newtacotron/tacotron/')
#     parser.add_argument('--input', default='training/test.tsv')  # TSV with text and audio path
#     parser.add_argument('--model', default='tacotron')
#     parser.add_argument('--checkpoint_step', type=int, required=True)
#     parser.add_argument('--name', help='Run name (defaults to model name)')
#     parser.add_argument('--hparams', default='')
#     parser.add_argument('--tf_log_level', type=int, default=1)
#     args = parser.parse_args()

#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
#     run_name = args.name or args.model
#     log_dir = r'E:\newtacotron\tacotron\nepali\log'
#     infolog.init(os.path.join(log_dir, 'eval.log'), run_name, None)
#     hparams.parse(args.hparams)

#     evaluate(log_dir, args)


# if __name__ == '_main_':
#     main()


import os
import tensorflow as tf
import numpy as np
import traceback
import argparse
from datetime import datetime
import pandas as pd

# Force CPU-only (suppress GPU errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import your Tacotron modules here (adjust paths if needed)
from hparams import hparams, hparams_debug_string
from models import create_model
from datasets.datafeeder import TestFeeder
from util import audio, plot
from text import sequence_to_text

import logging

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'eval.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.info

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def evaluate(log_dir, args, log):
    tsv_path = r'E:\newtacotron\tacotron\nepali\formatted_test.tsv'
    checkpoint_path = r'E:\newtacotron\tacotron\nepali\checkpoint'
    test_output_dir = r'E:\newtacotron\tacotron\nepali\test'
    os.makedirs(test_output_dir, exist_ok=True)

    log(f"Hyperparameters:\n{hparams_debug_string()}")

    feeder = TestFeeder(tsv_path, hparams)
    feeder.setup()

    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hparams)
        # We’ll feed inputs dynamically, so placeholders expected
        model.initialize(feeder_input=model.inputs,
                         input_lengths=model.input_lengths,
                         mel_targets=model.mel_targets,
                         linear_targets=model.linear_targets)
        model.add_loss()

    with tf.Session() as sess:
        try:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            log(f"✅ Restored checkpoint from: {checkpoint_path}")

            total_loss = 0.0
            test_losses = []
            step = args.checkpoint_step

            for idx, input_seq, mel_target, linear_target, input_length, input_text in feeder.get_next():
                feed_dict = {
                    model.inputs: [input_seq],
                    model.input_lengths: [input_length],
                    model.mel_targets: [mel_target],
                    model.linear_targets: [linear_target]
                }

                linear_output, loss, alignment = sess.run(
                    [model.linear_outputs, model.loss, model.alignments],
                    feed_dict=feed_dict
                )

                linear_output = linear_output[0]
                alignment = alignment[0]
                total_loss += loss

                # Save generated audio
                waveform = audio.inv_spectrogram(linear_output.T)
                audio.save_wav(waveform, os.path.join(test_output_dir, f'eval-{idx}.wav'))

                # Save alignment plot
                plot.plot_alignment(
                    alignment,
                    os.path.join(test_output_dir, f'align-{idx}.png'),
                    info=f'{args.model}, step={step}, loss={loss:.5f}'
                )

                log(f"[Eval {idx}] loss={loss:.5f}, text=\"{input_text}\"")
                test_losses.append((idx, loss, input_text))

            # Save losses CSV
            loss_csv_path = os.path.join(test_output_dir, 'losses.csv')
            with open(loss_csv_path, 'w', encoding='utf-8') as f:
                f.write("index,loss,text\n")
                for idx, loss_val, text in test_losses:
                    # Escape double quotes in text
                    safe_text = text.replace('"', '""')
                    f.write(f'{idx},{loss_val:.5f},"{safe_text}"\n')

            avg_loss = total_loss / feeder.num_examples
            log(f"\n✅ Average Loss over {feeder.num_examples} samples: {avg_loss:.5f}")

            print(f"Evaluation complete. Logs + outputs saved to:\n{test_output_dir}")

        except Exception as e:
            log(f"❌ Evaluation failed: {e}")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='E:/newtacotron/tacotron/')
    parser.add_argument('--input', default='training/test.tsv', help="TSV file with text and audio_path columns")
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--checkpoint_step', type=int, required=True, help="Checkpoint step to load")
    parser.add_argument('--name', help="Run name (used for log folder)")
    parser.add_argument('--hparams', default='')
    parser.add_argument('--tf_log_level', type=int, default=1)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)

    run_name = args.name or args.model

    log_dir = r'E:\newtacotron\tacotron\nepali\log'

    # Setup logging
    log = setup_logging(log_dir)

    # Parse hyperparams overrides
    hparams.parse(args.hparams)

    evaluate(log_dir, args, log)

if __name__ == '__main__':
    main()


# # python eval.py --checkpoint_step 3000 --input training/test.tsv
