# import argparse
# import os
# import re
# from hparams import hparams, hparams_debug_string
# from synthesizer import Synthesizer


# sentences = [
#   'नेपाल एक सुन्दर देश हो।',
#   'म बिहान छिटो उठ्छु।',
#   'पढ्नाले मानिसको जीवन परिवर्तन हुन्छ।',
#   'आज मौसम धेरै राम्रो छ।',
#   'तिमीलाई कस्तो लाग्यो?',
#   'शिक्षा नै जीवनको मूल आधार हो।',
#   'उसले मलाई पुस्तक दियो।',
#   'के तिमी मलाई सुनिरहेछौ?',
# ]



# def get_output_base_path(checkpoint_path):
#   base_dir = os.path.dirname(checkpoint_path)
#   m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
#   name = 'eval-%d' % int(m.group(1)) if m else 'eval'
#   return os.path.join(base_dir, name)


# def run_eval(args):
#   print(hparams_debug_string())
#   synth = Synthesizer()
#   synth.load(args.checkpoint)
#   base_path = get_output_base_path(args.checkpoint)
#   for i, text in enumerate(sentences):
#     path = '%s-%d.wav' % (base_path, i)
#     print('Synthesizing: %s' % path)
#     with open(path, 'wb') as f:
#       f.write(synth.synthesize(text))


# def main():
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
#   parser.add_argument('--hparams', default='',
#     help='Hyperparameter overrides as a comma-separated list of name=value pairs')
#   args = parser.parse_args()
#   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#   hparams.parse(args.hparams)
#   run_eval(args)


# if __name__ == '__main__':
#   main()

import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from scipy.io import wavfile
from util import audio


# Nepali sentences and corresponding ground truth mel spectrograms (aligned)
# Make sure the mel spectrogram ground truths are stored as .npy files
sentences = [
  'दीपा धामीको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'डिग्रा देवीको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'टेकबहादुर ऐरको जन्म सुदूरपश्चिम नेपालको डडेलधुरा जिल्लामा भएको हो',
'सुमन शेखरमानन्धर नेपालको कृषि अर्थविद् तथा गायक हुन्',
'आधिकारिक रूपमा पहिलो नेपाली चलचित्र नायक शिव शंकर मानन्धरलाई मानिन्छ',
'नेपाली सङ्गीतमा पुर्‍याएको योगदानका लागि सङ्गीत उद्योगले सम्झिरहनु पर्ने एउटा नामकोइलीदेवी हो',
'रेडियो नेपालमा कार्यरत रहँदा मास्टर रत्नदास प्रकाश निकै लोकप्रिय थिए',
'प्रकाश श्रेष्ठ आधुनिक नेपाली गायक हुन्',
'सेते नेपाली श्रोतामाझ भिजेका गायक हुन्',
'शान्ति धामी एक लोकप्रिय लोक गायक हुन्',
'गोपिसिंह नेपालीको जन्म सुदूरपश्चिम नेपालको अछाम जिल्लामा भएको हो',
'बेनीमाधव भट्टको जन्म सुदूरपश्चिम नेपालको डोटी जिल्लामा भएको हो',
'जगदिश समाल नेपाली आधुनिक तथा गजल गायक हुन',
'गोपाल बिस्टको जन्म सुदूरपश्चिम नेपालको बैतडी जिल्लामा भएको हो',
'बिद्या तिमिल्सिनाको जन्म सुदूरपश्चिम नेपालको बाजुरा जिल्लामा भएको हो',
'बिमला बुढाको जन्म सुदूरपश्चिम नेपालको बाजुरा जिल्लामा भएको हो',
'कुमार बस्नेत नेपालका लोकप्रिय लोक गायक हुन्',
'चाँदनी मल्लको जन्म सुदूरपश्चिम नेपालको दार्चुला जिल्लामा भएको हो',
'सबिन राई नेपाली पप गायक तथा संगीतकार हुन्',
'गणेशबहादुर सिंहको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'सबिता भट्टले सुदूरपश्चिमका गीतहरू गाएकी छन्',
'नारायण बिस्टको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'दुर्गादेवी जोशीको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'गोरखबहादुर थापाको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'पाण्डव सुनुवारले नेपाली साङ्गीतिक दुनियाँमा हत्तपत्त नमेटिने पहिचान मात्र बनाएका थिएनन्',
'झंकर साउदको जन्म सुदूरपश्चिम नेपालको अछाम जिल्लामा भएको हो',
'मोहन बजुलिया देउडा गीतका प्रखर गायक हुन्',
'गणेश बमको जन्म सुदूरपश्चिम नेपालको दार्चुला जिल्लामा भएको हो',
'यस क्षेत्रमा राजेन्द्रबहादुर थापाको योगदान अद्वितीय छ',
'मनोरथ खडायतको जन्म सुदूरपश्चिम नेपालको डडेलधुरा जिल्लामा भएको हो',
'डिल्लीराज फुलाराको जन्म सुदूरपश्चिम नेपालको डोटी जिल्लामा भएको हो',
'निमराज ओझाको जन्म सुदूरपश्चिम नेपालको डोटी जिल्लामा भएको हो',
'जयभक्त जोशीको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'शीर्षक राखेका रहेछन् सन्तोष शर्माले',
'नन्दसिंह धामीको जन्म सुदूरपश्चिम नेपालको डोटी जिल्लामा भएको हो',
'नन्दकृष्ण जोशी देउडा गायक हुन्',
'राजु नेपालीका गीति एल्बमहरू निकै लोकप्रिय छन्',
'नृप स्वारको जन्म सुदूरपश्चिम नेपालको अछाम जिल्लामा भएको हो',
'देउडा गायक मान बहादुर केसीको कठै दैव तलाई एल्बम बजारमा आउँदै छ',
'पदमबहादुर रोकायाको जन्म सुदूरपश्चिम नेपालको बझाङ जिल्लामा भएको हो',
'गायक रामकृष्ण ढकाल यिनलाई आफ्नो गडफादर मान्दछन्',
'डबल बोहराको जन्म सुदूरपश्चिम नेपालको डोटी जिल्लामा भएको हो',
'मित्रसेन थापा मगर जन्मँदा उनका बाबु फर्स्ट गोरखा राइफल्समा कार्यरत थिए',
'नवीन भट्टराई नेपाली पप गायनमा ख्याति कमाएका गायक हुन्',
'रत्ना जोशीको नाम साङ्गीतिक क्षेत्रमा निकै लोकप्रिय छ',
'शैलेन्द्र मोहन प्रधान बाबु सङ्गीत संयोजक हुन् र उहाँको जन्म हेटौँडाको स्कुल रोडमा भएको हो',
'यस चलचित्रका निर्देशक प्रदीप श्रेष्ठ हुन्',
'सुवि शाहद्वारा सङ्कलित धादिङको पञ्चैबाजाको धुन रेडियो नेपालबाट विजयादशमीभर बजाइन्थ्यो',
'मनोहरराम दमाईको जन्म सुदूरपश्चिम नेपालको बैतडी जिल्लामा भएको हो',
'तिला जुम्ला जिल्लाको पूर्वी भागबाट पश्चिम हुँदै कालिकोट जिल्लामा प्रवेश गरी सेरीगाड नजिकै कर्णालीमा मिसिन्छ',
]

# Ground truth mel spectrogram paths (aligned with sentences)
ground_truth_mel_paths = [
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00001.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00002.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00003.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00004.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00005.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00006.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00007.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00008.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00009.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00010.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00011.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00012.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00013.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00014.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00015.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00016.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00017.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00018.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00019.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00020.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00021.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00022.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00023.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00024.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00025.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00026.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00027.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00028.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00029.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00030.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00031.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00032.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00033.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00034.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00035.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00036.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00037.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00038.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00039.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00040.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00041.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00042.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00043.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00044.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00045.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00046.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00047.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00048.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00049.npy',
'E:/newtacotron/tacotron/our_code/mel_output/nepali-mel-00050.npy',
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)

  base_path = get_output_base_path(args.checkpoint)
  prediction_dir = os.path.join(base_path, "predictions")
  os.makedirs(prediction_dir, exist_ok=True)

  loss_log_path = r"E:\newtacotron\tacotron\our_code\my_logs\loss.log"
  with open(loss_log_path, "w", encoding="utf-8") as log_file:
    for i, (text, gt_path) in enumerate(zip(sentences, ground_truth_mel_paths)):
      print(f"Processing {i}: {text}")

      # Predict mel spectrogram
      mel_pred = synth.synthesize_mel(text)
      pred_path = os.path.join(prediction_dir, f"mel_{i}_pred.npy")
      np.save(pred_path, mel_pred)

      # Load ground truth
      if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        continue

      mel_gt = np.load(gt_path)

      # Crop to same length
      min_len = min(mel_pred.shape[0], mel_gt.shape[0])
      mel_pred = mel_pred[:min_len]
      mel_gt = mel_gt[:min_len]

      # Compute L1 loss
      loss = np.mean(np.abs(mel_pred - mel_gt))

      log_file.write(f"Sample {i} | Loss: {loss:.6f} | Text: {text}\n")
      print(f" -> Loss: {loss:.6f}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='', help='Override hyperparams (comma-separated)')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
