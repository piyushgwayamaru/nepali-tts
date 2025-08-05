import os
import argparse
import falcon
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
import scipy.io.wavfile as wavfile
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



# -----------------------
# CORS Middleware
# -----------------------
class CORSMiddleware:
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        resp.set_header('Access-Control-Allow-Headers', '*')
        # Handle CORS preflight requests
        if req.method == 'OPTIONS':
            resp.status = falcon.HTTP_200
            return

# -----------------------
# Falcon Resource
# -----------------------
class SynthesisResource:
    def on_get(self, req, res):
        text = req.params.get('text', '')
        if not text or not re.match(r'^[\u0900-\u097F\s.,!?।\d\-–—\n\r\t]*$', text):
            raise falcon.HTTPBadRequest('Invalid Input', 'Only Nepali (Devanagari) text is allowed.')
        
        res.data = synthesizer.synthesize(text)
        res.content_type = 'audio/wav'


# -----------------------
# Initialize
# -----------------------
synthesizer = Synthesizer()
api = falcon.API(middleware=[CORSMiddleware()])
api.add_route('/synthesize', SynthesisResource())

# -----------------------
# Run Server
# -----------------------
if __name__ == '__main__':
    from wsgiref import simple_server
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    print(hparams_debug_string())

    synthesizer.load(args.checkpoint)
    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
    synthesizer.load(os.environ['CHECKPOINT'])

