from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv

from flask import Flask, request, Response, jsonify

app = Flask(__name__)

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')


tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'inception',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS

def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None

def classify_many_single_crop_server(sess, label_list, softmax_output, coder, images, image_files, writer=None):
    results = []
    try:
        num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        for j in range(int(num_batches)):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))

            batch_image_files = image_files[start_offset:end_offset]
            image_batch = make_multi_image_batch(batch_image_files, coder, len(batch_image_files))
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                f = batch_image_files[i]
                result = (f, best_choice[0], '%.2f' % best_choice[1])
                if writer is not None:
                    writer.writerow(result)
                results.append(result)
    except Exception as e:
        raise e
    return results

def main(argv=None):  # pylint: disable=unused-argument

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):

            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()

            requested_step = FLAGS.requested_step if FLAGS.requested_step else None

            checkpoint_path = '%s' % (FLAGS.model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)

            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)

            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            @app.route('/facecheck', methods=['POST'])
            def image():
                i = request.files['image']
                data = np.fromstring(i.stream.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                face_detect_dlib = face_detection_model("dlib", "shape_predictor_68_face_landmarks.dat")
                face_detect_cv = face_detection_model("", "haarcascade_profileface.xml")

                results = face_detect_cv.run_raw(img, [], True)
                image_files = face_detect_dlib.run_raw(img, results)

                results = classify_many_single_crop_server(sess, label_list, softmax_output, coder, images, image_files)
                return jsonify(results)

            app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    tf.app.run()
