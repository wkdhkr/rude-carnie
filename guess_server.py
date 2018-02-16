from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
import base64
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
import re

from flask import Flask, request, Response, jsonify

app = Flask(__name__)

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_boolean('debug', False,
                           'debug')

tf.app.flags.DEFINE_integer('port', '5001',
                           'flask http server port number')

tf.app.flags.DEFINE_string('work_dir', '.',
                           'Working directory')

tf.app.flags.DEFINE_boolean('no_sweep', False,
                           'Sweep working directory')

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

def classify_many_single_crop_server(sess, label_list, softmax_output, coder, images, image_items, writer=None):
    results = []
    try:
        num_batches = math.ceil(len(image_items) / MAX_BATCH_SZ)
        for j in range(int(num_batches)):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_items))

            batch_image_items = image_items[start_offset:end_offset]
            image_batch = make_multi_image_batch([x["file_path"] for x in batch_image_items], coder, len(batch_image_items))
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                f = batch_image_items[i]
                result = (f, best_choice[0], '%.2f' % best_choice[1])
                if writer is not None:
                    writer.writerow(result)
                results.append(result)
    except Exception as e:
        raise e
    return results

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

def main(argv=None):  # pylint: disable=unused-argument
    tgtdir = FLAGS.work_dir

    port_number = FLAGS.port
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

            @app.route('/face/predict', methods=['POST'])
            def predict():
                image_items = json.loads(request.form.get("data_set"))
                required_class = request.form.getlist("class")
                no_data_flag = request.form.get("no_data")

                request_id = str(uuid.uuid4())

                image_files = []
                i = 0
                for image_item in image_items:
                    i = i + 1
                    id = image_item.get("id", request_id + "_" + str(i))
                    image_item["id"] = id
                    if "data" in image_item:
                        new_file_path = tgtdir + "/" + ("frontal-face-%s.jpg" % id)
                        write_base64_jpeg_file(new_file_path, image_item["data"])
                        image_item["file_path"] = new_file_path

                results = classify_many_single_crop_server(
                    sess,
                    label_list,
                    softmax_output,
                    coder,
                    images,
                    image_items
                )
                final_results = []
                for result in results:
                    append_flag = True
                    if (len(required_class) and result[1] not in required_class):
                        append_flag = False
                    if append_flag:
                        image_item = result[0]
                        prev_prediction = image_item.get("prediction", None)
                        prev_score = image_item.get("score", None)
                        image_item["prediction"] = result[1]
                        image_item["score"] = result[2]
                        if not prev_prediction is None:
                            image_item["prev_prediction"] = prev_prediction
                        if not prev_score is None:
                            image_item["prev_score"] = prev_score
                        if not no_data_flag:
                            with open(result[0]["file_path"], 'rb') as f:
                                image_item["data"] = base64.b64encode(f.read()).decode("utf-8")
                        else:
                            del image_item["data"]
                        if not FLAGS.no_sweep:
                            purge(tgtdir, request_id)
                        final_results.append(image_item)
                    if not FLAGS.no_sweep:
                        os.remove(result[0]["file_path"])
                return jsonify(final_results)

            @app.route('/face/detect', methods=['POST'])
            def detect():
                i = request.files['image']
                required_class = request.form.getlist("class")
                no_data_flag = bool(request.form.get("no_data"))
                min_size = request.form.get("min_size")
                is_original = bool(request.form.get("original"))

                request_id = str(uuid.uuid4())

                data = np.fromstring(i.stream.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                face_detect_dlib = face_detection_model("dlib", "shape_predictor_68_face_landmarks.dat", tgtdir)
                face_detect_cv = face_detection_model("", "haarcascade_profileface.xml", tgtdir)

                results = face_detect_cv.run_profile_raw(img, [], True, is_original, request_id, min_size)
                image_items = face_detect_dlib.run_raw(
                    img,
                    results,
                    False,
                    is_original,
                    request_id,
                    int(min_size or 0)
                )

                results = classify_many_single_crop_server(
                    sess, label_list, softmax_output, coder, images, image_items
                )
                final_results = []
                for result in results:
                    append_flag = True
                    if (len(required_class) and result[1] not in required_class):
                        append_flag = False
                    if append_flag:
                        item = result[0]
                        item["prediction"] = result[1]
                        item["score"] = result[2]
                        if not no_data_flag:
                            with open(result[0]["file_path"], 'rb') as f:
                                item["data"] = base64.b64encode(f.read()).decode("utf-8")
                        final_results.append(item)
                if not FLAGS.no_sweep:
                    purge(tgtdir, request_id)
                return jsonify(final_results)

            app.run(debug=FLAGS.debug, host='0.0.0.0', port=port_number)


if __name__ == '__main__':
    tf.app.run()
