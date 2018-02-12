import math
from math import sin, cos
import numpy as np
import cv2
FACE_PAD = 5

class ObjectDetector(object):
    def __init__(self):
        pass

    def run(self, image_file):
        pass

# OpenCV's cascade object detector
class ObjectDetectorCascadeOpenCV(ObjectDetector):
    def __init__(self, model_name, basename='frontal-face', tgtdir='.', min_height_dec=20, min_width_dec=20,
                 min_height_thresh=50, min_width_thresh=50):
        self.min_height_dec = min_height_dec
        self.min_width_dec = min_width_dec
        self.min_height_thresh = min_height_thresh
        self.min_width_thresh = min_width_thresh
        self.tgtdir = tgtdir
        self.basename = basename
        self.face_cascade = cv2.CascadeClassifier(model_name)

    def run_raw(self, img, deduped_results=[], is_crop_only=False):
        shape = img.shape
        # max_size = 20000
        # if max(shape[0], shape[1]) > max_size:
        #     l = max(shape[0], shape[1])
        #     img = cv2.resize(img, (int(shape[1] * max_size / l), int(shape[0] * max_size / l)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect fliped or sideways face
        rows, cols, _ = img.shape
        hypot = int(math.ceil(math.hypot(rows, cols)))
        frame = np.zeros((hypot, hypot), np.uint8)
        frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray
        def translate(coordinate, angle):
            x, y = coordinate
            rad = math.radians(angle)
            return {
                'x': (  cos(rad) * x + sin(rad) * y - hypot * 0.5 * cos(rad) - hypot * 0.5 * sin(rad) + hypot * 0.5 - (hypot - cols) * 0.5) / float(cols) * 100.0,
                'y': (- sin(rad) * x + cos(rad) * y + hypot * 0.5 * sin(rad) - hypot * 0.5 * cos(rad) + hypot * 0.5 - (hypot - rows) * 0.5) / float(rows) * 100.0,
            }
        results = []
        for angle in list(range(-80, 81, 40)):
            M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), angle, 1.0)
            rotated_img = cv2.warpAffine(frame, M, (hypot, hypot))
            min_h = int(max(rotated_img.shape[0] / self.min_height_dec, self.min_height_thresh))
            min_w = int(max(rotated_img.shape[1] / self.min_width_dec, self.min_width_thresh))
            faces = self.face_cascade.detectMultiScale(
                rotated_img, 1.3, minNeighbors=6, minSize=(min_h, min_w)
            )
            print (angle, len(faces))
            for i, (x, y, w, h) in enumerate(faces):
                # TODO: size limitation
                results.append({
                    'angle': angle,
                    'center': translate([x + w * 0.5, y + h * 0.5], -angle),
                    'w': float(w) / float(cols) * 100.0,
                    'h': float(h) / float(rows) * 100.0,
                    'ax': x,
                    'ay': y,
                    'aw': w,
                    'ah': h
                })

        for result in results:
            x, y = result['center']['x'], result['center']['y']
            exists = False
            for i in range(len(deduped_results)):
                face = deduped_results[i]
                if (
                    face['center']['x'] - face['w'] * 0.5 < x < face['center']['x'] + face['w'] * 0.5 and
                    face['center']['y'] - face['h'] * 0.5 < y < face['center']['y'] + face['h'] * 0.5
                ):
                    print(face)
                    exists = True
                    if abs(result['angle']) <= abs(face['angle']):
                        deduped_results[i] = result
                        break
            if not exists:
                deduped_results.append(result)

        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(deduped_results)
        if is_crop_only:
            return deduped_results

        images = []
        i = 0
        color_frame = np.zeros((hypot, hypot, img.shape[2]), np.uint8)
        color_frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = img
        for deduped_result in deduped_results:
            print(deduped_result["angle"])
            M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deduped_result['angle'], 1.0)
            rotated_img = cv2.warpAffine(color_frame, M, (hypot, hypot))
            i = i + 1
            images.append(
                self.sub_image(
                    '%s/%s-%d.jpg' % (self.tgtdir, self.basename, i),
                    rotated_img,
                    deduped_result['ax'],
                    deduped_result['ay'],
                    deduped_result['aw'],
                    deduped_result['ah']
                )
            )
        return images

    def run(self, image_file):
        print(image_file)
        img = cv2.imread(image_file)
        min_h = int(max(img.shape[0] / self.min_height_dec, self.min_height_thresh))
        min_w = int(max(img.shape[1] / self.min_width_dec, self.min_width_thresh))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(min_h, min_w))

        images = []
        for i, (x, y, w, h) in enumerate(faces):
            images.append(self.sub_image('%s/%s-%d.jpg' % (self.tgtdir, self.basename, i + 1), img, x, y, w, h))

        print('%d faces detected' % len(images))

        for (x, y, w, h) in faces:
            self.draw_rect(img, x, y, w, h)
            # Fix in case nothing found in the image
        outfile = '%s/%s.jpg' % (self.tgtdir, self.basename)
        cv2.imwrite(outfile, img)
        return images, outfile

    def sub_image(self, name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        cv2.imwrite(name, roi_color)
        return name

    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)

