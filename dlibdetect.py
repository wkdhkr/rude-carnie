from detect import ObjectDetector

from functools import reduce
import asyncio
import math
from math import sin, cos
import numpy as np
import dlib
import cv2
FACE_PAD = 5


class FaceDetectorDlib(ObjectDetector):
    def __init__(self, model_name, basename='frontal-face', tgtdir='.'):
        self.tgtdir = tgtdir
        self.basename = basename
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_name)

    def run_raw(
        self,
        original_img,
        deduped_results=[],
        is_crop_only=False,
        is_original=False,
        id="id",
        min_size=0
    ):
        img = original_img
        shape = img.shape
        max_size = 1024
        resize_ratio = 1
        resize_flag = False
        if max(shape[0], shape[1]) > max_size:
            resize_flag = True
            l = max(shape[0], shape[1])
            img = cv2.resize(
                original_img, (int(shape[1] * max_size / l), int(shape[0] * max_size / l)))
            resize_ratio = l / max_size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect fliped or sideways face
        original_rows, original_cols, _ = original_img.shape
        original_hypot = int(
            math.ceil(math.hypot(original_rows, original_cols)))
        rows, cols, _ = img.shape
        hypot = int(math.ceil(math.hypot(rows, cols)))

        frame = np.zeros((hypot, hypot), np.uint8)
        frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5),
              int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray
        color_frame = np.zeros((hypot, hypot, img.shape[2]), np.uint8)
        color_frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5),
                    int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = img
        original_color_frame = np.zeros(
            (original_hypot, original_hypot, original_img.shape[2]), np.uint8)
        original_color_frame[
            int((original_hypot - original_rows) * 0.5):int((original_hypot + original_rows) * 0.5), int((original_hypot - original_cols) * 0.5):int((original_hypot + original_cols) * 0.5)
        ] = original_img

        def translate(coordinate, angle):
            x, y = coordinate
            rad = math.radians(angle)
            return {
                'x': (cos(rad) * x + sin(rad) * y - hypot * 0.5 * cos(rad) - hypot * 0.5 * sin(rad) + hypot * 0.5 - (hypot - cols) * 0.5) / float(cols) * 100.0,
                'y': (- sin(rad) * x + cos(rad) * y + hypot * 0.5 * sin(rad) - hypot * 0.5 * cos(rad) + hypot * 0.5 - (hypot - rows) * 0.5) / float(rows) * 100.0,
            }
        results = []
        # if set 1, 40x40 face detection chance available
        upsample_num_times = 0
        for angle in range(-90, 91, 30):
            M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), angle, 1.0)
            rotated_img = cv2.warpAffine(frame, M, (hypot, hypot))
            faces, scores, types = self.detector.run(
                rotated_img, upsample_num_times, 0.15)
            for (i, rect) in enumerate(faces):
                x = rect.left()
                y = rect.top()
                w = rect.right() - x
                h = rect.bottom() - y

                resize_ratio_y = y / rotated_img.shape[0]
                resize_ratio_h = h / rotated_img.shape[0]
                resize_ratio_x = x / rotated_img.shape[1]
                resize_ratio_w = w / rotated_img.shape[1]

                results.append({
                    'dlib_score': scores[i],
                    'direct': 'frontal',
                    'angle': angle,
                    'center': translate([x + w * 0.5, y + h * 0.5], -angle),
                    'w': float(w) / float(cols) * 100.0,
                    'h': float(h) / float(rows) * 100.0,
                    'ax': x,
                    'ay': y,
                    'aw': w,
                    'ah': h,
                    'resize_ratio_y': resize_ratio_y,
                    'resize_ratio_h': resize_ratio_h,
                    'resize_ratio_x': resize_ratio_x,
                    'resize_ratio_w': resize_ratio_w
                })

        for result in results:
            x, y = result['center']['x'], result['center']['y']
            exists = False
            for i in range(len(deduped_results)):
                face = deduped_results[i]
                if (
                    face['center']['x'] - face['w'] * 0.5 < x < face['center']['x'] + face['w'] * 0.5 and
                    face['center']['y'] - face['h'] *
                        0.5 < y < face['center']['y'] + face['h'] * 0.5
                ):
                    exists = True
                    # if result.get("dlib_score", 0) < face.get("dlib_score", 0):
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
        for deduped_result in deduped_results:
            i = i + 1
            full_id = id + "_" + str(i)
            name_prefix = self.basename
            M = cv2.getRotationMatrix2D(
                (hypot * 0.5, hypot * 0.5), deduped_result['angle'], 1.0)
            rotated_img = cv2.warpAffine(color_frame, M, (hypot, hypot))
            file_path = self.sub_image(
                '%s/%s-%s.jpg' % (self.tgtdir, name_prefix, full_id),
                rotated_img,
                deduped_result['ax'],
                deduped_result['ay'],
                deduped_result['aw'],
                deduped_result['ah']
            )
            original_file_path = None
            w = deduped_result['aw']
            h = deduped_result['ah']

            # original
            if resize_flag:
                M = cv2.getRotationMatrix2D(
                    (original_hypot * 0.5, original_hypot * 0.5), deduped_result['angle'], 1.0)
                rotated_img = cv2.warpAffine(
                    original_color_frame, M, (original_hypot, original_hypot))
                w = int(
                    round(rotated_img.shape[1] * deduped_result["resize_ratio_w"]))
                h = int(
                    round(rotated_img.shape[0] * deduped_result["resize_ratio_h"]))

            if is_original:
                original_file_path = self.sub_image(
                    '%s/%s-%s.jpg' % (self.tgtdir,
                                      "original_" + name_prefix, full_id),
                    rotated_img,
                    int(round(rotated_img.shape[1] *
                              deduped_result["resize_ratio_x"])),
                    int(round(rotated_img.shape[0] *
                              deduped_result["resize_ratio_y"])),
                    w,
                    h
                )

            # skip small size image
            if max(w, h) < min_size:
                continue

            image = {
                'id': full_id,
                'file_path': file_path,
                'original_file_path': original_file_path,
                'opencv_score': deduped_result.get("opencv_score", None),
                'dlib_score': deduped_result.get("dlib_score", None),
                'direct': deduped_result["direct"],
                'angle': deduped_result["angle"],
                'width': w,
                'height': h
            }
            images.append(image)
        return images

    def run(self, image_file):
        print(image_file)
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        images = []
        bb = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bb.append((x, y, w, h))
            images.append(self.sub_image('%s/%s-%d.jpg' %
                                         (self.tgtdir, self.basename, i + 1), img, x, y, w, h))

        print('%d faces detected' % len(images))

        for (x, y, w, h) in bb:
            self.draw_rect(img, x, y, w, h)
            # Fix in case nothing found in the image
        outfile = '%s/%s.jpg' % (self.tgtdir, self.basename)
        cv2.imwrite(outfile, img)
        return images, outfile

    def sub_image(self, name, img, x, y, w, h):
        # remove black margin
        upper_cut = [min(img.shape[0], y + h + FACE_PAD),
                     min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[round(lower_cut[0]):round(
            upper_cut[0]), round(lower_cut[1]):round(upper_cut[1])]
        final_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = roi_color[y:y+h, x:x+w]
        cv2.imwrite(name, crop)
        return name

    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD),
                     min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]),
                      (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)
