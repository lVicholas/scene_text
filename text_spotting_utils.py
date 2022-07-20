import numpy as np
from string import ascii_letters, digits, punctuation
import cv2
import unicodedata

def get_detection_model(model_path, detect_size, in_BGR):

    det_net = cv2.dnn.readNet(model_path)
    det_model = cv2.dnn_TextDetectionModel_DB(det_net)

    input_params = {
        'scale': 1.0 / 255.0,
        'size': detect_size,
        'mean': (123.68, 116.78, 103.94),
        'swapRB': False
    }
    det_model.setInputParams(**input_params)

    det_model.setBinaryThreshold(.3)
    det_model.setPolygonThreshold(.5)
    det_model.setUnclipRatio(2.0)
    det_model.setMaxCandidates(200)

    return det_model

def get_recognition_model(model_path, in_BGR):

    rec_net = cv2.dnn.readNet(model_path)
    rec_model = cv2.dnn_TextRecognitionModel(rec_net)

    voc = digits + ascii_letters + punctuation

    rec_params = {
        'scale': 1.0 / 255.0,
        'size': (100, 32),
        'mean': (127.5, 127.5, 127.5),
        'swapRB': not in_BGR
    }

    rec_model.setDecodeType('CTC-greedy')
    rec_model.setVocabulary(voc)
    rec_model.setInputParams(**rec_params)

    return rec_model

def fourPointsTransform(frame, vertices):

    # Helper for cropping detection ROI for recognition

    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32"
    )
    
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

def spatially_order_recognitions(
    detections, recognitions, img_shape, line_height_factor=.1
):
    '''
    Order recognitions from top-to-bottom, left-to-right, based on the 
    average coordinates of the detection polygon
    
    If recognitions are sufficiently close in height, they are considered
    to be on the same text line, and so are ordered according only
    to horizontal position
    '''

    if len(detections) == 0 or len(recognitions) == 0:
        return [], []

    assert all(len(det.shape) == 2 and det.shape[1] == 2 for det in detections)

    Y, X = img_shape[:2]
    line_height_distance = round(Y * line_height_factor)
    line_markers = np.arange(Y, step=line_height_distance)

    mean_points = [poly.mean(axis=0) for poly in detections]

    line_numbers = np.digitize([y for _, y in mean_points], line_markers)
    mean_points = [(x, y) for (x, _), y in zip(mean_points, line_numbers)]
    position_scores = [x + y*max(X, Y) for x, y in mean_points]

    t2b_l2r_argsort = np.argsort(position_scores)
    t2b_l2r_mean_points = list(np.array(mean_points)[t2b_l2r_argsort])
    t2b_l2r_recognitions = list(np.array(recognitions)[t2b_l2r_argsort])

    return t2b_l2r_mean_points, t2b_l2r_recognitions

def get_recognition_results_lines(t2b_l2r_mean_points, t2b_l2r_recognitions):

    # Return recognitions spatially ordered, separated by text line

    if len(t2b_l2r_mean_points) == 0 or len(t2b_l2r_recognitions) == 0:
        return ''

    lines = []
    current_line = ''
    current_line_idx = min(t2b_l2r_mean_points, key=lambda p: p[1])[1]
    for (_, y), rec in zip(t2b_l2r_mean_points, t2b_l2r_recognitions):
        
        if current_line_idx < y:
            lines.append(current_line.lower())
            current_line = ''
            current_line_idx = y

        if len(current_line) == 0:
            current_line += rec
        else:
            current_line += ' ' + rec
    lines.append(current_line.lower())

    return lines 

def get_processed_recognition_results(img, rec_model, detections):

    recognitions = [
        rec_model.recognize(fourPointsTransform(img, det)).lower()
        for det in detections
    ]
    
    sorted_mean_points, sorted_recs = spatially_order_recognitions(
        detections, recognitions, img.shape
    )
    result_lines = get_recognition_results_lines(
        sorted_mean_points, sorted_recs
    )

    return result_lines

def string_ascii_process(string):     

    # Convert accented characters to ascii version
    text = unicodedata \
            .normalize('NFD', string) \
            .encode('ascii', 'ignore') \
            .decode("utf-8")

    # Remove all non-ascii characters
    voc = ascii_letters + digits + punctuation + ' '
    text = ''.join([c for c in text if c in voc])

    return text

def detection_recognition_translation_pipeline(
    img, 
    det_model, rec_model, translator, 
    translation_fail_str
):

    detections, conf = det_model.detect(img)

    recognition_lines = []
    translation_str = ''
    draw_str = ''

    if rec_model is not None:

        recognition_lines = get_processed_recognition_results(
            img, rec_model, detections
        )
        draw_str = '\n'.join(recognition_lines)

        if translator is not None:
            try:
                en_str = ' '.join(recognition_lines)
                translation_str = translator.translate(en_str)
            except RuntimeError:
                translation_str = translation_fail_str

            if translation_str != translation_fail_str:
                draw_str = string_ascii_process(translation_str)

    return detections, recognition_lines, translation_str, draw_str

def put_lines(img, lines, scale=1, thickness=3):

    # Draw recognitions on image

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)

    if isinstance(lines, str):
        lines = lines.split('\n')

    line_height = img.shape[0] // (len(lines) + 1)

    for l, line in enumerate(lines):
        org = round(img.shape[1] * .05), (l + 1) * line_height
        cv2.putText(img, line, org, font, 1, color, 3)

def string_to_lines(string, words_per_line_plus_1=4):

    lines, line = [], []
    for w, word in enumerate(string.split(' ')):
        if (w+1) % words_per_line_plus_1 == 0:
            lines.append(' '.join(line))
            line = []
        else:
            line.append(word.replace('\n', ' '))
    lines.append(' '.join(line))

    return lines

def outputs(
    img, 
    det, rec_lines, trans_lines, draw_lines,
    color, line_thickness, display_size, 
    draw_recognitions, print_to_console
):

    img = cv2.polylines(img, det, True, color, line_thickness)
    img = cv2.resize(img, dsize=display_size)

    if draw_recognitions:
        if isinstance(draw_lines, list):
            put_lines(img, draw_lines)
        else:
            lines = string_to_lines(draw_lines)
            put_lines(img, lines)

    if print_to_console:
        if len(rec_lines) != 0:
            print(f'\nrecognition results:\n{rec_lines}')
        if len(trans_lines) != 0:
            print(f'\ntranslation results:\n{trans_lines}')

    return img