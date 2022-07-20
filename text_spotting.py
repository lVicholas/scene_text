import argparse
import numpy as np
import time
import cv2
import text_spotting_utils as utils
from translate import Translator

def main():

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        'detection_model_path',
        help='path to detection model'
    )
    argparser.add_argument(
        '-recognition_model_path',
        help=\
        'path to model for text recognition / OCR, no recognition by default',
        default=None
    )
    argparser.add_argument(
        '-translation_language',
        help='language to translate english text detections to, default None',
        default=None
    )
    argparser.add_argument(
        '-img_path',
        help='path of image to detect text of',
        default='sample_data\p1.jpg'
    )
    argparser.add_argument(
        '-video',
        help='path to video or 0 for webcam',
        default=None
    )
    argparser.add_argument(
        '-video_output',
        help='path to save video results to, only availabe for video input',
        default=None
    )
    argparser.add_argument(
        '-detection_size',
        help='resize image for detection, default 320x320',
        default=(640, 640),
        nargs=2,
        type=int
    )
    argparser.add_argument(
        '-display_size',
        help='size to display image, default 480x480',
        default=(640, 640),
        nargs=2,
        type=int
    )
    argparser.add_argument(
        '-line_thickness',
        help='thickness of detection polygon lines, default 2',
        type=int,
        default=4
    )
    argparser.add_argument(
        '-fps',
        help='fps to detect in video, defualt 20, ignored without video',
        default=20
    )
    argparser.add_argument(
        '--bgr',
        help='whether image or video is in BGR format',
        action='store_const',
        const=True,
        default=False
    )
    argparser.add_argument(
        '--draw_recognitions',
        help='whether to draw text recognitions on image, defualt false',
        action='store_const',
        const=True,
        default=False
    )

    t0 = time.time()

    args = argparser.parse_args()

    det_model_path = args.detection_model_path
    rec_model_path = args.recognition_model_path
    translation_lang = args.translation_language

    img_path = args.img_path
    video = args.video
    video_output = args.video_output
    fps = args.fps

    detect_size = args.detection_size # Resize image for detection pipeline
    display_size = args.display_size # Display size of results image
    in_BGR = args.bgr # Whether input is already in BGR format
    line_thickness = args.line_thickness # Thickness of detection boxes
    draw_recognitions = args.draw_recognitions # Should draw recognized text ?

    GREEN = (0, 255, 0)
    TRANSLATION_FAIL_STR = 'translation failed :('

    img_vid_aem = 'must provide image or video, none provided'
    assert img_path is not None or video is not None, img_vid_aem

    det_model = utils.get_detection_model(det_model_path, detect_size, in_BGR)

    if rec_model_path is not None:
        rec_model = utils.get_recognition_model(rec_model_path, in_BGR)
    else:
        rec_model = None
    
    if translation_lang is not None:
        translator = Translator(to_lang=translation_lang)
    else:
        translator = None

    if video is None:

        img = cv2.imread(img_path)

        pipeline_results = utils.detection_recognition_translation_pipeline(
            img, det_model, rec_model, translator,
            TRANSLATION_FAIL_STR
        )

        pipeline_time = round(time.time() - t0, 3)
        print('\ntotal pipeline time:', pipeline_time, 'seconds')

        img = utils.outputs(
            img,
            *pipeline_results, 
            GREEN, line_thickness, display_size,
            draw_recognitions, True
        )
        
        cv2.imshow(f'detections of {img_path}', img)     
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif video == '0':

        cap = cv2.VideoCapture(0)

        while True:

            _, img = cap.read()

            pipeline_results = utils.detection_recognition_translation_pipeline(
                img, det_model, rec_model, translator,
                TRANSLATION_FAIL_STR
            )

            img = utils.outputs(
                img,
                *pipeline_results, 
                GREEN, line_thickness, display_size,
                draw_recognitions, False
            )
            
            # Display output
            cv2.imshow('webcam_text_detection', img)

            # Press q to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:

        assert video_output is not None, 'must provide path for video output'

        vid = cv2.VideoCapture(video)
        empty_frames = 0

        if video_output is not None:
            fps = int(fps)
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_output, codec, fps, display_size)
        else:
            out = None

        while True:

            _, img = vid.read()

            if img is None or 0 in img.shape:
                empty_frames += 1
                time.sleep(.1)
                if empty_frames == 5:
                    break
                continue
            else:
                empty_frames = 0

            pipeline_results = utils.detection_recognition_translation_pipeline(
                img, det_model, rec_model, translator,
                TRANSLATION_FAIL_STR
            )

            img = utils.outputs(
                img,
                *pipeline_results, 
                GREEN, line_thickness, display_size,
                draw_recognitions, False
            )

            if out is not None:
                out.write(img)    

        vid.release()
        cv2.destroyAllWindows()

        if out is not None:
            out.release()


if __name__ == '__main__':
    main()