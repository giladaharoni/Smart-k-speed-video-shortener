import random
import sys
from functools import partial

import skimage.metrics
from skimage.morphology import skeletonize
import numpy as np
import cv2
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import torch


def difference_function(first_frame, second_frame):
    return np.linalg.norm(first_frame - second_frame)


def binarization_preprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thinned = skeletonize(thresh)
    return thinned


def deep_preprocessing(frame, processor, model):
    inputs = processor(images=[frame], return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits


def tensor_distance(log1, log2):
    return float(torch.sqrt(torch.sum(torch.pow(torch.subtract(log1[0], log2[0]), 2), dim=0)))


def maximal_subset(elements, k_frames):
    new_elements = [(i, cost) for i, cost in enumerate(elements)]
    new_elements.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [i for i, cost in new_elements[:k_frames]]
    selected_indices.sort(reverse=False)
    return selected_indices


def draw_play_bar(frame, number, markers, total_frames, height, width):
    def position(marker):
        return int(start_bar + (end_bar - start_bar) * (marker / total_frames))

    start_bar = int(width / 10)
    end_bar = int(9 * width / 10)
    height_line = int(9 * height / 10)
    height_unit = int(height / 10)
    white = (255, 255, 255)
    new_frame = cv2.line(frame, (start_bar, height_line), (end_bar, height_line), white, 2)
    # draw the ends
    new_frame = cv2.line(new_frame, (start_bar, int(height_line + 0.5 * height_unit)), (start_bar,
                                                                                        int(height_line - 0.5 * height_unit)),
                         white, 3)
    cv2.line(frame, (end_bar, int(height_line + 0.5 * height_unit)), (end_bar, int(height_line - 0.5 * height_unit)),
             white, 3)
    # draw the markers
    for mark in markers:
        positio = position(mark)
        new_frame = cv2.line(new_frame, (positio, int(height_line + 0.2 * height_unit)), (positio,
                                                                                          int(height_line - 0.2 * height_unit)),
                             white, 2)
    current_pos = position(number)
    new_frame = cv2.line(new_frame, (current_pos, int(height_line + 0.3 * height_unit)), (current_pos,
                                                                                          int(
                                                                                              height_line - 0.3 * height_unit)),
                         (250, 0, 0), 3)
    return new_frame


def frame_importance_processing(cap, total_frame_count, method, preprocessing):
    frame_number = 0
    last_frame = None
    frames_importance = []
    while True:
        ret, frame = cap.read()
        processed_frame = preprocessing(frame)
        if not ret:
            break  # Break the loop if no frames are left
        if frame_number > 0:
            frames_importance.append(method(last_frame, processed_frame))

        frame_number += 1
        sys.stdout.write(f"\rProcessing frame {frame_number} of {total_frame_count}")
        sys.stdout.flush()
        last_frame = processed_frame
    print("\nprocessing frames is completed")
    return frames_importance


def output_selected_frames(frame_indices, cap, out, bar):
    for i, index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at index {index}.")
            continue
        if bar:
            new_frame = draw_play_bar(frame, index, frame_indices, total_frame_count, frame_height, frame_width)
        else:
            new_frame = frame
        sys.stdout.write(f"\rWriting frame {i} of {reduced_frame_count}")
        sys.stdout.flush()
        out.write(new_frame)
    print("\nsaving is completed")


def parse_args():
    args = sys.argv
    I_path, O_path, short_factor, method, draw_bar = args[1:]
    if method == '3':
        model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    else:
        model = None
        processor = None

    method_dict = {
        '1': (difference_function, lambda x: x),
        '2': (skimage.metrics.hausdorff_distance, binarization_preprocessing),
        '3': (tensor_distance, partial(deep_preprocessing, model=model, processor=processor))
    }

    return {'input': I_path, 'output': O_path, 'method': method_dict[method], 'playbar': eval(draw_bar),
            'short_factor': eval(short_factor)}


if __name__ == '__main__':
    config = parse_args()
    video_path = config['input']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reduced_frame_count = round(total_frame_count / config['short_factor'])

    frames_importance = frame_importance_processing(cap, total_frame_count, *config['method'])
    frame_indices = maximal_subset(frames_importance, reduced_frame_count)
    output_video_path = config['output']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    output_selected_frames(frame_indices, cap, out, config['playbar'])

    cap.release()
    cv2.destroyAllWindows()
