#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import yaml
import os
import sys
import cv2

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import tf_retinanet.bin  # noqa: F401
    __package__ = "tf_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..utils.visualization import draw_annotations, draw_boxes, draw_caption
from ..utils.anchors import anchors_for_shape, compute_gt_annotations
from ..generators import get_generators
from ..backbones import get_backbone



def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')

    parser.add_argument('config', help='Configuration file.')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--display-name', help='Display image name on the bottom left corner.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def run(generator, args):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    i = 0
    while True:
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)
        if len(annotations['labels']) > 0 :
            # apply random transformations
            if args.random_transform:
                image, annotations = generator.random_transform_group_entry(image, annotations)
                image, annotations = generator.random_visual_effect_group_entry(image, annotations)

            # resize the image and annotations
            if args.resize:
                image, image_scale = generator.resize_image(image)
                annotations['bboxes'] *= image_scale

            anchors = anchors_for_shape(image.shape)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

            # draw anchors on the image
            if args.anchors:
                draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

            # draw annotations on the image
            if args.annotations:
                # draw annotations in red
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

                # draw regressed anchors in green to override most red annotations
                # result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

            # display name on the image
            if args.display_name:
                draw_caption(image, [0, image.shape[0]], os.path.basename(generator.image_path(i)))

        cv2.imshow('Image', image)
        key = cv2.waitKey()
        print(int(key))

        # note that the right and left keybindings are probably different for windows
        # press right for next image and left for previous (linux)
        # if you run macOS, it might be convenient using "n" and "m" key (key == 110 and key == 109)

        if key == 100:
            i = (i + 1) % generator.size()
        if key == 97:
            i -= 1
            if i < 0:
                i = generator.size() - 1

        # press q or Esc to quit
        if (key == ord('q')) or (key == 27):
            return False

    return True

def set_defaults(config):
    if not config['callbacks']['snapshots_path']:
        from pathlib import Path
        home = str(Path.home())
        config['callbacks']['snapshots_path'] = os.path.join(home, 'retinanet-snapshots')
    if not config['callbacks']['project_name']:
        from datetime import datetime
        config['callbacks']['project_name'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return config

def parse_yaml(config):
    # TODO get the filename using a parser
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return set_defaults(config)
        except yaml.YAMLError as exc:
            raise(exc)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = parse_yaml(args.config)
    # Get the backbone.
    backbone = get_backbone(config)
    # Get the generators.
    generators = get_generators(
        config,
        preprocess_image=backbone.preprocess_image
    )

    generator = generators['validation']

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    run(generator, args)


if __name__ == '__main__':
    main()
