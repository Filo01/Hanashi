"""
This module extracts text from manga images.
"""

import cv2
import logging
import random
from itertools import product

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageOps

from hanashi.model.rectangle import Rectangle
from hanashi.model.ufarray import UFarray

logger = logging.getLogger("CCL")
logger.setLevel(logging.INFO)
logging.basicConfig(format='[%(asctime)-15s %(levelname)s] [%(name)s] %(message)s')


def crop_size(verts):
    """
    Calculates the sides of the
    bounding box for a series of points
    :param verts:
    :return:
    :rtype: (int, int, int, int)
    """
    x_list = [v[0] for v in verts]
    y_list = [v[1] for v in verts]
    x_max = max(x_list)
    x_min = min(x_list)
    y_max = max(y_list)
    y_min = min(y_list)
    return x_max, x_min, y_max, y_min


def make_component_dict(labels, uf_arr, use_colors=False):
    components = {}
    colors = {}

    for coords, label in labels.items():
        # Name of the component the current point belongs to
        component = uf_arr.find(label)

        # Update the labels with correct information
        # labels[(x, y)] = temp
        components_list = components.get(component, [])
        components_list.append(coords)
        components[component] = components_list

        # Colorize the image
        if use_colors:
            colors[component] = (random.randint(0, 255),
                                 random.randint(0, 255),
                                 random.randint(0, 255))

    return components


def threshold_components(original_components, min_threshold=None, max_threshold=None):
    components = dict()

    for label, points in original_components.items():
        if min_threshold and len(points) > min_threshold:
            components[label] = points
        elif max_threshold and len(points) < max_threshold:
            components[label] = points
        elif min_threshold is None and max_threshold is None:
            components[label] = points

    return components


def connected_components(img, use_colors=False):
    """
    Finds all the black objects in an image and
    returns them in a dict. The dict has int keys
    for each component and a list of coordinates that belong to it
    :param img:
    :type img: PIL.Image
    :param use_colors: draw and show an image with the different
                    black components colored in different ways
    :type use_colors: bool
    :return: returns black components
    :rtype: dict[int,list]
    """
    data = img.load()
    width, height = img.size

    black_uf = UFarray()

    # Dictionary of point:label pairs
    black_labels = {}

    for y, x in product(range(height), range(width)):

        pixel = data[x, y]

        if pixel == 0:
            if y > 0 and data[x, y - 1] == 0:
                black_labels[x, y] = black_labels[(x, y - 1)]

            elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

                c = black_labels[(x + 1, y - 1)]
                black_labels[x, y] = c

                if x > 0 and data[x - 1, y - 1] == 0:
                    a = black_labels[(x - 1, y - 1)]
                    black_uf.union(c, a)

                elif x > 0 and data[x - 1, y] == 0:
                    d = black_labels[(x - 1, y)]
                    black_uf.union(c, d)

            elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
                black_labels[x, y] = black_labels[(x - 1, y - 1)]

            elif x > 0 and data[x - 1, y] == 0:
                black_labels[x, y] = black_labels[(x - 1, y)]

            else:
                black_labels[x, y] = black_uf.makeLabel()

    black_uf.flatten()
    black_temp = make_component_dict(black_labels, black_uf)

    max_threshold = width*height/15
    black_components = threshold_components(black_temp, max_threshold=max_threshold)

    for label in list(black_components.keys()):
        if len(black_components[label]) < 50:
            del black_components[label]

    if use_colors:
        """
        output_img = Image.new("RGB", (width, height))
        out_data = output_img.load()
        for component in black_components:
            for (x, y) in black_components[component]:
                if component in black_colors:
                    out_data[x, y] = black_colors[component]
        output_img.show()
        """
        pass

    return black_components


def threshold_image(img, level):
    """
    Apply threshold level to the supplied image
    :param img:
    :param level:
    :return: the resulting image
    :rtype: PIL.Image
    """
    img1 = Image.new("RGB", img.size)
    img1.paste(img)
    img1 = img1.point(lambda p: p > level and 255)
    img1 = img1.convert('1')

    return img1


def adaptive_threshold(filename):
    """
    Calculate the threshold
    :param filename: the filename of the image
    :return: threshold level
    :rtype: int
    """
    img = cv2.imread(filename, 0)

    mean = np.mean(img)
    if 100 > mean > 230:
        mean = mean + mean * 0.2

    return mean


def third_pass(rectangles):
    possible_letters = list()

    for rec in rectangles:
        top_rec = Rectangle(rec.x, rec.y - rec.height,
                            rec.height, rec.height)
        bottom_rec = Rectangle(rec.x, rec.y + rec.height,
                               rec.height, rec.height)
        right_rec = Rectangle(rec.x + rec.width, rec.y,
                              rec.height, rec.height)
        left_rec = Rectangle(rec.x - rec.width, rec.y - rec.height,
                             rec.height, rec.height)

        for rect2 in rectangles:
            if (rect2.intersects(top_rec) or
                    rect2.intersects(right_rec) or
                    rect2.intersects(bottom_rec) or
                    rect2.intersects(left_rec) or
                    rect2.intersects(rec)):
                if rec not in possible_letters:
                    possible_letters.append(rec)

    logger.info("found " + str(len(possible_letters)) + " possible letters")
    return possible_letters


def is_first_letter(rect, rectangles):
    left_rec = Rectangle(rect.x - rect.height * 0.8,
                         rect.y, rect.height, rect.height)
    is_first = True
    for rect1 in rectangles:
        if rect1 is not rect and rect1.overlaps_with(left_rec):
            is_first = False
            break
    return is_first


def remove_overlappings(rectangles):
    remove = []
    i = 0
    for rect in rectangles:
        for rect1 in rectangles:
            if rect.overlap_percentage(rect1) > 30 or rect.height < 5 or rect.width < 5:
                remove.append(rect)
                i += 1
    for rect in rectangles[:]:
        if rect in remove:
            rectangles.remove(rect)
    logger.info("Removed " + str(i))
    return rectangles


def get_lines(rectangles):
    """
    Finds all rectangles that are aligned with each other
    and have similar height and groups them
    :param rectangles:
    :type: Rectangle
    :return: groups of rectangles
    :rtype: dict[int, list]
    """
    rectangles.sort(key=lambda rec: rec.x)
    n = 0
    lines = dict()
    for rect in rectangles:
        if is_first_letter(rect, rectangles):

            last_rect = rect
            lines[n] = list()
            lines[n].append(last_rect)
            for rect1 in rectangles:
                right_last_rect = Rectangle(last_rect.x + last_rect.width,
                                            last_rect.y, last_rect.height * 1.5,
                                            last_rect.height)

                if rect is not rect1 and \
                        rect.inline(rect1) and \
                        right_last_rect.intersects(rect1):  # and \
                    # math.sqrt(math.pow(r.height-o.height, 2)) < 0.5*r.height:
                    last_rect = rect1
                    lines[n].append(last_rect)

            n += 1
    result = []
    for key in lines:
        line = line_bounding_box(lines[key])
        result.append(line)
    return result


def line_bounding_box(line):
    left = min([v.l_top.x for v in line])
    right = max([v.r_bot.x for v in line])
    top = min([v.l_top.y for v in line])
    bottom = max([v.r_bot.y for v in line])
    return Rectangle(left, top, (right - left), (bottom - top))


def group_lines(lines):
    """
    Groups lines together
    :param lines: dictionary that contains all the
    black pixels in a line
    :type lines: dict[int, list]
    :return:
    :rtype: dict[int, list[Rectangle]]
    """
    bounding_boxes = dict()
    uf_arr = UFarray()
    n = 0
    for line in lines:
        bounding_boxes[n] = line
        n += 1

    groups = dict()
    uf_arr.setLabel(len(bounding_boxes))
    for n in bounding_boxes:
        rect = bounding_boxes[n]
        top_rect = Rectangle(rect.x, rect.y + rect.height,
                             rect.width, rect.height)
        bottom_rect = Rectangle(rect.x, rect.y - rect.height,
                                rect.width, rect.height)
        for k in bounding_boxes:
            rect1 = bounding_boxes[k]
            if rect is not rect1:
                if (rect1.intersects(bottom_rect) or
                        rect1.intersects(top_rect)) and \
                        abs(rect.height - rect1.height) < 0.3 * rect.height:
                    uf_arr.setLabel(max(n, k))
                    uf_arr.union(n, k)
    uf_arr.flatten()

    for n in bounding_boxes:
        index = uf_arr.find(n)
        line_list = groups.get(index, list())
        line_list.append(bounding_boxes[n])
        groups[index] = line_list
    return groups


def white_percent(img, img2):
    img2 = img2.convert('L')
    inverted_poly = ImageOps.invert(img2)
    img1 = img.copy()
    img1.paste(inverted_poly, (0, 0), mask=inverted_poly)

    white = 0
    for pixel in img1.getdata():
        if pixel == (255, 255, 255):
            white += 1

    width, height = img.size
    area = float(width * height)
    return white / area * 100.0


def crop_size_rectangles(rectangles):
    (x_max, x_min, y_max, y_min) = (0, float("inf"), 0, float("inf"))
    for rect in rectangles:
        x_max = max(rect.r_bot.x, x_max)
        x_min = min(rect.l_top.x, x_min)
        y_max = max(rect.r_bot.y, y_max)
        y_min = min(rect.l_top.y, y_min)
    return x_max, x_min, y_max, y_min


def mask_groups(img, groups):
    """
    Returns list of masked images
    :param img: image to mask
    :type img: Image
    :param groups: group of rectangles to use as masks
    :return: list of masked images and their
    top left corner position on the original image
    :rtype: list[int, int, Image, list[Rectangles]]

    """
    masks = []

    for label in groups:
        line_length = len(groups[label])
        if line_length > 1 or \
                (line_length == 1 and
                    groups[label][0].width/groups[label][0].height > 2 and
                    groups[label][0].width * groups[label][0].height > 500):
            masked_img = Image.new("L", img.size, color=255)
            draw = ImageDraw.Draw(masked_img)
            (x_max, x_min, y_max, y_min) = crop_size_rectangles(groups[label])
            bounding_box = Rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
            for rect in groups[label]:
                rect.draw(draw, fill=True)
            temp_img = img.copy()
            temp_img.paste(masked_img, mask=masked_img)
            temp_img = temp_img.crop((bounding_box.x, bounding_box.y,
                                      bounding_box.x + bounding_box.width,
                                      bounding_box.y + bounding_box.height))

            masks.append((bounding_box.x, bounding_box.y, temp_img, groups[label]))

    return masks


def mask_img(img, masks):
    """

    :param img:
    :param masks:
    :return:
    """
    line_masks = list()
    for rect in masks:
        img2 = Image.new("L", img.size, color="white")
        rect.draw(ImageDraw.Draw(img2), fill=True)
        img3 = img.copy()
        img3.paste(img2, mask=img2)
        img3 = img3.crop((rect.l_top.x, rect.l_top.y, rect.r_bot.x, rect.r_bot.y))
        line_masks.append((rect.l_top.x, rect.l_top.y, img3))

    return line_masks


def compare_image(original, masks):
    original = original.copy()
    original = original.resize([int(2 * s) for s in original.size], Image.ANTIALIAS)
    img3 = Image.new("RGB", (original.size[0] * 2, original.size[1]))
    img3.paste(original, box=(original.width, 0))
    for mask in masks:
        img3.paste(mask[2], box=(mask[0], mask[1]))
    img3 = img3.resize([int(0.5 * s) for s in img3.size], Image.ANTIALIAS)
    return img3


def generate_rectangles(components):
    """
    Calculates the bounding box of each component
    :param components:
    :type components: dict[int, list]
    :return:
    :rtype: list[Rectangle]
    """
    rectangles = []
    for component in components:
        (x_max, x_min, y_max, y_min) = crop_size(components[component])
        try:
            rect = Rectangle(x_min, y_min, x_max - x_min, y_max - y_min,
                             points=components[component], label=component)
        except AssertionError:
            continue
        rectangles.append(rect)
    return rectangles


def process(filename):
    """
    Process an page of a manga and return a list of images that contain text
    :param filename:
    :return: list of (Image objects, (x,y)position on the original image)
    :rtype: list
    """
    img = Image.open(filename)
    img = img.resize([int(2 * s) for s in img.size], Image.ANTIALIAS)
    threshold = adaptive_threshold(filename)
    threshold_img = threshold_image(img.filter(ImageFilter.BLUR), threshold)
    black_components = connected_components(threshold_img)
    logger.debug("Removing uninteresting objects")
    rectangles = generate_rectangles(black_components)
    logger.debug("Getting Lines")
    lines = get_lines(rectangles)
    groups = group_lines(lines)
    logger.debug("Applying mask")
    masks = mask_groups(img, groups)
    return masks


if __name__ == "__main__":

    pass
