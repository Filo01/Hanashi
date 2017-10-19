"""
This module extracts text from manga images.
"""

import cv2
import logging
import pytesseract
import math
import random
import numpy as np
from PIL import Image, ImageDraw

from hanashi.model.rectangle import Rectangle
from hanashi.model.ufarray import UFarray
from hanashi.model.quadtree import Quadtree
from hanashi.model.contour_tree import Tree, Node

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


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2_connected_components(img1, min_size=50, max_size=100000):
    img1 = cv2.bitwise_not(img1)
    labelnum, labels, stats, centroids = cv2.connectedComponentsWithStats(img1)

    rectangles = []

    for label in range(1, labelnum):

        x, y, w, h, size = stats[label]
        rect = Rectangle(x,y,w,h)

        if min_size < rect.area() < max_size:
            rectangles.append(rect)


    return labelnum, labels, rectangles


def find_bubbles(labelnum, labels, rectangles, img, show_image=False):
    height, width = img.shape
    colors = dict()
    colors[0] = 0
    for label in range(1, labelnum):
        labels[labels == label] = random.randint(0, 255)
    labels = labels.astype(np.uint8)
    img1 = cv2.bitwise_not(img)

    _, contours, hierarchy = cv2.findContours(
        img1, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    area = float(labels.shape[0] * labels.shape[1])
    medium_tree = Tree()
    large_tree = Tree()
    for elem in hierarchy[0]:
        contour = contours[n]
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = Rectangle(x, y, w, h)
        box_area = bounding_box.area()
        node = Node(elem[3], n, contour, bounding_box)
        if box_area > area / 10:
            large_tree.add(node)
        elif box_area > area / 1000:
            medium_tree.add(node)
        n += 1

    possible_bubbles = [(node, level)
                        for node, level in
                        medium_tree.level_order_traversal()]
    img2 = np.zeros((height, width, 3), np.uint8)

    if show_image:
        for node, level1 in medium_tree.level_order_traversal():
            if node.n != -1:
                cv2.drawContours(img2, contours, node.n,
                                 (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        for rect in rectangles:
            print("Drawing ", rect)
            img2 = cv2.rectangle(img2, (rect.x, rect.y), (rect.r_bot.x, rect.r_bot.y), (255, 255, 0), 1)

            """
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (rect.l_bot.x, rect.l_bot.y)
            fontScale = 0.4
            fontColor = (255, 255, 255)
            cv2.putText(img2, str(rect.area()),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor)
            """
        show(img2)
    return possible_bubbles

def adaptive_segmentation(cv2_img):
    segmentation_levels = []
    resize_factor = 1
    cv2_img = cv2.resize(cv2_img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    min_size = 10
    max_size = 100000
    resize_factor *= resize_factor
    min_size *= resize_factor
    max_size *= resize_factor
    prev_len = float("inf")
    mean = int(cv2.mean(cv2_img)[0])
    height, width = cv2_img.shape
    for level in range(mean, 230, 5):
        thresholded = threshold_image(cv2_img.copy(), level)
        labelnum, labels, rectangles = cv2_connected_components(thresholded, min_size, max_size)
        #possible_bubbles = merge_bubbles(possible_bubbles)
        segmentation_levels.append((level, rectangles))
        n_rectangles = len(rectangles)
        if n_rectangles < prev_len:
            prev_len = n_rectangles
        else:
            break
    possible_bubbles = find_bubbles(labelnum, labels, rectangles, thresholded)
    rectangles = overlaps(rectangles, possible_bubbles, width, height)

    return rectangles, possible_bubbles, thresholded


def iterate_rectangles(rectangles1, rectangles2, width, height):
    quadtree = Quadtree(0, Rectangle(0, 0, width, height))
    for rect in rectangles2:
        if rect:
            quadtree.insert(rect)
    for rect in rectangles1:
        neighbours = []
        quadtree.retrieve(neighbours, rect)
        for rect1 in neighbours:
            yield rect, rect1


def merge_bubbles(bubbles, width, height):
    pass

def overlaps(rectangles, possible_bubbles, width, height):
    bubbles = [bubble[0].box for bubble in possible_bubbles]
    rectangles2 = [rect for rect, bubble in
                   iterate_rectangles(rectangles, bubbles, width, height)
                   if rect in bubble]
    return rectangles2


def threshold_image(img, level):
    """
    Apply threshold level to the supplied image
    :param img:
    :param level:
    :return: the resulting image
    :rtype: cv2 image
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, dst = cv2.threshold(blur, level, 255, cv2.THRESH_BINARY)

    return dst


def adaptive_threshold(image):
    """
    Calculate the threshold
    :param image: the cv2 image
    :return: threshold level
    :rtype: int
    """

    mean = np.mean(image)
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


def is_first_letter(rect, quadtree, draw = None):
    left_rec = Rectangle(rect.x - rect.height,
                         rect.y, rect.height, rect.height)
    if draw:
        left_rec.draw(draw, outline="yellow")
    neighbours = []
    quadtree.retrieve(neighbours, rect)
    for rect1 in neighbours:
        if rect1 is not rect and rect1.overlaps_with(left_rec):
            return False
    return True


def remove_overlaps(rectangles, width, height):
    remove = []
    i = 0
    for rect, rect1 in iterate_rectangles(rectangles, rectangles, width, height):
        percentage = rect.overlap_percentage(rect1)
        if percentage == 100:
            remove.append(rect1)
            i += 1
    for rect in rectangles[:]:
        if rect in remove:
            rectangles.remove(rect)
    logger.info("Removed " + str(i))
    return rectangles


def get_lines(rectangles, width, height):
    """
    Finds all rectangles that are aligned with each other
    and have similar height and groups them
    :param rectangles:
    :type: Rectangle
    :return: groups of rectangles
    :rtype: dict[int, list]
    """
    n = 0
    lines = dict()
    quadtree = Quadtree(0, Rectangle(0,0,width, height))
    for rect in rectangles:
        quadtree.insert(rect)
    for rect in rectangles:
        is_first = is_first_letter(rect, quadtree)
        if is_first:

            last_rect = rect
            lines[n] = list()
            lines[n].append(last_rect)
            neighbours = []
            quadtree.retrieve(neighbours, rect)
            for rect1 in sorted(neighbours, key=lambda rec: rec.x):
                right_last_rect = Rectangle(last_rect.x + last_rect.width,
                                            last_rect.y, last_rect.height * 1.5,
                                            last_rect.height)

                if rect is not rect1 and \
                        rect.inline(rect1) and \
                        right_last_rect.intersects(rect1) and \
                        math.sqrt(pow(rect.height-rect1.height, 2)) < 0.5*rect.height:
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


def crop_size_rectangles(rectangles):
    (x_max, x_min, y_max, y_min) = (0, float("inf"), 0, float("inf"))
    for rect in rectangles:
        x_max = max(rect.r_bot.x, x_max)
        x_min = min(rect.l_top.x, x_min)
        y_max = max(rect.r_bot.y, y_max)
        y_min = min(rect.l_top.y, y_min)
    return x_max, x_min, y_max, y_min


def mask_groups(img, groups, possible_bubbles):
    """
    Returns list of masked images
    :param img: image to mask
    :type img: Image
    :param groups: group of rectangles to use as masks
    :return: list of masked images and their
    top left corner position on the original image
    :rtype: list[int, int, Image, Rectangle, list[Rectangles], list[Rectangle]]

    """
    masks = []
    width, height = img.size
    used_bubbles = []
    for label in groups:
        lines = groups[label]
        (x_max, x_min, y_max, y_min) = crop_size_rectangles(groups[label])
        bounding_box = Rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
        line_length = len(lines)
        if line_length <= 1:
            continue
        """
        if line_length > 1 or \
                (line_length == 1 and
                    lines[0].width/lines[0].height > 2 and
                    lines[0].width * lines[0].height > 500):
        """
        highest_level = 0
        index = -1
        for i, bubble in enumerate(possible_bubbles):
            if i != 0 and \
                    bounding_box in bubble[0].box:
                if i in used_bubbles:
                    break
                if highest_level < bubble[1]:
                    highest_level=bubble[1]
                    index = i
        if index > 0 and index not in used_bubbles:
            used_bubbles.append(index)
            bubble = possible_bubbles[index][0]
            cv2_img = np.full((height, width), 255, dtype=np.uint8)
            cv2.drawContours(cv2_img, [bubble.contour], 0, 0, -1)

            masked_img = Image.fromarray(cv2_img)
            draw = ImageDraw.Draw(masked_img)

            for rect in lines:
                rect.draw(draw, fill=True)
            temp_img = img.copy()
            temp_img.paste(masked_img, mask=masked_img)
            bounding_box = bubble.box
            temp_img = temp_img.crop((bounding_box.x, bounding_box.y,
                                      bounding_box.x + bounding_box.width,
                                      bounding_box.y + bounding_box.height))
            masks.append((bounding_box.x, bounding_box.y, temp_img, bounding_box, lines, bubble))

            continue

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


def compare_with_original(filename, masks):
    original = Image.open(filename)
    #original = original.resize([int(2 * s) for s in original.size], Image.ANTIALIAS)
    img3 = Image.new("RGB", (original.size[0] * 2, original.size[1]), color="white")
    img3.paste(original, box=(original.width, 0))
    for mask in masks:
        img3.paste(mask[2], box=(mask[0], mask[1]))
    #img3 = img3.resize([int(0.5 * s) for s in img3.size], Image.ANTIALIAS)
    return img3


def apply_masks(original, masks):

    masked = np.zeros(original.shape, dtype=np.uint8)
    for mask in masks:
        cv2.drawContours(masked,[mask[5].contour],0, 255, -1)
        for line in mask[4]:
            cv2.rectangle(masked,
                          (line.l_top.x, line.l_top.y),
                          (line.r_bot.x, line.r_bot.y),
                          255, thickness=cv2.FILLED)
    fg_masked = cv2.bitwise_or(original, original, mask=masked)
    masked = cv2.bitwise_not(masked)
    bk = np.full(original.shape, 255, dtype=np.uint8)
    bk_masked = cv2.bitwise_and(bk, bk, mask=masked)
    final = cv2.bitwise_or(fg_masked, bk_masked)

    return final


def compare_image(img1, img2):
    img3 = cv2.subtract(img1, img2)
    img1 = cv2.bitwise_not(img1)
    total = cv2.countNonZero(img1)
    non_zero = cv2.countNonZero(img3)
    return float(non_zero)/total*100


def process(filename):
    """
    Process an page of a manga and return a list of images that contain text
    :param filename:
    :return: list of (Image objects, (x,y)position on the original image)
    :rtype: list
    """
    img = Image.open(filename)
    cv2_img = cv2.imread(filename,0)
    width, height = img.size

    rectangles, possible_bubbles = adaptive_segmentation(cv2_img)
    #rectangles = remove_overlaps(rectangles, width, height)
    logger.debug("Getting Lines")
    lines = get_lines(rectangles, width, height)
    groups = group_lines(lines)
    logger.debug("Applying mask")
    masks = mask_groups(img, groups, possible_bubbles)
    return masks, lines, rectangles


def extract_text(masks):
    resize_factor = 2.5
    if not masks:
        return "", []
    width = max([int(mask[2].size[0]*resize_factor) for mask in masks])
    height = sum([int(mask[2].size[1]*resize_factor) for mask in masks])
    image = Image.new("RGB", (width, height))
    height = 0
    masks1 = []
    text = []
    for mask in masks:
        resized = mask[2].resize([int(n*resize_factor) for n in mask[2].size], Image.ANTIALIAS)
        s = (pytesseract.image_to_string(resized)).strip()
        if s != "":
            masks1.append(mask)
            text.append(s)
        """
        box = masked_img.size
        x = int(width/2 - box[0]/2)
        y = height
        image.paste(masked_img, box=(x,y))
        height += box[1]
        """

    return text, masks1


if __name__ == "__main__":
    filename = "/media/filippo/HDD1/pythonProjects/Github-Hanashi/Hanashi/Hanashi/tests/resources/onepunch.jpg"

    img = cv2.imread(filename,0)

    pass
