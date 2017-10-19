"""
Main entry point of the library
"""
import argparse
import cv2
import os

from hanashi.processor import page_processor


def page(filename):
    """
    Extract text from a single page.
    """
    masks, lines, rectangles = page_processor.process(filename)
    text,  masks= page_processor.extract_text(masks)
    return text, masks


def compare(path, output=None):
    """Compare original image with result
    :param path: Path of the image
    :param output: Path to save the image comparison"""
    if path.endswith("jpg") or path.endswith("png"):
        filenames = [path]
    else:
        filenames = [os.path.join(path, f) for f in os.listdir(path)
                    if f.endswith("jpg") or f.endswith("png")]
    for filename in filenames:
        print(filename)
        text, masks = page(filename)
        masked = page_processor.compare_with_original(filename, masks)
        if output:
            masked.save(output)
        else:
            masked.show()


def extract_text(
        path: "Path to the folder containing "
                "all the pages for a single chapter or"
                "a single page"):
    """
    Extract text from a series of pages or a single page.
    """
    if path.endswith("jpg") or path.endswith("png"):
        filenames = [path]
    else:
        filenames = [os.path.join(path, f) for f in os.listdir(path)
                    if f.endswith("jpg") or f.endswith("png")]

    for filename in filenames:
        text, masks = page(filename)
        print("================================")
        print("Filename: ", filename)
        print("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from manga' )
    """
    parser_a = parser.add_subparsers().add_parser('compare', help='Compare the original image to the final image')
    parser_a.
    parser_a.set_defaults(func=compare)

    """
    parser.add_argument('--compare', '-c',required=False, action='store_false',
                        help='Show comparison between original and processed image')
    parser.add_argument('--output', '-o', required=False,
                        help='Output path of the comparison image')
    parser.add_argument('path', type=str, help='path to the folder or single image to analyze')
    args= parser.parse_args()
    if args.compare is False:
        if args.output:
            compare(args.path, args.output)
        else:
            compare(args.path, args.compare)
    else:
        extract_text(args.path)

