"""
Main entry point of the library
"""

import begin
import pytesseract

from hanashi.processor import page_processor


@begin.start
def extract_text(filename: "Path to the image"):
    """
    Extract text from an image
    """
    masks = page_processor.process(filename)
    lines = []
    for mask in masks:
        s = (pytesseract.image_to_string(mask[2])).strip()
        if s != "":
            lines.append(s)

    print("\n----\n".join(lines))


