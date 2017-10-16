## Hanashi

A library to extract text from mangas.

OCR is done by tesseract.

## Code Example


```
masks = page_processor.process(filename)
    lines = []
    for mask in masks:
        s = (pytesseract.image_to_string(mask[2])).strip()
        if s != "":
            lines.append(s)

    print("\n----\n".join(lines))
```