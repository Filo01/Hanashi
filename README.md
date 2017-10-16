## Hanashi

A library to extract text from mangas.

OCR is done by tesseract.

## Code Example


```python
import pytesseract
from import hanashi.processor import page_processor

masks, lines, rectangles = page_processor.process(filename)
results = page_processor.extract_text(masks)
blocks_of_text = [result[0] for result in results]
print("\n----\n".join(blocks_of_text))
```