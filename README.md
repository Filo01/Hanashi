# Hanashi

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

## Prerequisites

```
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Installing

First make a directory and clone the repo
```
cd ./github
git clone https://github.com/Filo01/Hanashi.git
```

Use venv and use setup.py
```
python setup.py install
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
