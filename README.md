PyPlagDetect
============


Intrinsic plagiarism detection system built with Python 2.7, nltk and scikit-learn.

Works with `.txt` files **only**.

This is my Master's Degree project. It started as a course assignment, but I realized this is what I have to do.

Usage
------

After cloning, first install all necessary packages by running

```pip install requirements.txt```

To start the process, `python main.py`. Choose between the three modes available:

1. `Tokenize and export dump via Pickle`  - This will tokenize a whole wikipedia dump [link here](https://dumps.wikimedia.org/enwiki/20190901/enwiki-20190901-pages-articles-multistream.xml.bz2) and generate a `.pickle` binary file to be imported and thus quickening the execution times.

Licence MIT.

