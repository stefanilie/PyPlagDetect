# PyPlagDetect

Intrinsic plagiarism detection system built with Python 2.7, nltk and scikit-learn.

Works with `.txt` files **only**.

This is my Master's Degree project. It started as a course assignment, then it blossomed into this beast.

## Installation

1. Clone and open the folder ```git clone https://github.com/stefanilie/PyPlagDetect.git && cd PyPlagDetect```
2. (optional) Setup a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv) for less hassle.
3. Install requirements ```pip install -r requirements.txt```
4. (optional) If you want to use my [static `.pickle` files](https://drive.google.com/open?id=1gWe9ScXxVQeWGMkPpRb8RAnZ0OMHq1KU), download this zip and extract it.
   1. Run these commands in the root folder of the project:

   ```bash
    mkdir dumps OANC
    mv tokenized.pickle unigram_tagger.pickle wiki.pickle dumps
    unzip OANC-1.0.1-UTF8.zip -d OANC
   ```

5. (optional) If you want to generate your own `.pickle` files, you first have to download the most recent wikipedia dump. Mind you, you will need serious computing powers. The wiki xml has around 85GB.
   1. Go [here](https://dumps.wikimedia.org/enwiki/) 
   2. Choose on to the most recent folder, and download the file ending in `pages-articles-multistream.xml.bz2`
   3. Add the filename in `config.py` under `WIKI_FILE_NAME`
6. (optional) If you wish to test on the PAN 2009 Intrinsic Plagiarism Detection corpus, go [here](https://pan.webis.de/sepln09/pan09-web/plagiarism-detection.html) and download corpus. Store the fils inside a `suspicious-documents` folder inside the root of the project.

## Usage

To start the process, `python main.py`. Choose between the three modes available:

1. `Tokenize and export dump via Pickle`  - This will tokenize a whole wikipedia dump and generate a `.pickle` binary file to be imported and thus quickening the execution times.

2. `Import dumps using Pickle and Analyze PAN corpus` -  Imports existing file dumps for wikipedia and unigram tagger. If you wish to use your own, please first use `menu item 1`. Next up, we're presented with two options:

   1. `/suspicious: (21 files, 2 without real plag data)`: quick test folder, the one I use for demos

   2. `/suspicious-documents: PAN 2009 corpus`: as it's name suggests, this will analize the plagiarism and compare the results to see if the algorithm is right. It uses the PAN 2009 corpus. For this, you will need to do the 6th step of the `Installation` process.

3. `Analyse files (without precision output)`: analize files without computing performance of the algorithm. It will find the suspicious paragraphs and it will output them.


Licence MIT.

