import xml.etree.ElementTree as ET

from os import listdir
from os.path import isfile, join
from src.config import SUSPICIOUS

class ResultsAnalyzer:

  def __init___(self, corpus):
    self.corpus = corpus

  def get_files_in_folder(self):
    fileids = [f for f in listdir(SUSPICIOUS) if isfile(join(SUSPICIOUS, f))]
    return fileids

  def get_offset_from_xml(self, file_name):
    '''
    Returns offset and length of plagiarised 
    part of document.
    Takes file_name, adds .xml extention 
    and checks to see if the file exists.
    @param file_name: name of file to search.
    '''
    fileids = self.get_files_in_folder()
    if file_name not in fileids:
      raise Exception("File %s not present in folder" %(file_name))
    else:
      root_file_name = file_name.split('.')[0]
      root_file_name += '.xml'
      if root_file_name not in fileids:
        raise Exception("File %s not present in folder" %(file_name))
      tree = ET.parse(root_file_name)
      root = tree.getroot()
      for child in root:
        return {
          "length": child['this_length'],
          "offset": child['this_offset'],
        }


  def get_plagiarised(self, file_name, offset, length):
    '''
    Returns plagiarized paragraph from suspicious file
    based on the provided offset and length.
    '''
    if file_name not in self.corpus.fileids():
      raise Exception("File %s not present in corpus" %(file_name))
    else:
      f = open(file_name, 'r')
      f.seek(offset)
      plagiarised = f.read(length)
      return plagiarised
