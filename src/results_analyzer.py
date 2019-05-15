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
    fileids = self.get_files_in_folder()
    if file_name not in fileids:
      raise Exception("File %s not present in folder" %(file_name))
    else:
      root_file_name = file_name.split('.')[0]
      root_file_name += '.xml'
      tree = ET.parse(root_file_name)
      root = tree.getroot()
      # TODO: return offset value


  def get_plagiarised(self, file_name, offset, length):
    '''
    Takes file_name
    '''
    if file_name not in self.corpus.fileids():
      raise Exception("File %s not present in corpus" %(file_name))
    else:
      f = open(file_name, 'r')
      f.seek(offset)
      plagiarised = f.read(length)
      return plagiarised
