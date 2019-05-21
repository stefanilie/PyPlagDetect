import pdb
import os
import xml.etree.ElementTree as ET
import numpy as np

from os import listdir
from os.path import isfile, join
from src.config import SUSPICIOUS

class ResultsAnalyzer:

  def __init__(self, corpus):
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
    arr_offset_length = []
    fileids = self.get_files_in_folder()
    if file_name not in fileids:
      raise Exception("File %s not present in folder" %(file_name))
    else:
      try:
        current_directory=os.getcwd()
        os.chdir(SUSPICIOUS)
        
        root_file_name = file_name.split('.')[0]
        root_file_name += '.xml'
        if root_file_name not in fileids:
          raise Exception("File %s not present in folder" %(root_file_name))
        tree = ET.parse(root_file_name)
        root = tree.getroot()
        for child in root:
          arr_offset_length.append({
            "length": child.attrib['this_length'],
            "offset": child.attrib['this_offset'],
          })
        return arr_offset_length
      except:
        print "File %s not present in folder" %(root_file_name)


  def get_plagiarised(self, file_name, xml_data):
    '''
    Returns plagiarized paragraph from suspicious file
    based on the provided offset and length.
    '''
    if file_name not in self.corpus.fileids():
      raise Exception("File %s not present in corpus" %(file_name))
    else:
      arr_plagiarised = []
      f = open(file_name, 'r')
      for xml_line in xml_data:
        f.seek(int(xml_line['offset']))
        arr_plagiarised.append(f.read(int(xml_line['length'])))
      return arr_plagiarised

  def chunks_to_passages(self, sentences, chunks):
    arr_passages = []
    for chunk in chunks:
      if len(chunk) == 1:
        arr_passages.append(sentences[chunk[0]])
      elif len(chunk) > 1:
        arr_passages.append(sentences[chunk[0]:chunk[-1]])
    return arr_passages
