import pdb
import os
import xml.etree.ElementTree as ET
import numpy as np

from os import listdir
from compiler.ast import flatten
from os.path import isfile, join
from src.config import SUSPICIOUS

class ResultsAnalyzer:

  def __init__(self, corpus, file_name):
    self.corpus = corpus
    self.file_name = file_name

  def get_files_in_folder(self):
    fileids = [f for f in listdir(SUSPICIOUS) if isfile(join(SUSPICIOUS, f))]
    return fileids

  def get_offset_from_xml(self):
    '''
    Returns offset and length of plagiarised 
    part of document.
    Takes self.file_name, adds .xml extention 
    and checks to see if the file exists.
    '''
    arr_offset_length = []
    fileids = self.get_files_in_folder()
    if self.file_name not in fileids:
      raise Exception("File %s not present in folder" %(self.file_name))
    else:
      try:
        current_directory=os.getcwd()
        os.chdir(SUSPICIOUS)
        
        root_file_name = self.file_name.split('.')[0]
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


  def get_plagiarised(self, xml_data):
    '''
    Returns plagiarized paragraph from suspicious file
    based on the provided offset and length.
    '''
    if self.file_name not in self.corpus.fileids():
      raise Exception("File %s not present in corpus" %(self.file_name))
    else:
      arr_plagiarised = []
      for xml_line in xml_data:
        arr_plagiarised.append(self.read_by_offset(xml_line['offset'], xml_line['length']))
      return arr_plagiarised
    
  def read_by_offset(self, offset, length):
    '''
    Returns text from file starting from offset.
    TODO: remove file_name and add it to self.
    '''
    f = open(self.file_name, 'r')
    f.seek(int(offset))
    toReturn = f.read(int(length))
    return toReturn

  def chunks_to_passages(self, dict_offset_index, chunks):
    '''
    Returns detected sentences using the offset of each sentence.
    @param: dict_offset_index - contains where each sentence starts and its length
    @param: chunks - index of possbile plagiarized sentences.
    @return arr_passages - string with actual characters plagiarized from source file.
    '''
    arr_passages = []
    for chunk in chunks:
      if len(chunk) == 1:
        offset = dict_offset_index[chunk[0]][0]
        length = dict_offset_index[chunk[0]][-1]
        arr_passages.append(self.read_by_offset(offset, length))
      elif len(chunk) > 1:
        offset = dict_offset_index[chunk[0]][0]
        length = 0
        for item in chunk:
          pdb.set_trace()
          length += dict_offset_index[item][-1]
        arr_passages.append(self.read_by_offset(offset, length))
    return flatten(arr_passages)
