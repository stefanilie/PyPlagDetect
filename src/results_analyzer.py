import pdb
import os
import xml.etree.ElementTree as ET
import numpy as np

from os import listdir
from compiler.ast import flatten
from os.path import isfile, join
from src.config import SUSPICIOUS, SUSPICIOUS_DOCUMENTS
from helper import Helper


class ResultsAnalyzer:

  def __init__(self, corpus, file_name):
    self.corpus = corpus
    self.file_name = file_name

  def get_files_in_folder(self):
    current_directory = os.getcwd()
    fileids = [f for f in listdir(current_directory) if isfile(join(current_directory, f))]
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
      raise Exception("\nFile %s not present in folder" %(self.file_name))
    else:
      try:
        current_directory=os.getcwd()

        os.chdir(current_directory)
        
        root_file_name = self.file_name.split('.')[0]
        root_file_name += '.xml'
        if root_file_name not in fileids:
          raise Exception("\nFile %s not present in folder" %(root_file_name))
        tree = ET.parse(root_file_name)
        root = tree.getroot()
        for child in root:
          arr_offset_length.append({
            "length": child.attrib['this_length'],
            "offset": child.attrib['this_offset'],
          })
        return arr_offset_length
      except:
        print "\nFile %s not present in folder" %(root_file_name)

  def get_plagiarised(self, xml_data):
    '''
    Returns plagiarized paragraph from suspicious file
    based on the provided offset and length.
    '''
    if self.file_name not in self.corpus.fileids():
      raise Exception("\nFile %s not present in corpus" %(self.file_name))
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
    @return arr_passages - arr of strings with actual characters plagiarized from source file.
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
          length += dict_offset_index[item][-1]
        arr_passages.append(self.read_by_offset(offset, length))
    return arr_passages

  def chunks_to_offset(self, dict_offset_index, chunks):
    '''
    Takes offset index of windows and suspect chunk indexes 
    and returns offset indexes of suspect windows.
    @param: dict_offset_index - contains where each sentence starts and its length
    @param: chunks - index of possbile plagiarized sentences.
    @return arr_passages - array of type [[a,b]] where 
    a is the begining offset
    b is the end offset.
    '''
    arr_passages = []
    for chunk in chunks:
      if len(chunk) == 1:
        offset = dict_offset_index[chunk[0]][0]
        length = dict_offset_index[chunk[0]][-1]
        arr_passages.append([offset, offset+length])
      elif len(chunk) > 1:
        offset = dict_offset_index[chunk[0]][0]
        length = 0
        for item in chunk:
          length += dict_offset_index[item][-1]
        arr_passages.append([offset, offset+length])
    return arr_passages

  def compare_offsets(self, arr_plag_offset, arr_suspect_offset):
    '''
    Compares only the offsets and tells how many chars were detected.
    '''
    if len(arr_plag_offset) == 0 or len(arr_suspect_offset) == 0:
      return 0, 0

    arr_overlap = [0] * len(arr_plag_offset)
    arr_suspect_overlap = [0] * len(arr_suspect_offset)
    
    for suspect_index, suspect_interval in enumerate(arr_suspect_offset):
      for plag_index, plag_interval in enumerate(arr_plag_offset):
        overlap = Helper.get_overlap(suspect_interval, plag_interval)
        # suspect_length = suspect_interval[1]-suspect_interval[0]
        # false_positive = suspect_length - overlap
        if overlap:
          if arr_overlap[plag_index]:
            arr_overlap[plag_index] += overlap
          else:
            arr_overlap[plag_index] = overlap
          if arr_suspect_overlap[suspect_index]:
            arr_suspect_overlap[suspect_index] += overlap
          else:
            arr_suspect_overlap[suspect_index] = overlap
    return arr_overlap, arr_suspect_overlap
        
        # TODO: remove this when certainly we don't need it.
        # if suspect_interval[0] > plag_interval[0] and suspect_interval[1] < plag_interval[1]:
        #     overlap = Helper.get_overlap(suspect_interval, plag_interval)
        #     if overlap:
        #       if arr_overlap[plag_index]:
        #         # Check why is this happening.
        #         pdb.set_trace()
        #     arr_overlap[plag_index] = overlap
        # elif suspect_interval[0] <= plag_interval[0]:   
        #   if suspect_interval[1] < plag_interval[0]:
        #     continue
        #   overlap = Helper.get_overlap(suspect_interval, plag_interval)
        #   if overlap:
        #     if arr_overlap[plag_index]:
        #       # Check why is this happening.
        #       pdb.set_trace()
        #     arr_overlap[plag_index] = overlap
        #   else: 
        #     continue
    