class ResultsAnalyzer:

  def __init___(self, corpus):
    self.corpus = corpus

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
