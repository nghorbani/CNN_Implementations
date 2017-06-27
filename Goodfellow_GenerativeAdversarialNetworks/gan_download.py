"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py

Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""

from __future__ import print_function
import os
import sys
import shutil
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib

def _list_categories(tag):
  url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
  f = urllib.request.urlopen(url)
  return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
  url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
      '&category={category}&set={set_name}'.format(**locals())
  print(url)
  if set_name == 'test':
    out_name = 'test_lmdb.zip'
  else:
    out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
  out_path = os.path.join(out_dir, out_name)
  cmd = ['curl', url, '-o', out_path]
  print(url)
  print('Downloading', category, set_name, 'set')
  #subprocess.call(cmd)

def download_lsun(dirpath):
  data_dir = os.path.join(dirpath, 'lsun')
  if os.path.exists(data_dir):
    print('Found LSUN - skip')
    return
  else:
    os.mkdir(data_dir)

  tag = 'latest'
  #categories = _list_categories(tag)
  categories = ['bedroom']

  for category in categories:
    _download_lsun(data_dir, category, 'train', tag)
    _download_lsun(data_dir, category, 'val', tag)
  _download_lsun(data_dir, '', 'test', tag)

def download_mnist(dirpath):
  data_dir = os.path.join(dirpath, 'mnist')
  if os.path.exists(data_dir):
    print('Found MNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://yann.lecun.com/exdb/mnist/'
  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir,file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)

def prepare_data_dir(path = './data'):
  if not os.path.exists(path):
    os.mkdir(path)

if __name__ == '__main__':
  prepare_data_dir()
  #download_lsun('./data')
  download_mnist('./data')
