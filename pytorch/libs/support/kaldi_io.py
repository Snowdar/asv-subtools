# -*- coding: utf-8 -*-

# Copyright 2014-2019 Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

# Snowdar: add chunk parameter for random reading.
# Zheng Li: add chunk parameter for ali reading, 2021-06-08.

from __future__ import print_function
from __future__ import division

import numpy as np
import sys, os, re, gzip, struct

#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT']='/work'

# Add kaldi tools to path,
path = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/')
os.environ['PATH'] = path.readline().strip() + ':' + os.environ['PATH']
path.close()


#################################################
# Define all custom exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass

class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass

class SubprocessFailed(Exception): pass

#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'wb') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd

def open_with_prefix(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    prefix = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
            if prefix != "scp" and prefix != "ark":
                raise TypeError("Specifier of {} is not scp or ark.".format(file))
        if prefix is None:
            raise TypeError("Specifier of {} is None, please specify this with scp or ark.".format(file))
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'wb') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        raise TypeError("{} is not a file.".format(file))        
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return prefix, fd

# based on '/usr/local/lib/python3.6/os.py'
def popen(cmd, mode="rb"):
    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    import subprocess, io, threading

    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
        return

    # text-mode,
    if mode == "r":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    elif mode == "w":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    elif mode == "rb":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdout
    elif mode == "wb":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdin
    # sanity,
    else:
        raise ValueError("invalid mode %s" % mode)


def read_key(fd):
    """ [key] = read_key(fd)
     Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    assert('b' in fd.mode), "Error: 'fd' was opened in text mode (in python3 use sys.stdin.buffer)"

    key = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        key += char
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd, chunk=None):
    #Zheng Li 2021-06-08
    """ Alias to 'read_vec_int_ark()' """
    mat = read_vec_int_ark(file_or_fd, chunk=chunk)
    return mat

def read_vec_int_ark(file_or_fd, chunk=None):
    #Zheng Li 2021-06-08
    """ [mat] = read_vec_int_ark(file_or_fd)
     Create numpy matrix of (vector<int>) tuples, which reads from the ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        mat = read_vec_int(fd, chunk=chunk)
    finally:
        if fd is not file_or_fd: fd.close()
    return mat

def read_vec_int(file_or_fd, chunk=None):
    #Zheng Li 2021-06-08
    """ [int-vec] = read_vec_int(file_or_fd)
     Read kaldi integer vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()  
    if binary == '\0B': # binary flag
        assert(fd.read(1).decode() == '\4'); # int-size
        vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
        if vec_size == 0:
            return np.array([], dtype='int32')

        start_index = 0
        end_index = vec_size - 1 # 0-based

        if chunk is not None:
            start_index = int(chunk[0])
            end_index = int(chunk[1])

        cols = end_index - start_index + 1

        if start_index > 0: 
            fd.seek(start_index * 5, 1) # Seek from current position, '5' is the byte size of int, by Passionlee

        # Elements from int32 vector are sored,
        vec = np.frombuffer(fd.read(vec_size*5), dtype=[('size','int8'),('value','int32')], count=cols)
        vec = vec[:]['value'] # values are in 2nd column,

        return vec

    else: # ascii,
        print('ascii')
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        #ans = np.array(arr, dtype=int)
        mat = np.vstack(arr)
        return mat

    if fd is not file_or_fd : fd.close() # cleanup

# Writing,
def write_vec_int(file_or_fd, v, key=''):
    """ write_vec_int(f, v, key='')
     Write a binary kaldi integer vector to filename or stream.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_int(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,vec in dict.iteritems():
             kaldi_io.write_vec_flt(f, vec, key=key)
    """
    assert(isinstance(v, np.ndarray))
    assert(v.dtype == np.int32)
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # dim,
        fd.write('\4'.encode()) # int32 type,
        fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write('\4'.encode()) # int32 type,
            fd.write(struct.pack(np.dtype('int32').char, v[i])) # binary,
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

def read_vec_flt_auto(file):
    # Snowdar 2020-05-31
    # file: scp:foo.scp or ark:foo.ark
    prefix, fd = open_with_prefix(file)
    try:
        if prefix == "scp":
            for key, vec in read_vec_flt_scp(fd):
                yield key, vec
        elif prefix == "ark":
            for key, vec in read_vec_flt_ark(fd):
                yield key, vec
        else:
            raise TypeError("prefix of {} is not scp or ark.".format(file))
    finally:
        if fd is not file : fd.close()

# Reading,
def read_vec_flt_scp(file_or_fd):
    """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
     Returns generator of (key,vector) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,vec in kaldi_io.read_vec_flt_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            vec = read_vec_flt(rxfile)
            yield key, vec
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
     Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
     Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        ans = _read_vec_flt_binary(fd)
    else:    # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

def _read_vec_flt_binary(fd):
    header = fd.read(3).decode()
    if header == 'FV ' : sample_size = 4 # floats
    elif header == 'DV ' : sample_size = 8 # doubles
    else : raise UnknownVectorHeader("The header contained '%s'" % header)
    assert (sample_size > 0)
    # Dimension,
    assert (fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
    if vec_size == 0:
        return np.array([], dtype='float32')
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    return ans


# Writing,
def write_vec_flt(file_or_fd, v, key=''):
    """ write_vec_flt(f, v, key='')
     Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_flt(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,vec in dict.iteritems():
             kaldi_io.write_vec_flt(f, vec, key=key)
    """
    assert(isinstance(v, np.ndarray))
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if v.dtype == 'float32': fd.write('FV '.encode())
        elif v.dtype == 'float64': fd.write('DV '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % v.dtype)
        # Dim,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, v.shape[0])) # dim
        # Data,
        fd.write(v.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
     Returns generator of (key,matrix) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,mat in kaldi_io.read_mat_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            mat = read_mat(rxfile)
            yield key, mat
    finally:
        if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
     Returns generator of (key,matrix) tuples, read from ark file/stream.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the ark:
     for key,mat in kaldi_io.read_mat_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield key, mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_mat(file_or_fd, chunk=None):
    """ [mat] = read_mat(file_or_fd)
     Reads single kaldi matrix, supports ascii and binary.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2).decode()
        if binary == '\0B' :
            mat = _read_mat_binary(fd, chunk=chunk)
        else:
            assert(binary == ' [')
            mat = _read_mat_ascii(fd, chunk=chunk)
    finally:
        if fd is not file_or_fd: fd.close()
    return mat

def _read_mat_binary(fd, chunk=None):
    # chunk:[start_line_index, end_line_index] # Snowdar 2019-09-19
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'): return _read_compressed_mat(fd, header, chunk=chunk)
    elif header == 'FM ': sample_size = 4 # floats
    elif header == 'DM ': sample_size = 8 # doubles
    else: raise UnknownMatrixHeader("The header contained '%s'" % header)
    assert(sample_size > 0)
    # Dimensions
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]

    start_index=0
    end_index=rows - 1 # 0-based

    if chunk is not None:
        start_index = int(chunk[0])
        end_index = int(chunk[1])

    rows = end_index - start_index + 1

    if start_index > 0: fd.seek(start_index * cols * sample_size, 1) # Seek from current position

    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    mat = np.reshape(vec,(rows,cols))
    return mat

def _read_mat_ascii(fd, chunk=None):
    # chunk:[start_line_index, end_line_index] # Snowdar 2019-09-19
    rows = []
    count = 0
    start_index = 0
    end_index = -1

    if chunk is not None:
        start_index = int(chunk[0])
        end_index = int(chunk[1])

    while 1:
        line = fd.readline().decode()
        if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
        if len(line.strip()) == 0 : continue # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            if count >= start_index :
                rows.append(np.array(arr,dtype='float32')) # not last line
            count += 1
            if count > end_index > -1 :
                mat = np.vstack(rows)
                return mat
        else:
            rows.append(np.array(arr[:-1],dtype='float32')) # last line
            mat = np.vstack(rows)
            return mat


def _read_compressed_mat(fd, format, chunk=None):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """
    # chunk:[start_line_index, end_line_index] # Snowdar 2019-09-19
    assert(format == 'CM ') # The formats CM2, CM3 are not supported...

    # Format of header 'struct',
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

    # Read global header,
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                                                 {                     cols                     }{         size                 }
    col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
    col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)

    data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

    mat = np.zeros((cols,rows), dtype='float32')
    p0 = col_headers[:, 0].reshape(-1, 1)
    p25 = col_headers[:, 1].reshape(-1, 1)
    p75 = col_headers[:, 2].reshape(-1, 1)
    p100 = col_headers[:, 3].reshape(-1, 1)
    mask_0_64 = (data <= 64)
    mask_193_255 = (data > 192)
    mask_65_192 = (~(mask_0_64 | mask_193_255))

    mat += (p0    + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
    mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
    mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

    start_index = 0
    end_index = rows - 1

    if chunk is not None:
        start_index = int(chunk[0])
        end_index = int(chunk[1])

    return mat.T[start_index:end_index+1] # transpose! col-major -> row-major,


# Writing,
def write_mat(file_or_fd, m, key=''):
    """ write_mat(f, m, key='')
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
     file_or_fd : filename of opened file descriptor for writing,
     m : the matrix to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

     Example of writing single matrix:
     kaldi_io.write_mat(filename, mat)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,mat in dict.iteritems():
             kaldi_io.write_mat(f, mat, key=key)
    """
    assert(isinstance(m, np.ndarray))
    assert(len(m.shape) == 2), "'m' has to be 2d matrix!"
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if m.dtype == 'float32': fd.write('FM '.encode())
        elif m.dtype == 'float64': fd.write('DM '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
        # Dims,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[0])) # rows
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[1])) # cols
        # Data,
        fd.write(m.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)

def read_post_rxspec(file_):
    """ adaptor to read both 'ark:...' and 'scp:...' inputs of posteriors,
    """
    if file_.startswith("ark:"):
        return read_post_ark(file_)
    elif file_.startswith("scp:"):
        return read_post_scp(file_)
    else:
        print("unsupported intput type: %s" % file_)
        print("it should begint with 'ark:' or 'scp:'")
        sys.exit(1)

def read_post_scp(file_or_fd):
    """ generator(key,post) = read_post_scp(file_or_fd)
     Returns generator of (key,post) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,post in kaldi_io.read_post_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:post for key,post in kaldi_io.read_post_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            post = read_post(rxfile)
            yield key, post
    finally:
        if fd is not file_or_fd : fd.close()

def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
     Returns generator of (key,posterior) tuples, read from ark file.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Iterate the ark:
     for key,post in kaldi_io.read_post_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield key, post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
     Reads single kaldi 'Posterior' in binary format.

     The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
     the outer-vector is usually time axis, inner-vector are the records
     at given time,    and the tuple is composed of an 'index' (integer)
     and a 'float-value'. The 'float-value' can represent a probability
     or any other numeric value.

     Returns vector of vectors of tuples.
    """
    fd = open_or_fd(file_or_fd)
    ans=[]
    binary = fd.read(2).decode(); assert(binary == '\0B'); # binary flag
    assert(fd.read(1).decode() == '\4'); # int-size
    outer_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert(fd.read(1).decode() == '\4'); # int-size
        inner_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of records for frame (or bin)
        data = np.frombuffer(fd.read(inner_vec_size*10), dtype=[('size_idx','int8'),('idx','int32'),('size_post','int8'),('post','float32')], count=inner_vec_size)
        assert(data[0]['size_idx'] == 4)
        assert(data[0]['size_post'] == 4)
        ans.append(data[['idx','post']].tolist())

    if fd is not file_or_fd: fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
     Returns generator of (key,cntime) tuples, read from ark file.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Iterate the ark:
     for key,time in kaldi_io.read_cntime_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield key, cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
     Reads single kaldi 'Confusion Network time info', in binary format:
     C++ type: vector<tuple<float,float> >.
     (begin/end times of bins at the confusion network).

     Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Returns vector of tuples.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode(); assert(binary == '\0B'); # assuming it's binary

    assert(fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    data = np.frombuffer(fd.read(vec_size*10), dtype=[('size_beg','int8'),('t_beg','float32'),('size_end','int8'),('t_end','float32')], count=vec_size)
    assert(data[0]['size_beg'] == 4)
    assert(data[0]['size_end'] == 4)
    ans = data[['t_beg','t_end']].tolist() # Return vector of tuples (t_beg,t_end),

    if fd is not file_or_fd : fd.close()
    return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
    """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
     using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
     - t-beg, t-end is in seconds,
     - assumed 100 frames/second,
    """
    segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
    # Sanity checks,
    assert(len(segs) > 0) # empty segmentation is an error,
    assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
    # Convert time to frame-indexes,
    start = np.rint([100 * rec[2] for rec in segs]).astype(int)
    end = np.rint([100 * rec[3] for rec in segs]).astype(int)
    # Taken from 'read_lab_to_bool_vec', htk.py,
    frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                     np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
    assert np.sum(end-start) == np.sum(frms)
    return frms

