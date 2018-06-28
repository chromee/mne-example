from __future__ import print_function
import socket
from contextlib import closing
from ast import literal_eval
import struct

def main():
  host = '127.0.0.1'
  port = 4000
  bufsize = 8192

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  with closing(sock):
    sock.bind((host, port))
    while True:
      buf = sock.recv(bufsize)
      print(struct.unpack('<1d', buf)[0])
  return

if __name__ == '__main__':
  main()