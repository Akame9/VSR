import subprocess
from glob import glob
import os
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--speaker", help="Root folder of Speaker", required=True)
args = parser.parse_args()
template = 'ffmpeg -i {} -ab 1k -ac 1 -ar 44100 -vn {}'

def main(args):
    print('Hello')
    filelist = glob(path.join(args.speaker,'*.mpg'))
    for f in filelist:
        wavname = f[11:-4]
        wavpath = path.join(args.speaker,'{}.wav'.format(wavname))
        command = template.format(f,wavpath)
        subprocess.call(command, shell=True)

if __name__ == '__main__':
	main(args)

