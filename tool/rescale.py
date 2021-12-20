import cv2
import os
import argparse
import glob

def main(args):
    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))
    # create path.
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)

    for filename in sorted(filename_list):
        src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #percent by which the image is resized
        scale_percent = 50

        #calculate the 50 percent of original dimensions
        width = int(src.shape[1] * scale_percent / 100)
        height = int(src.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)
        print
        # resize image
        output = cv2.resize(src, dsize)
        name = os.path.split(filename)[-1]
        outfile = os.path.join(args.outroot, name)
        print (outfile)#, '\r', end='')
        cv2.imwrite(outfile, output) 

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../data/blackswan', help="dataset for evaluation")
parser.add_argument('--outroot', default='../data/blackswan_scale', help="output directory")

args = parser.parse_args()
main(args)