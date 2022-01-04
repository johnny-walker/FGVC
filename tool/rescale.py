import cv2
import os
import argparse
import glob

class VideoWriter(object):
    """
    Video writer which handles video recording overhead
    Usage:
        object creation: provide path to write
        write:
        release:
    """
    def __init__(self, video_file, fps=25, scale=1.0):
        """

        :param video_file: path to write video. Perform nothing in case of None
        :param fps: frame per second
        :param scale: resize scale
        """
        self.video_file = video_file
        self.fps = fps
        self.writer = None
        self.scale = scale

    def write(self, frame):
        """

        :param frame: numpy array, (H, W, 3), BGR, frame to write
        :return:
        """
        h, w = frame.shape[:2]
        h_rsz, w_rsz = int(h * self.scale), int(w * self.scale)
        frame = cv2.resize(frame, (w_rsz, h_rsz))
        if self.writer is None:
            video_dir = os.path.dirname(os.path.realpath(self.video_file))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.video_file, fourcc, self.fps,
                                          tuple(frame.shape[1::-1]))
        self.writer.write(frame)

    def release(self):
        """
        Manually release
        :return:
        """
        if self.writer is None:
            return
        self.writer.release()

    def __del__(self):
        self.release()


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

        width = int(args.W_resize)
        height = int(args.H_resize)

        # dsize
        dsize = (width, height)
        print
        # resize image
        output = cv2.resize(src, dsize)
        name = os.path.split(filename)[-1]
        outfile = os.path.join(args.outroot, name)
        print (outfile, '\r', end='')
        cv2.imwrite(outfile, output) 

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../data/bumps_mask', help="dataset for evaluation")
parser.add_argument('--outroot', default='../data/bumps_mask', help="output directory")
parser.add_argument('--H_resize', dest='H_resize', default=720, type=int, help='H sesize')
parser.add_argument('--W_resize', dest='W_resize', default=1280,  type=int, help='W resize')

args = parser.parse_args()
main(args)