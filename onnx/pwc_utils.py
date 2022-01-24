###
# Sample mgmt
###
import numpy as np
import cv2

def resize_to_fit(im, input_size):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x is the np array, not the TF tensor.
    Args:
        x: input samples in list (H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (H,W,3) format
    """
    fit_ratio = input_size[0] / input_size[1]
    im_ratio = im.shape[0] / im.shape[1]
    if fit_ratio > im_ratio:
        # scale by width
        width = input_size[1]
        height = round(width * im_ratio)
    else:
        # scale by height
        height = input_size[0]
        width = round(height / im_ratio)

    x_adapt = cv2.resize(im, (width, height))

    pad_h, pad_w = input_size[0]-height, input_size[1]-width
    if pad_h != 0 or pad_w != 0:
        padding = [(0, pad_h), (0, pad_w), (0, 0)]
        x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, (height, width)

def unpad_and_upscale(flow, unpad_info, size):
    if unpad_info is not None:
        pred_flow = flow[0:unpad_info[0], 0:unpad_info[1], :]
        up_flow = cv2.resize(pred_flow, (size[1], size[0])) # (w,h)
        return up_flow


def flow_to_img(flow, normalize=True, info=None, flow_mag_max=None):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
        info: Text to superimpose on image (typically, the epe for the predicted flow)
        flow_mag_max: Max flow to map to 255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
        flow_avg: optionally, also return average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return img
