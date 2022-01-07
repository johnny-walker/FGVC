import cv2

class Pixels():
    def __init__(self):
        None

    def cropping(self, frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def validBBox(self, bbox):
        if bbox is None or (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            return True
        return False

    def getBBox(self, rectPts, imgWidth, imgHeight):
        left = imgWidth
        top = imgHeight
        right =  bottom = 0
        for pt in rectPts:
            left   = pt[0] if pt[0]<left   else left
            top    = pt[1] if pt[1]<top    else top
            right  = pt[0] if pt[0]>right  else right
            bottom = pt[1] if pt[1]>bottom else bottom
        return (left, top, right, bottom)

    def ptInBBox(self, point, bbox):
        #print(point)
        return True if (bbox[0] <= point[0] <= bbox[2]) and (bbox[1] <= point[1] <= bbox[3]) else False

    # ignore sqrt to save computing time, it doesn't matter, just want to select the minimun value
    def norm2Distance(self, pt1, pt2):
        dx = pt1[0] - pt2[0]
        dy = pt1[1] - pt2[1]
        return dx*dx + dy*dy   

    # draw rect on frame
    def drawLines(self, frame, rectPts):
        color = (128, 128, 128)
        points = tuple(tuple(map(int, pt)) for pt in rectPts)   # convert to int
        frame = cv2.line(frame, points[0], points[1], color, 1)
        frame = cv2.line(frame, points[1], points[2], color, 1)
        frame = cv2.line(frame, points[2], points[3], color, 1)
        frame = cv2.line(frame, points[3], points[0], color, 1)

    # draw point on frame
    def drawCircles(self, frame, keypoints, color=(0, 255, 255)):
        radius = 3
        for point in keypoints:
            cv2.circle(frame, tuple(map(int, point.pt)), radius, color, 1)
