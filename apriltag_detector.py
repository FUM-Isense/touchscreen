import apriltag
import cv2

class AprilTag:
    def __init__(self):
        options = apriltag.DetectorOptions(families="tag25h9")  # Use the family that fits your needs
        self.detector = apriltag.Detector(options)
        
        pass


    def tag_pos(self, image, draw = False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        tags = self.detector.detect(gray)

        for tag in tags:
            # Draw a box around the detected tag
            for i in range(4):
                pt1 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                pt2 = (int(tag.corners[(i+1) % 4][0]), int(tag.corners[(i+1) % 4][1]))

            # Draw the tag's center
            center = (int(tag.center[0]), int(tag.center[1]))
            if draw:
                cv2.circle(image, center, 5, (0, 0, 255), -1)

            return center
        return -1,-1

        