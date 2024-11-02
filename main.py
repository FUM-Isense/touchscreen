import math
import os
import time
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import re
import jellyfish

from TextToSpeechPlayer import TextToSpeechPlayer
from apriltag_detector import AprilTag

names = [
    "chicken","apple", "baconandeggs", "icecream", 
    "hamburger", "croissant", "cake", "lasagna",
    "sausages", "fruitsalad", "steak", "spaghetti",
    "sandwich", "broccoli", "mashedpotatoes","tomatosoup"
         ]

def __find_most_similar(text):
    max_sim = float('-inf')  # Start with a large number
    most_similar_name = None
    
    for name in names:
        sim = __similar(name.lower(),text.lower())
        
        if sim > max_sim:
            max_sim = sim
            most_similar_name = name
    
    return most_similar_name,max_sim

def __similar(a, b):
    return jellyfish.jaro_similarity(a, b)

def __clean_and_split_text(text):
    words = re.split(r'[^a-zA-Z]+', text)

    words = [word for word in words if word]
    word = ''.join(words)

    return word

def mask_button_area(image):
    lower_red = np.array([20, 50, 100])
    upper_red = np.array([90, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    button_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    button_result = cv2.bitwise_and(hsv_image, hsv_image, mask=button_mask)

    second_image = button_result.copy()
    edges = cv2.Canny(second_image, 10, 100, apertureSize=3)
    kernel = np.ones((30, 30), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    sizes = stats[:, -1]

    largest_label = np.argmax(sizes[1:]) + 1

    merged_component = np.zeros_like(dilated)
    merged_component[labels == largest_label] = 255
    x, y, w, h= cv2.boundingRect(merged_component)

    real_image = image[y:y+h,x:x+w]
    result = button_result[y:y+h,x:x+w]

    return result, real_image, (y,y+h,x,x+w)

def detect_button_area(img, area_th = 2000):
    edges = cv2.Canny(img, 20, 100, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours_merged, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_width = 0
    total_height = 0
    num_contours = 0
    boundings =[]
    
    y_centers = []
    x_centers = []
            
    for contour in contours_merged:
        x, y, w, h = cv2.boundingRect(contour)  # Extract the bounding box
        total_width += w
        total_height += h
        num_contours += 1
        area = cv2.contourArea(contour)
        if area > area_th:
            y_center = (y + y+h) // 2
            x_center = (x + x+w) // 2
            
            y_centers.append(y_center)
            x_centers.append(x_center)
            boundings.append((x, y, w, h))

    return boundings, x_centers, y_centers

def __create_boundaries(centers, img, axis='y'):
    centers = sorted(centers)
    diffs = np.diff(centers)

    threshold = np.mean(diffs)
    boundaries = [centers[0]]
    min_gap = 1000
    
    for i, diff in enumerate(diffs):
        if diff > threshold:
            center = centers[i + 1]
            min_gap = min(min_gap, diff)
            boundaries.append(center)  
    
    diff_bounds = np.diff(boundaries)
    for i, bnd_diff in enumerate(diff_bounds):
        if bnd_diff > min_gap * 2:
            boundaries.insert(i, boundaries[i] + min_gap)

    # if draw_debug:
    #     for bound in boundaries:
    #         if axis == 'y':
    #             cv2.line(img, (0, bound), (img.shape[1], bound), (100, 0, 0), 2)
    #         else:
    #             cv2.line(img, (bound, 0), (bound, img.shape[0]), (100, 0, 0), 2)
                
    return boundaries

def __assign_index(value, row_boundaries):
    lowest_bound = 10000
    lowest_bound_i = -1
    
    for i, boundary in enumerate(row_boundaries):
        diff = abs(value - boundary)
        if diff < lowest_bound:
            lowest_bound_i = i
            lowest_bound = diff
        
    return lowest_bound_i

def find_button_info(image, item_text="fruitsalad",ocr=True,score_th = 0.75, draw = True):
    result, real_image, (y1,y2,x1,x2) = mask_button_area(image)
    boundings, x_centers, y_centers = detect_button_area(result)
    
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    annotated = cv2.filter2D(real_image, -1, kernel)
    found = False
    buttons_info = []
    
    row_boundaries = __create_boundaries(y_centers, annotated, axis='y')
    column_boundaries = __create_boundaries(x_centers, annotated, axis='x')
    
    text_scores = []
    
    x_found = -1
    y_found = -1

    for (x, y, w, h) in boundings:
        button = annotated[y:y+h, x:x+w]

        area = w * h
        y_center = (y + y+h) // 2
        x_center = (x + x+w) // 2
        
        row = __assign_index(y_center, row_boundaries) + 1    
        column = __assign_index(x_center, column_boundaries) + 1   
        
        if ocr:
            results = reader.readtext(button)
            final_text = []
            for bbox1, text1, prob1 in results:
                text = __clean_and_split_text(text1)
                final_text.append(text)
                
            final_text = ''.join(final_text).lower()
            similar_text, score = __find_most_similar(final_text)
        else:
            similar_text = ""
            score = area
        
        button_info = {
            "bounding":(x, y, w, h),
            "text" : similar_text,
            "text_score":score,
            "column":column,
            "row":row 
        }
        
        text_scores.append(score)

        buttons_info.append(button_info)

        if draw:
            if similar_text == item_text and score > score_th:
                found = True
                x_found = row
                y_found = column
                # print(f"Found on row{row} , col{column}")
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.putText(annotated, f"{row},{column}", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)  # Blue text
            cv2.putText(annotated, f"{score:.2f}", (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 0), 2)  # Blue text
            if score > score_th:
                cv2.putText(annotated, f"{similar_text}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # Blue text
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
            else:
                cv2.putText(annotated, f"{similar_text}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # Blue text
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return found, buttons_info, annotated, row_boundaries, column_boundaries, x_found, y_found
            
def calculate_center(bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_nearest_button(position, buttons_info):
    nearest_buttons = []

    for button in buttons_info:
        if button['text'] is not None:
            center = calculate_center(button['bounding'])
            distance = calculate_distance(position, center)
            nearest_buttons.append((button,distance))

    nearest_buttons_sorted = sorted(nearest_buttons, key=lambda x: x[1])
    hover_button = nearest_buttons_sorted[0]

    return hover_button


def predict_hovered_button(position, buttons_info):
    hover_button = find_nearest_button(position, buttons_info)

    return hover_button[0]


last_time_say_lost = 0

def Say(text, reset_time=False):
    global last_time_say_lost

    if reset_time:
        last_time_say_lost = 0

    if time.time() - last_time_say_lost >= 2:
        last_time_say_lost = time.time()
        speech.say(text=text)


reader = easyocr.Reader(['en'])

video = False
video_path = "20241021_205327.mp4"

if video:
    cap = cv2.VideoCapture(video_path)
else:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("234322308671")

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()
    color_sensor = device.query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 125)

found = False
buttons_info = []
apriltag = AprilTag()
speech = TextToSpeechPlayer()
# while True:
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     image = np.asanyarray(color_frame.get_data())

#     (x, y) = apriltag.tag_pos(image, True)

#     print(x,y)
#     cv2.imshow("tag", image)
#     cv2.waitKey(1)

proccess_frame = 5
counter = 0
Say("Start")

while (not video) or (video and cap.isOpened()):
    
    if video:
        ret, image = cap.read()
    else:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    counter += 1
    # image = cv2.resize(image, (1280,720))
    if counter % proccess_frame == 0:
        try:
            found, buttons_info, annotated, row_boundaries, column_boundaries, x_found, y_found = find_button_info(image=image,item_text="fruitsalad",ocr=True,draw=True)
            if found and len(buttons_info) == 16:
                cv2.destroyAllWindows()
                cv2.imshow("Result", annotated)
                cv2.waitKey(1)
                Say("Button Found")
                print("Button Found")
                break
            
            cv2.imshow("Frame", annotated)
            cv2.waitKey(1)
        except:
            Say("Not Found")
            print("Error")
            cv2.destroyAllWindows()
       

proccess_frame = 5
counter = 0
while (not video) or (video and cap.isOpened()):
    
    if video:
        ret, img_hand = cap.read()
    else:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img_hand = np.asanyarray(color_frame.get_data())
        # img_hand = cv2.rotate(img_hand, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # img_hand = cv2.resize(img_hand, (1280,720))
    counter += 1
    
    if counter % proccess_frame == 0:
        try:
            result, real_image, (y1,y2,x1,x2) = mask_button_area(img_hand)
            boundings, x_centers, y_centers = detect_button_area(result)

            height, width = annotated.shape[:2]

            resized_image = cv2.resize(real_image, (width, height))

            kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
            resized_image = cv2.filter2D(resized_image, -1, kernel)
    
            if len(buttons_info) == 16 and len(row_boundaries) == 4 and len(column_boundaries) == 4:
                hovered_buttons = np.zeros((4,4),dtype=np.uint8)
                
                (x, y) = apriltag.tag_pos(resized_image, True)

                if x != -1 and y != -1:
                    hover_buuton = predict_hovered_button((x, y), buttons_info)
                    row = hover_buuton["row"]
                    col = hover_buuton['column']
                    hovered_buttons[row-1][col-1] = 1
                    
                    col_diff = y_found - col
                    row_diff = x_found - row
                    
                    if row_diff < 0:
                        Say("Up")
                        print("Up")
                    elif row_diff > 0:
                        Say("Down")
                        print("Down")
                    elif col_diff < 0:
                        Say("Left")
                        print("Left")
                    elif col_diff > 0:
                        Say("Right")
                        print("Right")
                    else:
                        Say("Correct")
                        print("Correct")
                        
                    
                    print(f"Target On {x_found} and {y_found} | Finger on {row} , {col}", end='\r')
                else:
                    Say("Tag not detected")
                    print("Tag not detected")
                    
                
            scaled_array = cv2.resize(hovered_buttons, (400, 400), interpolation=cv2.INTER_NEAREST)
            hover_image = cv2.cvtColor(scaled_array * 255, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow('4x4 Black and White Squares', hover_image)
            cv2.imshow("Result Hand", resized_image)
            cv2.waitKey(1)
            
        except:
            cv2.imshow("Result Hand", img_hand)
            cv2.waitKey(1)