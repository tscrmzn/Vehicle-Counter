# Ultralytics YOLOv8 was used.
# Source:
# @software{yolov8_ultralytics,
#   author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
#   title = {Ultralytics YOLOv8},
#   version = {8.0.0},
#   year = {2023},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
# }

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator

track_history = defaultdict(lambda: [])

model = YOLO("Vehicle.pt")
model.to("cuda")
names = model.model.names
video_path = "Crossroad.mp4"

thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5

if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path "
                            f"'{video_path}' "
                            f"does not exist.")

cap = cv2.VideoCapture(video_path)

Blue =[]
White =[]
Black =[]
Brown =[]
Yellow =[]
Red =[]
Orange =[]
Purple = []

color_red = (0, 0, 255)
color_brown = (0, 76, 153)
color_orange = (0, 150, 255)
color_white = (255, 255, 255)
color_yellow = (0, 255, 255)
color_black = (0, 0, 0)
color_purple = (128, 0, 128)
color_blue = (255, 0, 0)

blue_square = np.array([[532,400],[540,300],[345,339],[338,391]],np.int32)
white_square = np.array([[887,303],[1010,280],[1032,336],[895,428]],np.int32)
black_square = np.array([[631,524],[753,530],[801,631],[750,638]],np.int32)
brown_square = np.array([[565,68],[620,65],[743,202],[643,214]],np.int32)
yellow_square = np.array([[385,404],[562,449],[576,503],[379,445]],np.int32)
red_square = np.array([[543,98],[583,96],[635,217],[567,242]],np.int32)
orange_square = np.array([[827,215],[1000,213],[1000,270],[868,273]],np.int32)
purple_square = np.array([[808,514],[886,480],[867,636],[803,636]],np.int32)

with open("Results.txt","w") as f:
    f.write("Number of Vehicles in Blue Zone: 0\n")
    f.write("Number of Vehicles in White Zone: 0\n")
    f.write("Number of Vehicles in Black Zone: 0\n")
    f.write("Number of Vehicles in Brown Zone: 0\n")
    f.write("Number of Vehicles in Yellow Zone: 0\n")
    f.write("Number of Vehicles in Red Zone: 0\n")
    f.write("Number of Vehicles in Orange Zone: 0\n")
    f.write("Number of Vehicles in Purple Zone: 0\n")

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.resize(frame, (1400, 700))

    if success:
        results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")     
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xywh.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotator = Annotator(frame, line_width=1, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                x1, y1, x2, y2 = (x - (w / 2), y - (h / 2),
                                  x + (w / 2), y + (h / 2))
                label = str(names[cls])
                annotator.box_label([x1, y1, x2, y2],label, color_black)

                # Tracking Lines plot
                track = track_history[track_id]
                track.append((float(box[0]), float(box[1])))
                if len(track) > 15:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False,
                              color=(37, 255, 225), thickness=2)

                # Center circle
                cv2.circle(frame,
                           (int(track[-1][0]), int(track[-1][1])),
                           5, (235, 219, 11), -1)


                if (338 < x and x < 540) and (300 < y and y < 400):
                    if track_id not in Blue:
                        Blue.append(int(track_id))
                elif (887 < x and x < 1032) and (280 < y and y < 428):
                    if track_id not in White:
                        White.append(int(track_id))
                elif (631 < x and x < 801) and (524 < y and y < 638):
                    if track_id not in Black:
                        Black.append(int(track_id))
                elif (565 < x and x < 743) and (65 < y and y < 214):
                    if track_id not in Brown:
                        Brown.append(int(track_id))
                elif (379 < x and x < 576) and (404 < y and y < 503):
                    if track_id not in Yellow:
                        Yellow.append(int(track_id))
                elif (543 < x and x < 635) and (96 < y and y < 242):
                    if track_id not in Red:
                        Red.append(int(track_id))
                elif (827 < x and x < 1000) and (213 < y and y < 270):
                    if track_id not in Orange:
                        Orange.append(int(track_id))
                elif (803 < x and x < 886) and (480 < y and y < 636):
                    if track_id not in Purple:
                        Purple.append(int(track_id))
            
            with open("Results.txt","w") as f:
                f.write(f"Number of Vehicles in Blue Zone: {len(Blue)}\n")
                f.write(f"Number of Vehicles in White Zone: {len(White)}\n")
                f.write(f"Number of Vehicles in Brown Zone: {len(Brown)}\n")
                f.write(f"Number of Vehicles in Black Zone: {len(Black)}\n")
                f.write(f"Number of Vehicles in Yellow Zone: {len(Yellow)}\n")
                f.write(f"Number of Vehicles in Red Zone: {len(Red)}\n")
                f.write(f"Number of Vehicles in Orange Zone: {len(Orange)}\n")
                f.write(f"Number of Vehicles in Purple Zone: {len(Purple)}\n")

    cv2.polylines(frame, [blue_square], isClosed=True, color=color_blue, thickness=thickness)
    cv2.polylines(frame, [white_square], isClosed=True, color=color_white, thickness=thickness)
    cv2.polylines(frame, [brown_square], isClosed=True, color=color_brown, thickness=thickness)
    cv2.polylines(frame, [black_square], isClosed=True, color=color_black, thickness=thickness)
    cv2.polylines(frame, [yellow_square], isClosed=True, color=color_yellow, thickness=thickness)
    cv2.polylines(frame, [red_square], isClosed=True, color=color_red, thickness=thickness)
    cv2.polylines(frame, [orange_square], isClosed=True, color=color_orange, thickness=thickness)
    cv2.polylines(frame, [purple_square], isClosed=True, color=color_purple, thickness=thickness)

    blue_counter_text = "Blue: {}".format(str(len(Blue)))
    white_counter_text = "White: {}".format(str(len(White)))
    brown_counter_text = "Brown: {}".format(str(len(Brown)))
    black_counter_text = "Black: {}".format(str(len(Black)))
    yellow_counter_text = "Yellow: {}".format(str(len(Yellow)))
    red_counter_text = "Red: {}".format(str(len(Red)))
    orange_counter_text = "Orange: {}".format(str(len(Orange)))
    purple_counter_text = "Purple: {}".format(str(len(Purple)))


    cv2.putText(frame, blue_counter_text, (10, 20), font, font_scale,  color_white, thickness)
    cv2.putText(frame, white_counter_text, (10, 40), font, font_scale, color_white, thickness)
    cv2.putText(frame, brown_counter_text, (10, 60), font, font_scale, color_white, thickness)
    cv2.putText(frame, black_counter_text, (10, 80), font, font_scale, color_white, thickness)
    cv2.putText(frame, yellow_counter_text, (10, 100), font, font_scale, color_white, thickness)
    cv2.putText(frame, red_counter_text, (10, 120), font, font_scale, color_white, thickness)
    cv2.putText(frame, orange_counter_text, (10, 140), font, font_scale, color_white, thickness)
    cv2.putText(frame, purple_counter_text, (10, 160), font, font_scale, color_white, thickness)

    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()