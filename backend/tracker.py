import os
import cv2
import time
import numpy as np
import pandas as pd

import supervision as sv

from collections import defaultdict
from collections import deque

from backend.config import OUTPUT_DIR


def process_video(
    video_path,
    model
):

    # ==========================================
    # VIDEO INFO
    # ==========================================

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_video_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_annotated.avi"
    )

    output_csv_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_analytics.csv"
    )

    # ==========================================
    # VIDEO READER
    # ==========================================

    cap = cv2.VideoCapture(
        video_path
    )
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps_video = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps_video <= 0:
        fps_video = 20

    # ==========================================
    # VIDEO WRITER
    # ==========================================

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps_video,
        (width, height)
    )

    print(
        "Writer Opened:",
        writer.isOpened()
    )

    if not writer.isOpened():

        raise RuntimeError(
            "VideoWriter failed to initialize"
        )

    # ==========================================
    # BYTE TRACK
    # ==========================================

    tracker = sv.ByteTrack()

    # ==========================================
    # ANNOTATORS
    # ==========================================

    box_annotator = sv.BoxAnnotator()

    label_annotator = sv.LabelAnnotator()

    trace_annotator = sv.TraceAnnotator()

    # ==========================================
    # DYNAMIC LINE COUNTER
    # ==========================================

    LINE_START = sv.Point(
        int(width * 0.10),
        int(height * 0.50)
    )

    LINE_END = sv.Point(
        int(width * 0.90),
        int(height * 0.50)
    )

    line_zone = sv.LineZone(
        start=LINE_START,
        end=LINE_END
    )

    line_zone_annotator = (
        sv.LineZoneAnnotator()
    )

    # ==========================================
    # DYNAMIC LANES
    # ==========================================

    lane_1_end = int(
        height * 0.25
    )

    lane_2_end = int(
        height * 0.50
    )

    lane_3_end = int(
        height * 0.75
    )

    # ==========================================
    # TRACKING STRUCTURES
    # ==========================================

    fps = fps_video

    track_history = defaultdict(
        lambda: deque(maxlen=30)
    )

    #speed_dict = {}

    total_vehicles = set()

    frame_vehicle_counts = []

    speed_values = []

    vehicle_type_counts = defaultdict(
        int
    )

    vehicle_type_ids = defaultdict(
        set
    )

    lane_vehicle_ids = {

        1: set(),

        2: set(),

        3: set(),

        4: set()

    }

    vehicle_direction = {}
    
    counted_inbound = set()
    counted_outbound = set()

    inbound_count = 0

    outbound_count = 0

    peak_vehicle_count = 0

    peak_frame = 0

    frame_idx = 0

    # ==========================================
    # MAIN LOOP
    # ==========================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        start_time = time.time()

        # ======================================
        # YOLO INFERENCE
        # ======================================

        results = model.predict(

            source=frame,

            conf=0.3,

            imgsz=640,

            verbose=False

        )

        result = results[0]

        # ======================================
        # DETECTIONS
        # ======================================

        detections = (
            sv.Detections.from_ultralytics(
                result
            )
        )

        # ======================================
        # VEHICLE CLASSES ONLY
        # ======================================

        vehicle_class_ids = [

            2,  # car

            3,  # motorcycle

            5,  # bus

            7   # truck

        ]

        mask = np.isin(

            detections.class_id,

            vehicle_class_ids

        )

        detections = detections[
            mask
        ]

        # ======================================
        # BYTE TRACK
        # ======================================

        detections = (
            tracker.update_with_detections(
                detections
            )
        )

        labels = []

        # ======================================
        # TRACK PROCESSING
        # ======================================

        for i in range(
            len(detections)
        ):
            tracker_id = detections.tracker_id[i]

            if tracker_id is None:
                continue

            tracker_id = int(
                tracker_id
            )

            class_id = detections.class_id[i]

            total_vehicles.add(
                tracker_id
            )

            # ==================================
            # BOUNDING BOX
            # ==================================

            x1, y1, x2, y2 = (
                detections.xyxy[i]
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            # ==================================
            # DYNAMIC LANE ASSIGNMENT
            # ==================================

            if center_y < lane_1_end:

                lane = 1

            elif center_y < lane_2_end:

                lane = 2

            elif center_y < lane_3_end:

                lane = 3

            else:

                lane = 4

            lane_vehicle_ids[
                lane
            ].add(
                tracker_id
            )

            # ==================================
            # TRACK HISTORY
            # ==================================

            track_history[
                tracker_id
            ].append(

                (
                    center_x,
                    center_y
                )

            )

            # ==================================
            # DIRECTION ANALYSIS
            # ==================================

            if tracker_id not in vehicle_direction:

                vehicle_direction[
                    tracker_id
                ] = center_x

            else:

                start_x = (
                    vehicle_direction[
                        tracker_id
                    ]
                )

                displacement = (
                    center_x -
                    start_x
                )

                if (displacement > 150 and tracker_id not in counted_inbound):

                    inbound_count += 1

                    counted_inbound.add(
                        tracker_id
                    )

                    vehicle_direction[
                        tracker_id
                    ] = center_x

                elif (displacement < -150 and tracker_id not in counted_outbound):

                    outbound_count += 1

                    counted_outbound.add(
                        tracker_id
                    )

                    vehicle_direction[
                        tracker_id
                    ] = center_x

            # ==================================
            # SPEED ESTIMATION
            # ==================================

            speed = 0

            if len(
                track_history[
                    tracker_id
                ]
            ) >= 2:

                prev_x, prev_y = (
                    track_history[
                        tracker_id
                    ][-2]
                )

                distance = np.sqrt(

                    (
                        center_x -
                        prev_x
                    ) ** 2

                    +

                    (
                        center_y -
                        prev_y
                    ) ** 2

                )

                speed = (
                    distance *
                    fps *
                    0.1
                )

                speed_values.append(
                    speed
                )

            # ==================================
            # VEHICLE TYPE ANALYTICS
            # ==================================

            class_name = (
                model.names[
                    class_id
                ]
            )

            vehicle_type_counts[
                class_name
            ] += 1

            vehicle_type_ids[
                class_name
            ].add(
                tracker_id
            )

            # ==================================
            # LABEL
            # ==================================

            label = (

                f"ID:{tracker_id} "

                f"{class_name} "

                f"{speed:.1f} km/h"

            )

            labels.append(
                label
            )
        # ======================================
        # LINE COUNTING
        # ======================================

        line_zone.trigger(
            detections
        )

        # ======================================
        # TRACE
        # ======================================

        frame = trace_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # ======================================
        # BOXES
        # ======================================

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # ======================================
        # LABELS
        # ======================================

        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # ======================================
        # LINE ANNOTATION
        # ======================================

        frame = line_zone_annotator.annotate(
            frame=frame,
            line_counter=line_zone
        )

        # ======================================
        # ANALYTICS
        # ======================================

        current_vehicle_count = len(
            detections
        )

        frame_vehicle_counts.append(
            current_vehicle_count
        )

        if (
            current_vehicle_count >
            peak_vehicle_count
        ):

            peak_vehicle_count = (
                current_vehicle_count
            )

            peak_frame = frame_idx

        congestion_status = "LOW"

        if current_vehicle_count > 20:

            congestion_status = "HIGH"

        elif current_vehicle_count > 10:

            congestion_status = "MEDIUM"

        end_time = time.time()

        processing_time = max(
            end_time - start_time,
            1e-6
        )

        fps_display = 1 / processing_time

        # ======================================
        # DISPLAY TEXT
        # ======================================

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Vehicles: {current_vehicle_count}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Unique Vehicles: {len(total_vehicles)}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            f"Congestion: {congestion_status}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            3
        )

        # ======================================
        # WRITE FRAME TO OUTPUT VIDEO
        # ======================================

        writer.write(
            frame
        )

        frame_idx += 1
    # ==========================================
    # CLEANUP
    # ==========================================

    cap.release()

    writer.release()
    print(
    "Annotated video exists:",
    os.path.exists(output_video_path)
)

    if os.path.exists(output_video_path):

        print(
            "Video size:",
            os.path.getsize(output_video_path)
        )

    # ==========================================
    # SAVE ANALYTICS CSV
    # ==========================================

    analytics_df = pd.DataFrame({

        "frame":
        range(
            len(frame_vehicle_counts)
        ),

        "vehicle_count":
        frame_vehicle_counts

    })

    analytics_df.to_csv(

        output_csv_path,

        index=False

    )

    # ==========================================
    # VEHICLE TYPE DATAFRAME
    # ==========================================

    vehicle_df = pd.DataFrame({

        "vehicle_type":
        list(
            vehicle_type_ids.keys()
        ),

        "count":
        [

            len(v)

            for v in
            vehicle_type_ids.values()

        ]

    })

    vehicle_df_path = os.path.join(

        OUTPUT_DIR,

        f"{video_name}_vehicle_types.csv"

    )

    vehicle_df.to_csv(

        vehicle_df_path,

        index=False

    )

    # ==========================================
    # SPEED SUMMARY
    # ==========================================

    avg_speed = (

        np.mean(
            speed_values
        )

        if len(speed_values) > 0

        else 0

    )

    max_speed = (

        np.max(
            speed_values
        )

        if len(speed_values) > 0

        else 0

    )

    # ==========================================
    # VEHICLE COUNT SUMMARY
    # ==========================================

    avg_vehicle_count = (

        np.mean(
            frame_vehicle_counts
        )

        if len(frame_vehicle_counts) > 0

        else 0

    )

    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    summary = {

        "video_name":
        video_name,

        "unique_vehicles":
        len(total_vehicles),

        "avg_speed":
        round(
            avg_speed,
            2
        ),

        "max_speed":
        round(
            max_speed,
            2
        ),

        "avg_vehicle_count":
        round(
            avg_vehicle_count,
            2
        ),

        "peak_vehicle_count":
        int(
            peak_vehicle_count
        ),

        "peak_frame":
        int(
            peak_frame
        ),

        "lane_1":
        len(
            lane_vehicle_ids[1]
        ),

        "lane_2":
        len(
            lane_vehicle_ids[2]
        ),

        "lane_3":
        len(
            lane_vehicle_ids[3]
        ),

        "lane_4":
        len(
            lane_vehicle_ids[4]
        ),

        "inbound":
        int(
            inbound_count
        ),

        "outbound":
        int(
            outbound_count
        ),

        "vehicle_types": {

            k: len(v)

            for k, v in

            vehicle_type_ids.items()

        }

    }

    # ==========================================
    # RETURN TO FASTAPI
    # ==========================================

    return {

        "summary":
        summary,

        "video_path":
        output_video_path,

        "csv_path":
        output_csv_path,

        "vehicle_type_csv":
        vehicle_df_path

    }