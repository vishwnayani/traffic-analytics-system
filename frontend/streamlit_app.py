import streamlit as st
import requests
import pandas as pd
import os
import cv2

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    layout="wide"
)

st.title("🚦 Traffic Analytics System")

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4"]
)

# ==========================================
# SHOW INPUT VIDEO
# ==========================================

if uploaded_file:

    st.video(uploaded_file)

    if st.button("Run Analytics"):

        with st.spinner(
            "Processing Video..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "video/mp4"
                    )
                }

                API_URL = os.getenv(
                    "API_URL",
                    "http://localhost:8000"
                )

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=3600
                )

                result = response.json()

                st.session_state[
                    "result"
                ] = result

            except Exception as e:

                st.error(
                    f"Backend Error:\n{e}"
                )

                st.stop()

# ==========================================
# DISPLAY RESULTS
# ==========================================

if "result" in st.session_state:

    result = st.session_state[
        "result"
    ]

    summary = result[
        "summary"
    ]

    video_path = result[
        "video_path"
    ]

    csv_path = result[
        "csv_path"
    ]

    vehicle_type_csv = result[
        "vehicle_type_csv"
    ]

    st.success(
        "Analysis Complete"
    )

    # ======================================
    # METRICS
    # ======================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Unique Vehicles",
        summary["unique_vehicles"]
    )

    col2.metric(
        "Avg Speed",
        summary["avg_speed"]
    )

    col3.metric(
        "Max Speed",
        summary["max_speed"]
    )

    col4.metric(
        "Peak Vehicles",
        summary["peak_vehicle_count"]
    )

    # ======================================
    # LANE UTILIZATION
    # ======================================

    st.divider()

    st.subheader(
        "Lane Utilization"
    )

    lane_df = pd.DataFrame({

        "Lane": [
            "Lane 1",
            "Lane 2",
            "Lane 3",
            "Lane 4"
        ],

        "Vehicles": [

            summary["lane_1"],
            summary["lane_2"],
            summary["lane_3"],
            summary["lane_4"]

        ]

    })

    st.bar_chart(
        lane_df.set_index(
            "Lane"
        )
    )

    # ======================================
    # DIRECTION ANALYSIS
    # ======================================

    st.subheader(
        "Direction Analysis"
    )

    direction_df = pd.DataFrame({

        "Direction": [
            "Inbound",
            "Outbound"
        ],

        "Count": [

            summary["inbound"],
            summary["outbound"]

        ]

    })

    st.bar_chart(
        direction_df.set_index(
            "Direction"
        )
    )

    # ======================================
    # VEHICLE TYPES
    # ======================================

    st.subheader(
        "Vehicle Type Distribution"
    )

    if os.path.exists(
        vehicle_type_csv
    ):

        vehicle_df = pd.read_csv(
            vehicle_type_csv
        )

        st.bar_chart(
            vehicle_df.set_index(
                "vehicle_type"
            )
        )

    # ======================================
    # SUMMARY TABLE
    # ======================================

    st.subheader(
        "Summary"
    )

    for k, v in summary.items():

        if isinstance(
            v,
            dict
        ):
            continue

        st.write(
            f"**{k}** : {v}"
        )

    # ======================================
    # DEBUG VIDEO INFO
    # ======================================

    st.divider()

    st.subheader(
        "Annotated Video Preview"
    )

    st.write(
        "Video Path:",
        video_path
    )

    if os.path.exists(video_path):

        cap = cv2.VideoCapture(
            video_path
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            total_frames // 2
        )

        ret, frame = cap.read()

        if ret:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                frame,
                caption="First Frame Preview"
            )

        cap.release()

        with open(
            video_path,
            "rb"
        ) as f:

            st.download_button(

                label=
                "Download Annotated Video",

                data=f.read(),

                file_name=
                os.path.basename(
                    video_path
                ),

                mime=
                "application/octet-stream"

            )

    else:

        st.error(
            "Annotated video not found"
        )

    # ======================================
    # DOWNLOADS
    # ======================================

    st.divider()

    st.subheader(
        "Download Reports"
    )

    if os.path.exists(
        csv_path
    ):

        with open(
            csv_path,
            "rb"
        ) as f:

            st.download_button(

                label=
                "Download Analytics CSV",

                data=f.read(),

                file_name=
                os.path.basename(
                    csv_path
                ),

                mime=
                "text/csv"

            )

    if os.path.exists(
        vehicle_type_csv
    ):

        with open(
            vehicle_type_csv,
            "rb"
        ) as f:

            st.download_button(

                label=
                "Download Vehicle Type CSV",

                data=f.read(),

                file_name=
                os.path.basename(
                    vehicle_type_csv
                ),

                mime=
                "text/csv"

            )