import os
import cv2
import supervision as sv
import trackers
from dotenv import load_dotenv
from roboflow import Roboflow


load_dotenv()

project_id = "football-players-detection-3zvbc-nvuzb"
version = "3"
api_key = os.getenv("api_key")

def process_video(source_video_path: str, target_video_path: str):
    print(f"Processing video: {source_video_path}")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("gsac-soccer-project-2026").project(project_id)
    model = project.version(int(version)).model
    
    
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    print(f"Video Info: {video_info}")

    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 1. Initialize the BoT-SORT tracker and DetectionsSmoother
    tracker = trackers.BoTSORTTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=video_info.fps
    )
    smoother = sv.DetectionsSmoother()
    
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in sv.get_video_frames_generator(source_path=source_video_path):
            
            
            prediction = model.predict(frame, confidence=40, overlap=30).json()
            
            
            detections = sv.Detections.from_inference(prediction)
            
            # 2. Filter out the ball detections
            # Only track player, referee, and goalkeeper
            detections = detections[detections.data['class_name'] != 'ball']
            
            # 3. Plug the detections into the tracker
            # This assigns a unique 'tracker_id' to every player
            detections = tracker.update(detections=detections, frame=frame)
            
            # Filter out unconfirmed tracks (tracker_id = -1)
            detections = detections[detections.tracker_id != -1]
            
            # 4. Smooth the detections to reduce flickering
            detections = smoother.update_with_detections(detections=detections)
            
            class_names = detections.data.get('class_name', [])
            tracker_ids = detections.tracker_id if detections.tracker_id is not None else []

            labels = [
                f"#{tracker_id} {class_name}"
                for tracker_id, class_name 
                in zip(tracker_ids, class_names)
            ]
            
            
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            
            sink.write_frame(frame=annotated_frame)

    print(f"Finished processing. Output saved to {target_video_path}")

if __name__ == "__main__":

    process_video(source_video_path="inputs/Davisclip3.mp4", target_video_path="outputs/model3_davisclip3.mp4")
