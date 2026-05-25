import os
import cv2
import supervision as sv
from dotenv import load_dotenv
from roboflow import Roboflow


load_dotenv()

project_id = "football-players-detection-3zvbc-nvuzb"
version = "2"
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

    
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in sv.get_video_frames_generator(source_path=source_video_path):
            
            
            prediction = model.predict(frame, confidence=40, overlap=30).json()
            
            
            detections = sv.Detections.from_inference(prediction)
            
            
            labels = [
                f"{class_name} {confidence:0.2f}"
                for class_name, confidence in zip(detections.data['class_name'], detections.confidence)
            ]
            
            
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            
            sink.write_frame(frame=annotated_frame)

    print(f"Finished processing. Output saved to {target_video_path}")

if __name__ == "__main__":

    process_video(source_video_path="sacstate_test.mp4", target_video_path="model2_sac_output.mp4")
    
