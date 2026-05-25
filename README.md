# GSAC-Football-Player-Detection-2026

The model processing of interest is in process_video_tracked.py.
The code runs the model through the roboflow api. 

https://app.roboflow.com/gsac-soccer-project-2026/football-players-detection-3zvbc-nvuzb/models
project_id = "football-players-detection-3zvbc-nvuzb"
model = 3 is the current model trained on all the games. 

you will to get an api key from roboflow and add it to the .env to run the code.

currently, you have to change the output and input paths manually in the code before running "process_video_tracked.py".
process_video(source_video_path="inputs/example.mp4", target_video_path="outputs/example.mp4")
