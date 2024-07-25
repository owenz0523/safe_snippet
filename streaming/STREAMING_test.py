import os
from STREAMINGpt2 import file_streaming
 
def test_file_streaming():
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_file_path = os.path.join(current_dir, "crash.mp4")
 
    # Check if the file exists
    if not os.path.isfile(video_file_path):
        print(f"Error: The video file '{video_file_path}' does not exist.")
        return
 
    print(f"Using video file path: {video_file_path}")
    capture = file_streaming(video_file_path, frame_interval=5)
    assert capture is not None, "Failed to open the video file."
 
if __name__ == "__main__":
    test_file_streaming()