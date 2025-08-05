import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import gradio as gr
import json  # Import json module
from src import model
from src import util
from src.body import Body
from src.hand import Hand

# This function will generate and save the pose data as JSON
def save_json(candidate, subset, json_file_path='./pose_data.json'):
    pose_data = {
        'candidate': candidate.tolist(),
        'subset': subset.tolist()
    }
    with open(json_file_path, 'w') as json_file:
        json.dump(pose_data, json_file)
    return json_file_path

def pose_estimation(test_image):
    bgr_image_path = './test.png'
    with open(bgr_image_path, 'wb') as bgr_file:
        bgr_file.write(test_image)
    # Load the estimation models
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    oriImg = cv2.imread(bgr_image_path)  # B,G,R order

    # Perform pose estimation
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    out_image_path = './out.jpg'
    plt.savefig(out_image_path)

    # Save JSON data and return its path
    json_file_path = save_json(candidate, subset)

    return out_image_path, json_file_path

# Convert the image path to bytes for Gradio to display
def convert_image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Pose Estimation")
    with gr.Row():
        image = gr.File(label="Upload Image", type="binary")
        output_image = gr.Image(label="Estimation Result")
        output_json = gr.File(label="Download Pose Data as JSON", type="filepath")  # Add JSON output
    submit_button = gr.Button("Start Estimation")

    # Run pose estimation and display results when the button is clicked
    submit_button.click(
        pose_estimation,
        inputs=[image],
        outputs=[output_image, output_json]  # Update outputs
    )

    # Clear the results
    clear_button = gr.Button("Clear")
    def clear_outputs():
        output_image.clear()
        output_json.clear()  # Clear JSON output as well
    clear_button.click(
        clear_outputs,
        inputs=[],
        outputs=[output_image, output_json]  # Update outputs
    )

if __name__ == "__main__":
    demo.launch(debug=True)