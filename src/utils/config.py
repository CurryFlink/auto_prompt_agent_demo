TASK_DESCRIPTION = """
Create an autonomous driving scene image with the following specific annotations:
1. Lane Lines: Draw solid yellow lines along the edges of every lane on the road. The lines must be bright yellow and clearly mark the boundaries of the lanes.
2. 3D Bounding Boxes for Cars: For each car in the scene, draw a 3D bounding box using a "points-first" methodology:
   - First, mark the 8 key corner points of the car's 3D volume with small, visible dots.
   - Second, connect these dots with sharp lines to form the edges of the 3D box.
   - The final 3D box must accurately enclose the car, showing its orientation and volume correctly relative to the road's perspective.
"""

EVALUATION_PROMPT_TEMPLATE = """
As an AI evaluator for autonomous driving simulation data, please analyze the provided image based on these strict criteria:
The task was: {task_description}

Please evaluate the following:
1. Yellow Lane Lines: Are there bright yellow lines along the edges of EACH lane? (Score 1-10)
2. 3D Box Methodology: Are the 8 key corner points visible as dots? Are they connected by lines to form a 3D box? (Score 1-10)
3. 3D Box Accuracy: Do the boxes accurately fit the cars and follow the scene's perspective? (Score 1-10)

Provide a detailed feedback on what is missing or needs correction to strictly follow the "yellow lines" and "points-to-box" requirements.
"""
