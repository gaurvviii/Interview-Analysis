# Interview-Analysis

## Overview
The Interview Analysis project utilizes computer vision techniques to analyze video interviews, focusing on emotional expressions and eye movement patterns. This project aims to enhance the recruitment process by providing data-driven insights into candidates' non-verbal cues during interviews.

## Features
- **Emotional Detection**: Identifies and analyzes candidates' emotional expressions.
- **Eye Movement Tracking**: Monitors eye movement patterns to assess candidate engagement.
- **Data Visualization**: Provides visual representations of emotional responses and eye movement data.

## Requirements
- Python 3.x
- OpenCV
- TensorFlow (or any relevant libraries for your model)
- NumPy

## Installation
To set up the environment, run the following command:

```bash
pip install opencv-python tensorflow numpy
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Place your interview video files in the project directory or provide the path to the video in the code.

3. Update the video file path variable in the code to point to your interview video.

4. Run the script:
   ```bash
   python interview_analysis.py
   ```

5. The script will process the video and output the analysis results, including detected emotions and eye movement patterns.

## Example
To test the project, use a sample video of an interview. The analysis will provide insights based on the candidates' emotional states and engagement levels.

## Future Work
- Implement additional features, such as sentiment analysis on verbal responses.
- Explore advanced techniques for improved accuracy in emotion detection and eye movement tracking.

