Local server for presentation DEMO in CS256: Comparison between YOLOv7 and FasterRCNN on a swimmer dataset.

# Prerequisites

First, clone FasterRCNN model repo:

```git clone --quiet https://github.com/tensorflow/models.git```

Then install libraries for YOLOv7 model:

```pip install -r yolo/requirements.txt```

Then install dependencies for Faster RCNN model:

```pip install -r models/official/requirements.txt```

# Run

To run the local server:

```streamlit run main.py```
