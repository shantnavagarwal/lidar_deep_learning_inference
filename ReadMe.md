# 3D Lidar Perception
## Features
- Deploys Centerpoint via mmdetection3d. Many models in literature provide configs for mmdetection3d allowing us to use better models easily.
- Lidar data is received as a PointCloud2 message via a ROS topic
- 10 Lidar frames are used for making predictions. This improves temporal context
- Results are published as a MarkerArray message and can be visualized on RViZ

## Improvements
- Use mmdeploy + TensorRT to improve inference speed
- Read Rosbags via PythonAPI and batch inputs for faster processing
