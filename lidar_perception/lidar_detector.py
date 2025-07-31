import rclpy
from rclpy.node import Node
import os
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Pose, Vector3, Point, Quaternion
from std_msgs.msg import Header
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from mmdet3d.apis import LidarDet3DInferencer
from collections import deque

class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__('lidar_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/front/points',
            self.pointcloud_callback,
            1
        )
        self.publisher = self.create_publisher(
            MarkerArray,
            '/sensing/lidar/front/detections',
            1
        )
        init_args = {
            'model': "/home/twistbot/ws/twistbot/src/lidar_perception/lidar_perception/configs/centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d.py",
            'weights': '/home/twistbot/ws/twistbot/src/lidar_perception/lidar_perception/configs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20220811_045458-808e69ad.pth',
            'device': 'cuda:0',
        }
        self.inferencer = LidarDet3DInferencer(**init_args)
        self.get_logger().info('LidarDetectorNode initialized.')
        self.lidar_sweeps = deque(maxlen=10)

    @staticmethod
    def convert_to_marker_array(result_dict: dict, header: Header) -> MarkerArray:
        marker_array = MarkerArray()
        
        labels = result_dict['predictions'][0]['labels_3d']
        scores = result_dict['predictions'][0]['scores_3d']
        bboxes = result_dict['predictions'][0]['bboxes_3d']
        print(sorted(scores))
        for i, (label, score, bbox) in enumerate(zip(labels, scores, bboxes)):
            marker = Marker()
            marker.header = header
            
            # Use unique ID per marker (can also use i)
            marker.id = i
            
            marker.ns = "detections"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position of the box center
            marker.pose.position = Point(x=bbox[0], y=bbox[1], z=bbox[2])
            
            # Orientation (yaw angle around z-axis)
            q = Rotation.from_euler('z', bbox[6]).as_quat()
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            
            # Size of the box
            marker.scale = Vector3(x=bbox[3], y=bbox[4], z=bbox[5])
            
            # Color based on label or fixed color (example: red with alpha = score)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = max(0.1, min(score, 1.0))  # transparency by score
            
            if score > 0.6:
                marker_array.markers.append(marker)
        
        return marker_array
    
    def run_model(self):
        x = {'points': np.vstack(self.lidar_sweeps)}
        results = self.inferencer(x)
        return results

    def pointcloud_callback(self, msg):
        cloud_points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        cloud_points_1 = np.array(cloud_points)
        cloud_array = np.zeros((len(cloud_points_1), 5), dtype=np.float32)
        cloud_array[:, 0] = cloud_points_1['x']
        cloud_array[:, 1] = cloud_points_1['y']
        cloud_array[:, 2] = cloud_points_1['z']
        cloud_array[:, 3] = cloud_points_1['intensity']
        cloud_array[:, 4] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.lidar_sweeps.append(cloud_array)
        results = self.run_model()
        if results is not None:
            detection_array_msg = self.convert_to_marker_array(results, msg.header)
            self.publisher.publish(detection_array_msg)
            self.get_logger().info('Published detection array message.')
        return

def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()