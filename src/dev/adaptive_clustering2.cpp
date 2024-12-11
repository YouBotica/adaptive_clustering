
// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "blackandgold_msgs/msg/cluster_array.hpp"
#include "blackandgold_msgs/msg/polynomial4.hpp"
#include "blackandgold_msgs/msg/polynomial4_array.hpp"
#include <autoware_auto_perception_msgs/msg/bounding_box_array.hpp>
#include <autoware_auto_perception_msgs/msg/bounding_box.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>

// PCL
#include "pcl/pcl_config.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/filters/voxel_grid.h" 
#include "pcl/filters/passthrough.h" 
#include "pcl/segmentation/extract_clusters.h" 
#include "pcl/common/common.h" 
#include "pcl/common/centroid.h"
#include <iostream>
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

// RANSAC:
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>

using namespace std::chrono_literals;

//#define LOG
class AdaptiveClustering : public rclcpp::Node {

  public:
    AdaptiveClustering(): Node("adaptive_clustering"){


    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_cloud_pub_;
    // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
    // rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub;

    point_cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "ransac_non_ground", 10, std::bind(&AdaptiveClustering::pointCloudCallback, this, std::placeholders::_1)
        );

    cylinder_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder_cloud", 10);
    vehicle_boxes_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("lidar_vehicle_bboxes", 10);
    vehicle_marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("vehicle_lidar_markers", 10);
    

    //regions_[0] = 5; regions_[1] = 20; regions_[2] = 30; regions_[3] = 30; regions_[4] = 30; // FIXME: Add these to parameter files

    //reset = true;//fps
    //frames = 0;
    //start_time = clock();

  }
  
  private:

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub;
    autoware_auto_perception_msgs::msg::BoundingBox box;
    sensor_msgs::msg::PointCloud2::SharedPtr ros_pc2_in;
    blackandgold_msgs::msg::Polynomial4Array polynomials;
    autoware_auto_perception_msgs::msg::BoundingBoxArray vehicle_bounding_boxes;
    visualization_msgs::msg::MarkerArray vehicle_markers;




    // Polynomial structure:
    struct RootsAndCount 
    {
      int count;
      float roots[3];
    };


    bool isOutOfBounds_v2(blackandgold_msgs::msg::Polynomial4Array polynomials, autoware_auto_perception_msgs::msg::BoundingBox box) const
    {
      for(unsigned int i = 0; i < polynomials.polynomials.size(); i++) 
      {
        if (box.centroid.x >= polynomials.polynomials[i].x_min.data && 
          box.centroid.x <= polynomials.polynomials[i].x_max.data) 
        {
          auto polynomial = polynomials.polynomials[i].polynomial;
          float threshold_value = polynomial.data[0]*pow(box.centroid.x,4) + polynomial.data[1]*pow(box.centroid.x,3)
          + polynomial.data[2]*pow(box.centroid.x,2)  + polynomial.data[3]*box.centroid.x + polynomial.data[4];
          


          if (abs(box.centroid.y) >= abs(threshold_value) - 1.0 && std::signbit(box.centroid.y) == std::signbit(threshold_value))
          {
            RCLCPP_INFO(this->get_logger(), "Threshold: '%f'", threshold_value);
            return false;
          }

        }

      }
      return true;
    }


  

   pcl::PointCloud<pcl::PointXYZ>::Ptr detectCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_pc_in) {

    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_cloud_pub_;
    // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
    // rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
  
    // Create segmentation object for cylinder
    //pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    pcl::PointIndices::Ptr cylinder_inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr cylinder_coefficients(new pcl::ModelCoefficients());

    
    // Set axis for the cylinder model and epsilon angle tolerance
    seg.setAxis(Eigen::Vector3f(0, 1, 0));  // Adjust as needed for expected cylinder orientation
    seg.setEpsAngle(0.2);                   // Allow tolerance in angle

    // Define radius limits
    seg.setRadiusLimits(0.1, 0.5);  // Expected radius range of the cylinder

    // Normal estimation object
    //pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
   // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

    // Create a KdTree for searching neighbors
   // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
   // ne.setSearchMethod(tree);
   // ne.setInputCloud(pcl_pc_in);  // Input point cloud
    // ne.setRadiusSearch(0.03);  // Radius in meters (adjust based on your point cloud scale)
    // Alternatively, you can use setKSearch:
   // ne.setKSearch(50);  // Use 50 nearest neighbors
   // ne.compute(*normals);         // Compute the normals

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);  // Use cylinder model
    seg.setMethodType(pcl::SAC_RANSAC);       // Use RANSAC for model fitting
    seg.setMaxIterations(1000);               // Set max iterations
    seg.setDistanceThreshold(0.2);            // Set distance threshold for inliers

    // Set input point cloud and normals
    seg.setInputCloud(pcl_pc_in);
    // seg.setInputNormals(normals);  // Provide the normals


    seg.segment(*cylinder_inliers, *cylinder_coefficients);

    if (cylinder_inliers->indices.empty()) {
        RCLCPP_INFO(this->get_logger(), "No cylinder found.");
        return nullptr;
    }

    // Step 2: Extract inliers and calculate the length
    if (cylinder_inliers->indices.size() > 0) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(pcl_pc_in);
        extract.setIndices(cylinder_inliers);
        extract.setNegative(false);
        extract.filter(*cylinder_cloud);

        // Calculate cylinder axis direction vector from model coefficients
        Eigen::Vector3f cylinder_axis(cylinder_coefficients->values[3],
                                      cylinder_coefficients->values[4],
                                      cylinder_coefficients->values[5]);

        // Project points onto the cylinder axis
        Eigen::Vector3f base_point(cylinder_coefficients->values[0],
                                   cylinder_coefficients->values[1],
                                   cylinder_coefficients->values[2]);

        float min_proj = std::numeric_limits<float>::max();
        float max_proj = -std::numeric_limits<float>::max();

        for (const auto& point : cylinder_cloud->points) {
            Eigen::Vector3f pt(point.x, point.y, point.z);
            float projection = cylinder_axis.dot(pt - base_point);

            if (projection < min_proj) min_proj = projection;
            if (projection > max_proj) max_proj = projection;
        }

        // Compute the cylinder length
        float cylinder_length = max_proj - min_proj;

        // Step 3: Check if cylinder length matches desired length range
        float desired_min_length = .2;  // Define minimum length 1.0
        float desired_max_length = 5.0;  // Define maximum length 2.5
        if (cylinder_length >= desired_min_length && cylinder_length <= desired_max_length) {
            // Cylinder length is acceptable; proceed with processing
            sensor_msgs::msg::PointCloud2 cylinder_msg;
            pcl::toROSMsg(*cylinder_cloud, cylinder_msg);
            cylinder_msg.header.frame_id = "center_of_gravity";
            cylinder_cloud_pub_->publish(cylinder_msg);

            computeBoundingBoxForCylinder(cylinder_cloud, *cylinder_coefficients);
            return cylinder_cloud;

        } else {
            // Cylinder length is out of desired range; discard
            RCLCPP_INFO(this->get_logger(), "Cylinder length is out of desired range; discard");
            return nullptr;
        } 
    }
    return nullptr;
}


    void computeBoundingBoxForCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cylinder_cloud,
                                   const pcl::ModelCoefficients& coefficients) {

      // autoware_auto_perception_msgs::msg::BoundingBox box;
      // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_cloud_pub_;
      // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
      // rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
      // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
      // blackandgold_msgs::msg::Polynomial4Array polynomials;
      // autoware_auto_perception_msgs::msg::BoundingBoxArray vehicle_bounding_boxes;
      // visualization_msgs::msg::MarkerArray vehicle_markers;
      // sensor_msgs::msg::PointCloud2::SharedPtr ros_pc2_in;

      // Set centroid (x, y, z)
      box.centroid.x = coefficients.values[0];
      box.centroid.y = coefficients.values[1];
      box.centroid.z = coefficients.values[2];

      // Set dimensions based on cylinder radius and approximate height from cylinder_cloud
      float cylinder_radius = coefficients.values[6];
      box.size.x = cylinder_radius * 2;    // Diameter
      box.size.y = cylinder_radius * 2;    // Diameter
      box.size.z = computeCylinderHeight(cylinder_cloud);  // Calculate height

      // Set orientation along cylinder axis
      tf2::Quaternion orientation;
      orientation.setRPY(0, 0, atan2(coefficients.values[4], coefficients.values[3]));
      box.orientation.x = orientation.x();
      box.orientation.y = orientation.y();
      box.orientation.z = orientation.z();
      box.orientation.w = orientation.w();


      // deal with markers
      visualization_msgs::msg::Marker m;
      m.header = ros_pc2_in->header;
      m.ns = "bbox";
     // m.id = i;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = box.centroid.x;
      m.pose.position.y = box.centroid.y;
      m.pose.position.z = box.centroid.z;
      m.pose.orientation.x = box.orientation.x;
      m.pose.orientation.y = box.orientation.y;
      m.pose.orientation.z = box.orientation.z;
      m.pose.orientation.w = box.orientation.w;

      m.scale.x = box.size.x;
      m.scale.y = box.size.y;
      m.scale.z = box.size.z;

      // bool valid = isOutOfBounds_v2(polynomials, box);

      // color the vechicle box red
      RCLCPP_DEBUG(this->get_logger(), "Poly size: '%lu'", polynomials.polynomials.size());
      m.color.r = 1.0;
      m.color.g = 0.0;
      m.color.b = 0.0;
      m.color.a = 0.75;
      m.lifetime.sec = 0;
      m.lifetime.nanosec = 100000000;
      vehicle_bounding_boxes.boxes.push_back(box);
      vehicle_markers.markers.push_back(m);      

      if(vehicle_bounding_boxes.boxes.size()) {

        // Deal with headers:
        vehicle_bounding_boxes.header = ros_pc2_in->header;

        // Publish bounding box
        vehicle_boxes_pub_->publish(vehicle_bounding_boxes);
        vehicle_marker_array_pub_->publish(vehicle_markers);
      }

    }

    float computeCylinderHeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cylinder_cloud) {
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cylinder_cloud, min_pt, max_pt);
        return max_pt[2] - min_pt[2];
    }









    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ros_pc2_in)
    {


      // visualization_msgs::msg::MarkerArray boundary_points;
      // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub;
      // blackandgold_msgs::msg::Polynomial4Array polynomials;
      // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_cloud_pub_;
      // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
      // rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
 
      /*** Convert ROS message to PCL ***/
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);

      // check input to see if empty, useful for debugging
      if (pcl_pc_in->empty()) {
        RCLCPP_WARN(this->get_logger(), "Input cloud is empty. Skipping cylinder detection.");
        return;
        }


      // Call RANSAC Cylinder Function
      pcl::PointCloud<pcl::PointXYZ>::Ptr ransac_cylinder_cloud = detectCylinder(pcl_pc_in);



      

  }
};



int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AdaptiveClustering>());
  rclcpp::shutdown();

  return 0;
}