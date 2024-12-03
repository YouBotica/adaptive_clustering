
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


using namespace std::chrono_literals;

//#define LOG
class AdaptiveClustering : public rclcpp::Node {

  public:
    AdaptiveClustering(): Node("adaptive_clustering"){

    
    //private_nh.param<std::string>("sensor_model", sensor_model, "VLP-16"); // VLP-16, HDL-32E, HDL-64E
    this->declare_parameter<std::string>("sensor_model", "VLP-16");
    //private_nh.param<bool>("print_fps", print_fps_, false);
    this->declare_parameter<bool>("print_fps", false);
    //private_nh.param<float>("z_axis_min", z_axis_min_, -0.8);
    this->declare_parameter<float>("z_axis_min", -0.8);
    //private_nh.param<float>("z_axis_max", z_axis_max_, 2.0);
    this->declare_parameter<float>("z_axis_max", 10.0);

    std::vector<int64_t> default_cluster_size_min = {50, 25, 20, 10, 5}; // Default values for cluster_size_min
    std::vector<int64_t> default_cluster_size_max = {200, 150, 100, 50, 30}; // Default values for cluster_size_max
    //private_nh.param<int>("cluster_size_min", s, 3);
    this->declare_parameter<std::vector<int64_t>>("cluster_size_min", default_cluster_size_min);
    //private_nh.param<int>("cluster_size_max", cluster_size_max_, 2200000);
    this->declare_parameter<std::vector<int64_t>>("cluster_size_max", default_cluster_size_max);

    //private_nh.param<int>("leaf", leaf_, 3);
    this->declare_parameter<int>("leaf", 3);
    //private_nh.param<float>("k_merging_threshold", k_merging_threshold_, 0.1);
    this->declare_parameter<float>("k_merging_threshold", 0.1);
    //private_nh.param<float>("z_merging_threshold", z_merging_threshold_, 0.0);
    this->declare_parameter<float>("z_merging_threshold", 0.0);
    //private_nh.param<float>("radius_min", radius_min_, 0.0);
    this->declare_parameter<float>("radius_min", 0.4);
    //private_nh.param<float>("radius_max", radius_max_, 30.0);
    this->declare_parameter<float>("radius_max", 120.0);
    // Whether we want to output bounding boxes, or the original algorithm line markers
    this->declare_parameter<bool>("generate_bounding_boxes", true);
    this->declare_parameter<float>("car_width",2.0);
    this->declare_parameter<float>("car_length",4.8768);
    // get regions from param
    std::vector<int64_t> default_regions = {5, 20, 30, 30, 30}; // Default values for regions
    this->declare_parameter<std::vector<int64_t>>("regions", default_regions);
    // get tolerance from param
    this->declare_parameter<float>("tolerance",2.0);
    this->declare_parameter<int>("region_max", 5); // how many regions you want to detect.



    sensor_model = this->get_parameter("sensor_model").get_parameter_value().get<std::string>();
    print_fps_ = this->get_parameter("print_fps").get_parameter_value().get<bool>();
    z_axis_min_ = this->get_parameter("z_axis_min").get_parameter_value().get<float>();
    z_axis_max_ = this->get_parameter("z_axis_max").get_parameter_value().get<float>();
    cluster_size_min_ = this->get_parameter("cluster_size_min").get_parameter_value().get<std::vector<int64_t>>();
    cluster_size_max_ = this->get_parameter("cluster_size_max").get_parameter_value().get<std::vector<int64_t>>();
    leaf_ = this->get_parameter("leaf").get_parameter_value().get<int>();
    k_merging_threshold_ = this->get_parameter("k_merging_threshold").get_parameter_value().get<float>();
    z_merging_threshold_ = this->get_parameter("z_merging_threshold").get_parameter_value().get<float>();
    radius_min_ = this->get_parameter("radius_min").get_parameter_value().get<float>();
    radius_max_ = this->get_parameter("radius_max").get_parameter_value().get<float>();
    generate_bounding_boxes = this->get_parameter("generate_bounding_boxes").get_parameter_value().get<bool>();
    car_width_ = this->get_parameter("car_width").get_parameter_value().get<float>();
    car_length_ = this->get_parameter("car_length").get_parameter_value().get<float>();
    regions_ = this->get_parameter("regions").get_parameter_value().get<std::vector<int64_t>>();
    tolerance_ = this->get_parameter("tolerance").get_parameter_value().get<float>();
    region_max_ = this->get_parameter("region_max").get_parameter_value().get<int>();

    /*** Subscribers ***/
    point_cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("ransac_non_ground", 10, std::bind(&AdaptiveClustering::pointCloudCallback, 
      this, std::placeholders::_1));
    //wall_points_sub = this->create_subscription<blackandgold_msgs::msg::Polynomial4Array>("/perception/wall_point_markers", 10, std::bind(&AdaptiveClustering::wallsCallback, 
    //  this, std::placeholders::_1));
    //ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("velodyne_points", 1, pointCloudCallback);

    /*** Publishers ***/
    //cluster_array_pub_ = private_nh.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
    cluster_array_pub_ = this->create_publisher<blackandgold_msgs::msg::ClusterArray>("clusters", 10);
    //cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
    cloud_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_filtered", 10);
    //pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
    pose_array_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("poses", 10);
    //marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
    marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clustering_markers", 10);

    bounding_boxes_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("lidar_bboxes", 10);
    vehicle_boxes_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("lidar_vehicle_bboxes", 10);
    wall_boxes_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("lidar_wall_bboxes", 10);
    vehicle_marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("vehicle_lidar_markers", 10);


    reset = true;//fps
    frames = 0;
    start_time = clock();

  }
  
  private:

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


    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ros_pc2_in) const 
    {
      // Retrieve parameters for "on the run" tuning:
      sensor_model = this->get_parameter("sensor_model").get_parameter_value().get<std::string>();
      print_fps_ = this->get_parameter("print_fps").get_parameter_value().get<bool>();
      z_axis_min_ = this->get_parameter("z_axis_min").get_parameter_value().get<float>();
      z_axis_max_ = this->get_parameter("z_axis_max").get_parameter_value().get<float>();
      cluster_size_min_ = this->get_parameter("cluster_size_min").get_parameter_value().get<std::vector<int64_t>>();
      cluster_size_max_ = this->get_parameter("cluster_size_max").get_parameter_value().get<std::vector<int64_t>>();
      leaf_ = this->get_parameter("leaf").get_parameter_value().get<int>();
      k_merging_threshold_ = this->get_parameter("k_merging_threshold").get_parameter_value().get<float>();
      z_merging_threshold_ = this->get_parameter("z_merging_threshold").get_parameter_value().get<float>();
      radius_min_ = this->get_parameter("radius_min").get_parameter_value().get<float>();
      radius_max_ = this->get_parameter("radius_max").get_parameter_value().get<float>();
      regions_ = this->get_parameter("regions").get_parameter_value().get<std::vector<int64_t>>();
      tolerance_ = this->get_parameter("tolerance").get_parameter_value().get<float>();

      
      if(print_fps_ && reset){frames=0; start_time=clock(); reset=false;}//fps
      
      /*** Convert ROS message to PCL ***/
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);

      

      pcl::IndicesPtr pc_indices(new std::vector<int>);
      for(unsigned int i = 0; i < pcl_pc_in->size(); ++i) {
        pc_indices->push_back(i);
      }


      /*** Divide the point cloud into nested circular regions ***/
      #if PCL_VERSION_COMPARE(<, 1, 11, 0)
        boost::array<std::vector<int>, 5> indices_array;
      #else
        std::array<std::vector<int>, 5> indices_array;
      #endif

      for(unsigned int i = 0; i < pc_indices->size(); i++) {
        float range = 0.0;
        for(int j = 0; j < region_max_; j++) {
          float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
      pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
      pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
          if(d2 > radius_min_ * radius_min_ && d2 < radius_max_ * radius_max_ &&
      d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
            indices_array[j].push_back((*pc_indices)[i]);
            break;
          }
          range += regions_[j];
        }
      }
      
      /*** Euclidean clustering ***/
      float tolerance = tolerance_;
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr > > clusters;
      int last_clusters_begin = 0;
      int last_clusters_end = 0;

      auto pre_time = rclcpp::Clock{}.now();
      //auto current_time = clock->now();

      //RCLCPP_INFO(this->get_logger(), "Current time: %ld.%09ld", current_time.seconds(), current_time.nanoseconds());
      
      for(int i = 0; i < region_max_; i++) {
        tolerance += 0.5; //3*0.1;
        if(indices_array[i].size() > cluster_size_min_[i]) {
          #if PCL_VERSION_COMPARE(<, 1, 11, 0)
            boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
          #else
            std::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
          #endif
          pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
          tree->setInputCloud(pcl_pc_in, indices_array_ptr);
            auto clustering_start = std::chrono::high_resolution_clock::now();

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(tolerance);
            ec.setMinClusterSize(cluster_size_min_[i]);
            ec.setMaxClusterSize(cluster_size_max_[i]);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pcl_pc_in);
            ec.setIndices(indices_array_ptr);
            ec.extract(cluster_indices);

            auto clustering_end = std::chrono::high_resolution_clock::now();
            auto clustering_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clustering_end - clustering_start).count();
            RCLCPP_DEBUG(this->get_logger(), "Clustering took %ld ms", clustering_duration);
          
          for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
              cluster->points.push_back(pcl_pc_in->points[*pit]);
      }
      /*** Merge clusters separated by nested regions ***/
      for(int j = last_clusters_begin; j < last_clusters_end; j++) {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        int K = 1; //the number of neighbors to search for
        std::vector<int> k_indices(K);
        std::vector<float> k_sqr_distances(K);
        kdtree.setInputCloud(cluster);
        if(clusters[j]->points.size() >= 1) {
          if(kdtree.nearestKSearch(*clusters[j], clusters[j]->points.size()-1, K, k_indices, k_sqr_distances) > 0) {
            if(k_sqr_distances[0] < k_merging_threshold_) {
        *cluster += *clusters[j];
        clusters.erase(clusters.begin()+j);
        last_clusters_end--;
        // std::cerr << "k-merging: clusters " << j << " is merged" << std::endl; 
            }
          }
        }
      }
      /**************************************************/
            cluster->width = cluster->size();
            cluster->height = 1;
            cluster->is_dense = true;
            clusters.push_back(cluster);
          }
          /*** Merge z-axis clusters ***/
          for(int j = last_clusters_end; j < clusters.size(); j++) {
            Eigen::Vector4f j_min, j_max;
            pcl::getMinMax3D(*clusters[j], j_min, j_max);
            for(int k = j+1; k < clusters.size(); k++) {
              Eigen::Vector4f k_min, k_max;
              pcl::getMinMax3D(*clusters[k], k_min, k_max);
              if(std::max(std::min((double)j_max[0], (double)k_max[0]) - std::max((double)j_min[0], (double)k_min[0]), 0.0) * std::max(std::min((double)j_max[1], (double)k_max[1]) - std::max((double)j_min[1], (double)k_min[1]), 0.0) > z_merging_threshold_) {
                *clusters[j] += *clusters[k];
                clusters.erase(clusters.begin()+k);
                //std::cerr << "z-merging: clusters " << k << " is merged into " << j << std::endl; 
              }
            }
          }
          /*****************************/
          last_clusters_begin = last_clusters_end;
          last_clusters_end = clusters.size();
        }
      }

      /*** Output ***/
      
      //if(cloud_filtered_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZ>);
      sensor_msgs::msg::PointCloud2 ros_pc2_out;
      //pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
      pcl::toROSMsg(*pcl_pc_in, ros_pc2_out);

      auto post_time = rclcpp::Clock{}.now();

      auto time_elapsed = post_time.seconds() - pre_time.seconds();
      // RCLCPP_INFO(this->get_logger(), "Time taken: '%f'", time_elapsed);

      cloud_filtered_pub_->publish(ros_pc2_out);
      //} 
      

      blackandgold_msgs::msg::ClusterArray cluster_array;
      geometry_msgs::msg::PoseArray pose_array;
      visualization_msgs::msg::MarkerArray marker_array;
      autoware_auto_perception_msgs::msg::BoundingBoxArray bounding_boxes;
      autoware_auto_perception_msgs::msg::BoundingBoxArray wall_bounding_boxes;
      autoware_auto_perception_msgs::msg::BoundingBoxArray vehicle_bounding_boxes;
      visualization_msgs::msg::MarkerArray bounding_boxes_markers;
      visualization_msgs::msg::MarkerArray vehicle_markers;
      
      for(int i = 0; i < clusters.size(); i++) {
        //if(cluster_array_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::PointCloud2 ros_pc2_out;
        pcl::toROSMsg(*clusters[i], ros_pc2_out);
        cluster_array.clusters.push_back(ros_pc2_out);
        //}
        
        //if(pose_array_pub_->get_subscription_count() > 0) {

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clusters[i], centroid);

        // filter out the detection of ourselves
        if (fabs(centroid[0]) <= car_length_/2 && fabs(centroid[1]) <= car_width_/2) {
          continue;
        }

        Eigen::Vector4f min, max;
        pcl::getMinMax3D(*clusters[i], min, max);

        autoware_auto_perception_msgs::msg::BoundingBox box;
        box.centroid.x = centroid[0];
        box.centroid.y = centroid[1];
        box.centroid.z = centroid[2];

        // Compute sized of bounding box
        box.size.x = max[0] - min[0];
        box.size.y = max[1] - min[1];
        box.size.z = max[2] - min[2];

        // set the number of points to the value of the box
        box.value = clusters[i]->points.size();
        
        // Compute roll angle of bounding box
        // float roll = atan2(max[1] - min[1], max[0] - min[0]);
        
        // Create quaternion from roll angle
        // tf2::Quaternion quaternion;
        // quaternion.setRPY(roll, 0, 0);
        // geometry_msgs::msg::Quaternion quat_msg;
        // quat_msg.x = quaternion.x();
        // quat_msg.y = quaternion.y();
        // quat_msg.z = quaternion.z();
        // quat_msg.w = quaternion.w();

        // Print quaternion components:
        // RCLCPP_INFO(this->get_logger(), "Quat x: '%f'", quat_msg.x);

        // box.orientation.x = quat_msg.x;
        // box.orientation.y = quat_msg.y;
        // box.orientation.z = quat_msg.z;
        // box.orientation.w = quat_msg.w;

        // geometry_msgs::msg::Pose pose;
        // pose.position.x = centroid[0];
        // pose.position.y = centroid[1];
        // pose.position.z = centroid[2];
        // pose.orientation = quat_msg;
        // pose_array.poses.push_back(pose);

        // RCLCPP_INFO(this->get_logger(), "Roll: '%f'", roll);

        bounding_boxes.boxes.push_back(box);

        // deal with markers
        visualization_msgs::msg::Marker m;
        m.header = ros_pc2_in->header;
        m.ns = "bbox";
        m.id = i;
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

        bool valid = isOutOfBounds_v2(polynomials, box);

        // figure out geometrically if it is a wall

        //if (!valid) 
        // NOTE May not need this with the addition of the off-map filter (CarProximityReporter)
        if ((box.size.x * box.size.y * box.size.z >= 12.0) || box.size.x > 6.0 || (box.size.y / box.size.x > 1.0) || !valid)
        { // If this is true, the box is bigger than the car
          // marker color
          m.color.r = 0.0;
          m.color.g = 0.0;
          m.color.b = 1.0;
          m.color.a = 0.75;
          m.lifetime.sec = 0;
          m.lifetime.nanosec = 100000000;
          wall_bounding_boxes.boxes.push_back(box);

          
        }
        else// if (abs(box.centroid.y) < 20.0)
        { // The box is a vehicle
          // marker color
          RCLCPP_DEBUG(this->get_logger(), "Poly size: '%i'", polynomials.polynomials.size());
          m.color.r = 1.0;
          m.color.g = 0.0;
          m.color.b = 0.0;
          m.color.a = 0.75;
          m.lifetime.sec = 0;
          m.lifetime.nanosec = 100000000;
          vehicle_bounding_boxes.boxes.push_back(box);
          vehicle_markers.markers.push_back(m);
        }


        bounding_boxes_markers.markers.push_back(m);

      }

      if(bounding_boxes.boxes.size()) {

        // Deal with headers:
        bounding_boxes.header = ros_pc2_in->header;
        bounding_boxes_pub_->publish(bounding_boxes);
        vehicle_bounding_boxes.header = ros_pc2_in->header;
        wall_bounding_boxes.header = ros_pc2_in->header;
        marker_array_pub_->publish(bounding_boxes_markers);

        marker_array_pub_->publish(bounding_boxes_markers);
        wall_boxes_pub_->publish(wall_bounding_boxes);
        vehicle_boxes_pub_->publish(vehicle_bounding_boxes);
        vehicle_marker_array_pub_->publish(vehicle_markers);
      }
      
      if(cluster_array.clusters.size()) {
        cluster_array.header = ros_pc2_in->header;
        cluster_array_pub_->publish(cluster_array);
      }

      // if(pose_array.poses.size()) {
      //   pose_array.header = ros_pc2_in->header;
      //   pose_array_pub_->publish(pose_array);
      // }
      
      if(marker_array.markers.size()) {
        marker_array_pub_->publish(marker_array);
      }
      
      if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;};//fps
    }

    /*** Parameters ***/
    mutable std::string sensor_model = "VLP-16";
    mutable bool print_fps_;
    mutable float z_axis_min_;
    mutable float z_axis_max_;
    mutable std::vector<int64_t> cluster_size_min_;
    mutable std::vector<int64_t> cluster_size_max_;
    mutable int leaf_;
    mutable float k_merging_threshold_;
    mutable float z_merging_threshold_;
    mutable float radius_min_;
    mutable float radius_max_;
    mutable float car_width_;
    mutable float car_length_;
    mutable std::vector<int64_t> regions_;
    mutable float tolerance_;
    mutable int region_max_ = 5; // 10 Change this value to match how far you want to detect.

    mutable int frames; 
    mutable clock_t start_time; 
    mutable bool reset;

    visualization_msgs::msg::MarkerArray boundary_points;


    bool generate_bounding_boxes;
    mutable blackandgold_msgs::msg::Polynomial4Array polynomials;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_filtered_pub_;
    rclcpp::Publisher<blackandgold_msgs::msg::ClusterArray>::SharedPtr cluster_array_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr bounding_boxes_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr wall_boxes_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub;
    //rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr wall_points_sub;

};


int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AdaptiveClustering>());
  rclcpp::shutdown();

  return 0;
}