// ROS
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "blackandgold_msgs/msg/cluster_array.hpp"
#include "blackandgold_msgs/msg/polynomial4_array.hpp"
#include <autoware_auto_perception_msgs/msg/bounding_box_array.hpp>
#include <autoware_auto_perception_msgs/msg/bounding_box.hpp>
#include <tf2/LinearMath/Quaternion.h>

// PCL
// #include <pcl/search/kdtree.h>
#include "pcl/pcl_config.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/filters/voxel_grid.h" 
#include "pcl/filters/passthrough.h" 
#include "pcl/segmentation/conditional_euclidean_clustering.h"
#include "pcl/common/common.h"
#include "pcl/common/centroid.h"
#include <iostream>
#include <cmath>

using namespace std::chrono_literals;

class AdaptiveClustering : public rclcpp::Node {
public:
    AdaptiveClustering() : Node("adaptive_clustering") {
        // Declare parameters
        this->declare_parameter<std::string>("sensor_model", "VLP-16");
        this->declare_parameter<bool>("print_fps", false);
        this->declare_parameter<float>("z_axis_min", -0.8);
        this->declare_parameter<float>("z_axis_max", 10.0);
        this->declare_parameter<int>("cluster_size_min", 10);
        this->declare_parameter<int>("cluster_size_max", 5000);
        this->declare_parameter<float>("clustering_tolerance", 1.5);
        this->declare_parameter<float>("clustering_x_tolerance", 2.0);
        this->declare_parameter<float>("clustering_y_tolerance", 0.5);


        this->declare_parameter<float>("k_merging_threshold", 0.1);
        this->declare_parameter<float>("radius_min", 0.4);
        this->declare_parameter<float>("radius_max", 120.0);
        this->declare_parameter<bool>("generate_bounding_boxes", true);
        this->declare_parameter<float>("car_width", 2.0);
        this->declare_parameter<float>("car_length", 4.8768);
        this->declare_parameter<float>("");

        // Get parameter values
        sensor_model_ = this->get_parameter("sensor_model").get_parameter_value().get<std::string>();
        z_axis_min_ = this->get_parameter("z_axis_min").get_parameter_value().get<float>();
        z_axis_max_ = this->get_parameter("z_axis_max").get_parameter_value().get<float>();
        cluster_size_min_ = this->get_parameter("cluster_size_min").get_parameter_value().get<int>();
        cluster_size_max_ = this->get_parameter("cluster_size_max").get_parameter_value().get<int>();
        clustering_tolerance_ = this->get_parameter("clustering_tolerance").get_parameter_value().get<float>();
        clustering_x_tolerance_ = this->get_parameter("clustering_x_tolerance").get_parameter_value().get<float>();
        clustering_y_tolerance_ = this->get_parameter("clustering_y_tolerance").get_parameter_value().get<float>();
        radius_min_ = this->get_parameter("radius_min").get_parameter_value().get<float>();
        radius_max_ = this->get_parameter("radius_max").get_parameter_value().get<float>();

        // Create subscriber for point cloud data
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/perception/linefit_seg/ransac_non_ground", 10, 
            std::bind(&AdaptiveClustering::pointCloudCallback, this, std::placeholders::_1)
        );

        // Create publishers for clustering results
        cluster_array_pub_ = this->create_publisher<blackandgold_msgs::msg::ClusterArray>("clusters", 10);
        // cloud_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_filtered", 10);
        vehicle_boxes_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("/perception/lidar_vehicle_bboxes", 10);
        rejected_boxes_pub_ = create_publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>("/perception/rejected_boxes", 10);
        
        vehicle_marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/vehicle_lidar_markers", 10);
        rejected_marker_array_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/perception/rejected_markers", 10);

    }

private:
    // Parameters
    std::string sensor_model_;
    float z_axis_min_;
    float z_axis_max_;
    int cluster_size_min_;
    int cluster_size_max_;
    float clustering_tolerance_;
    float clustering_x_tolerance_;
    float clustering_y_tolerance_;
    float radius_min_;
    float radius_max_;

    // Publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<blackandgold_msgs::msg::ClusterArray>::SharedPtr cluster_array_pub_;
    // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_filtered_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr vehicle_boxes_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr rejected_boxes_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::BoundingBoxArray>::SharedPtr wall_boxes_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vehicle_marker_array_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr rejected_marker_array_pub_;

    // Custom conditional clustering function
    static bool conditionalClusteringFunction(const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance) {
        
        // float clustering_x_tolerance_ = this->get_parameter("clustering_x_tolerance").get_parameter_value().get<float>();
        // float clustering_y_tolerance_ = this->get_parameter("clustering_y_tolerance").get_parameter_value().get<float>();
        // Custom condition for clustering goes here:
        if (std::abs(point_a.x - point_b.x) < 1.5f && std::abs(point_a.y - point_b.y) < 0.5f) {
            return true;
        }
        return false;
    }


    // Function to remove overlapping clusters based on IoU threshold
    autoware_auto_perception_msgs::msg::BoundingBoxArray removeOverlappingClusters(autoware_auto_perception_msgs::msg::BoundingBoxArray& bounding_boxes_msg, float iou_threshold) const {
        autoware_auto_perception_msgs::msg::BoundingBoxArray filtered_boxes;
        auto bounding_boxes = bounding_boxes_msg.boxes;

        for (size_t i = 0; i < bounding_boxes.size(); ++i) {
            bool is_overlapping = false;
            for (size_t j = i + 1; j < bounding_boxes.size(); ++j) {
                if (computeIoU(bounding_boxes[i], bounding_boxes[j]) > iou_threshold) {
                    is_overlapping = true;
                    break;
                }
            }
            if (!is_overlapping) {
                filtered_boxes.boxes.push_back(bounding_boxes[i]);
            }
        }
        filtered_boxes.header = bounding_boxes_msg.header;
        return filtered_boxes;
    }

        // Helper function to compute IoU between two bounding boxes
    float computeIoU(const autoware_auto_perception_msgs::msg::BoundingBox& box1, const autoware_auto_perception_msgs::msg::BoundingBox& box2) const {
        float x_min = std::max(box1.centroid.x - box1.size.x / 2, box2.centroid.x - box2.size.x / 2);
        float x_max = std::min(box1.centroid.x + box1.size.x / 2, box2.centroid.x + box2.size.x / 2);
        float y_min = std::max(box1.centroid.y - box1.size.y / 2, box2.centroid.y - box2.size.y / 2);
        float y_max = std::min(box1.centroid.y + box1.size.y / 2, box2.centroid.y + box2.size.y / 2);

        float intersection_area = std::max(0.0f, x_max - x_min) * std::max(0.0f, y_max - y_min);
        float box1_area = box1.size.x * box1.size.y;
        float box2_area = box2.size.x * box2.size.y;

        float iou = intersection_area / (box1_area + box2_area - intersection_area);
        return iou;
    }


    // Callback for point cloud datak
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ros_pc2_in) const {
        // Print pointcloud size
        RCLCPP_DEBUG(this->get_logger(), "Received point cloud with %d points", ros_pc2_in->width * ros_pc2_in->height);

        // Convert ROS message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);

        // Manually remove NaN points (alternative to removeNaNFromPointCloud)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& point : pcl_pc_in->points) {
            if (pcl::isFinite(point)) {
                filtered_cloud->points.push_back(point);
            }
        }
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;
        
        // FIXME: COMMENT ONCE CONFIGURED
        float clustering_tolerance_ = this->get_parameter("clustering_tolerance").get_parameter_value().get<float>();

        // Set up KdTree for clustering
        // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        // tree->setInputCloud(pcl_pc_in);

        // Perform Conditional Euclidean Clustering
        pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec;
        // cec.extract_removed_clusters_ = true; // This requires a setting a PCL private member variable
        cec.setInputCloud(filtered_cloud); // pcl_pc_in
        cec.setConditionFunction(&conditionalClusteringFunction);
        cec.setClusterTolerance(clustering_tolerance_);  // Modify as needed
        cec.setMinClusterSize(cluster_size_min_);
        cec.setMaxClusterSize(cluster_size_max_);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::IndicesClustersPtr small_clusters(new std::vector<pcl::PointIndices>);
        pcl::IndicesClustersPtr large_clusters(new std::vector<pcl::PointIndices>);
        cec.segment(cluster_indices);
        // cec.getRemovedClusters(small_clusters, large_clusters);

        // Process and publish rejected clusters
        // processRejectedClusters(*small_clusters, *large_clusters, filtered_cloud, ros_pc2_in->header);
        

        // Generate bounding boxes with markers
        autoware_auto_perception_msgs::msg::BoundingBoxArray bounding_boxes;
        visualization_msgs::msg::MarkerArray markers;

        // Print number of clusters
        RCLCPP_INFO(this->get_logger(), "Found %d clusters", cluster_indices.size());

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (int idx : indices.indices) {
                cluster->points.push_back(pcl_pc_in->points[idx]);
            }

            // Compute centroid
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            Eigen::Vector4f min, max;
            pcl::getMinMax3D(*cluster, min, max);

            float car_width_ = 2.0;
            float car_length_ = 4.8768;

            // filter out the detection of ourselves
            if (fabs(centroid[0]) <= car_length_/2 && fabs(centroid[1]) <= car_width_/2) {
              continue;
            }

            // Compute bounding box
            autoware_auto_perception_msgs::msg::BoundingBox bounding_box;
            bounding_box.centroid.x = centroid[0];
            bounding_box.centroid.y = centroid[1];
            bounding_box.centroid.z = centroid[2];
            bounding_box.size.x = max[0] - min[0];
            bounding_box.size.y = max[1] - min[1];
            bounding_box.size.z = max[2] - min[2];
            bounding_boxes.boxes.push_back(bounding_box);


            // Create marker for bounding box
            visualization_msgs::msg::Marker marker;
            marker.header = ros_pc2_in->header;
            // marker.ns = "bounding_boxes";
            marker.id = bounding_boxes.boxes.size() - 1;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = bounding_box.centroid.x;
            marker.pose.position.y = bounding_box.centroid.y;
            marker.pose.position.z = bounding_box.centroid.z;

            marker.scale.x = bounding_box.size.x;
            marker.scale.y = bounding_box.size.y;
            marker.scale.z = bounding_box.size.z;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.5;
            markers.markers.push_back(marker);
        }

        // Remove overlapping clusters
        float iou_threshold = 0.5; // Adjust this threshold as needed
        auto filtered_boxes_msg = removeOverlappingClusters(bounding_boxes, iou_threshold);

        // Publish bounding boxes
        bounding_boxes.header = ros_pc2_in->header;
        vehicle_boxes_pub_->publish(bounding_boxes);
        // Publish markers
        vehicle_marker_array_pub_->publish(markers);


    }


    // Function to process and publish rejected clusters (both small and large)
    void processRejectedClusters(
        const std::vector<pcl::PointIndices>& small_clusters, 
        const std::vector<pcl::PointIndices>& large_clusters, 
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const std_msgs::msg::Header& header) const {
        
        autoware_auto_perception_msgs::msg::BoundingBoxArray rejected_boxes;
        visualization_msgs::msg::MarkerArray rejected_markers;
        
        // Helper function to process and generate bounding boxes and markers
        auto processCluster = [&](const std::vector<pcl::PointIndices>& clusters, const std::string& marker_ns) {
            for (const auto& indices : clusters) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
                for (int idx : indices.indices) {
                    cluster->points.push_back(input_cloud->points[idx]);
                    RCLCPP_INFO(this->get_logger(), "Rejected cluster size: %d", cluster->points.size());
                }

                // Compute centroid
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*cluster, centroid);

                Eigen::Vector4f min, max;
                pcl::getMinMax3D(*cluster, min, max);

                // Compute bounding box
                autoware_auto_perception_msgs::msg::BoundingBox bounding_box;
                bounding_box.centroid.x = centroid[0];
                bounding_box.centroid.y = centroid[1];
                bounding_box.centroid.z = centroid[2];
                bounding_box.size.x = max[0] - min[0];
                bounding_box.size.y = max[1] - min[1];
                bounding_box.size.z = max[2] - min[2];
                rejected_boxes.boxes.push_back(bounding_box);

                // Create marker for bounding box
                visualization_msgs::msg::Marker marker;
                marker.header = header;
                marker.ns = marker_ns;  // Give a fancy namespace for easier filtering
                marker.id = rejected_boxes.boxes.size() - 1;
                marker.type = visualization_msgs::msg::Marker::CUBE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = bounding_box.centroid.x;
                marker.pose.position.y = bounding_box.centroid.y;
                marker.pose.position.z = bounding_box.centroid.z;

                marker.scale.x = bounding_box.size.x;
                marker.scale.y = bounding_box.size.y;
                marker.scale.z = bounding_box.size.z;

                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;  // Give it a distinct color (blue)
                marker.color.a = 0.75; // Make it more transparent for debugging purposes

                rejected_markers.markers.push_back(marker);
            }
        };

        // Process small and large clusters separately
        processCluster(small_clusters, "small_rejected_clusters");
        processCluster(large_clusters, "large_rejected_clusters");

        // Publish rejected bounding boxes and markers
        rejected_boxes.header = header;
        rejected_boxes_pub_->publish(rejected_boxes);
        rejected_marker_array_pub_->publish(rejected_markers);
        }
    };

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AdaptiveClustering>());
    rclcpp::shutdown();
    return 0;
}
