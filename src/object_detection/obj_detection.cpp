#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
//#include <quadrobot_msgs/Plane.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ros/conversions.h>
#include "pcl_ros/point_cloud.h"

// PCL specific includes
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>


#include <iostream>
#include <fstream>
#include <boost/thread/thread.hpp>

int iter=0;

typedef pcl::PointXYZ PointT;

struct Scalar
{
  int r;
  int g;
  int b;
};



class Cluster
{
  private:
  ros::NodeHandle nh_;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber cloud_sub_;
  ros::Publisher pub_cluster,pub_marker,pub_plane,cloud_pub_;
  ros::Publisher pub_msg;
  
  public:
  //  Init params
  double val_pass=3.0, val_vox=0.015f;
  double ClusterTolerance=0.05;
  double x_max,x_min,y_max,y_min,z_max,z_min;
  int MinClusterSize = 200;
  int MaxClusterSize = 25000;
  bool doFILTER = true;

  public:
  Cluster(ros::NodeHandle nh)
  : nh_(nh)
  {      
    // Get params
    nh.getParam("/object_detection/doFILTER", doFILTER);
    nh.getParam("/object_detection/val_pass", val_pass);
    nh.getParam("/object_detection/val_vox", val_vox);
    nh.getParam("/object_detection/ClusterTolerance", ClusterTolerance);
    nh.getParam("/object_detection/MinClusterSize", MinClusterSize);
    nh.getParam("/object_detection/MaxClusterSize", MaxClusterSize);
    nh.getParam("/object_detection/x_min", x_min);
    nh.getParam("/object_detection/x_max", x_max);
    nh.getParam("/object_detection/y_min", y_min);
    nh.getParam("/object_detection/y_max", y_max);
    nh.getParam("/object_detection/z_min", z_min);
    nh.getParam("/object_detection/z_max", z_max);

    // Topics
    std::string point_cloud_topic_ = "/camera/depth_registered/points";
    nh.getParam("point_cloud_topic", point_cloud_topic_);

    
    // Create a ROS publisher
    cloud_sub_ = nh_.subscribe ("/camera/depth_registered/points", 1, &Cluster::cloud_cb,this);            

    // Create a ROS publisher for the output model coefficients
    pub_cluster = nh.advertise<sensor_msgs::PointCloud2> ("/object/clusters", 100);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray> ("/object/markers", 10);
    pub_msg = nh.advertise<std_msgs::String>("/object/num_obj", 1000);

  }
  ~Cluster()
  {

  }
  void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
  {
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // 
    pcl::fromROSMsg (*input, *cloud);    
   

    // All the objects needed
    pcl::PCDReader reader;
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    // Datasets
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    
    pcl::PointCloud<PointT>::Ptr cloud_voxel(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_noise(new pcl::PointCloud<PointT>);


    pcl::PointCloud<pcl::PointXYZI> coloured_point_cloud;

    // -------------------------------------------------
    // -----Create coloured point cloud for viewer -----
    // -------------------------------------------------

  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZ> vox;

  if (doFILTER)
  {
    // passthrough filter to remove spurious NaNs
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z"); 
    pass.setFilterLimits(0, val_pass); 
    pass.filter(*cloud_voxel);

    //voxel filtering
    vox.setInputCloud(cloud_voxel);
    vox.setLeafSize(val_vox, val_vox, val_vox);
    vox.filter(*cloud_filtered);
  }
  else
    cloud_filtered.swap(cloud);

  // Creating the KdTree object for the search method of the extraction
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (ClusterTolerance); // 2cm
  ec.setMinClusterSize (MinClusterSize);
  ec.setMaxClusterSize (MaxClusterSize);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  
  
  visualization_msgs::MarkerArray markers;
  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<PointT>::Ptr cloud_obj(new pcl::PointCloud<PointT>());

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      pcl::PointXYZ pt = cloud_filtered->points[*pit];
      pcl::PointXYZI pt2;
      pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z;
      pt2.intensity = (float)(j + 1);

      cloud_obj->points.push_back(pt);
      coloured_point_cloud.push_back(pt2);

    }
    getBoundingBox(cloud_obj, j, markers);
    j++;

  }

  // Convert To ROS data type
  sensor_msgs::PointCloud2 output;
  pcl::PCLPointCloud2 cloud_p;
  pcl::toPCLPointCloud2(coloured_point_cloud, cloud_p);

  pcl_conversions::fromPCL(cloud_p, output);
  output.header.frame_id = "camera_depth_optical_frame";
  pub_cluster.publish(output);

  // Publish markers
  pub_marker.publish(markers);

  std_msgs::String msg;

  std::stringstream ss;
  ss << "Number of objects= "<< j;
  msg.data = ss.str();
  pub_msg.publish(msg);
  }

  int getBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, int idx, visualization_msgs::MarkerArray &markers)
  {
    // compute principal direction
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*point_cloud_ptr, centroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*point_cloud_ptr, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    // move the points to the that reference frame
    Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
    Eigen::Matrix4f w2p(Eigen::Matrix4f::Identity());
    p2w.block<3, 3>(0, 0) = eigDx.transpose();
    p2w.block<3, 1>(0, 3) = -1.f * (p2w.block<3, 3>(0, 0) * centroid.head<3>());
    w2p = p2w.inverse();
    pcl::PointCloud<pcl::PointXYZ> cPoints;
    pcl::transformPointCloud(*point_cloud_ptr, cPoints, p2w);

    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(cPoints, min_pt, max_pt);
    const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());

    // final transform
    const Eigen::Quaternionf qfinal(eigDx);
    const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();

    double width = max_pt.z-min_pt.z;
    double depth = max_pt.y-min_pt.y;
    double height= max_pt.x-min_pt.x;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_ptr->resize(9);
    cloud_ptr->points[0] = pcl::PointXYZ(max_pt.x, max_pt.y, max_pt.z);		// max_pt
    cloud_ptr->points[1] = pcl::PointXYZ(max_pt.x, max_pt.y, min_pt.z);
    cloud_ptr->points[2] = pcl::PointXYZ(max_pt.x, min_pt.y, max_pt.z);
    cloud_ptr->points[3] = pcl::PointXYZ(min_pt.x, max_pt.y, max_pt.z);
    cloud_ptr->points[4] = pcl::PointXYZ(min_pt.x, min_pt.y, min_pt.z);		// min_pt
    cloud_ptr->points[5] = pcl::PointXYZ(min_pt.x, min_pt.y, max_pt.z);
    cloud_ptr->points[6] = pcl::PointXYZ(min_pt.x, max_pt.y, min_pt.z);
    cloud_ptr->points[7] = pcl::PointXYZ(max_pt.x, min_pt.y, min_pt.z);
    cloud_ptr->points[8] = pcl::PointXYZ(0.0, 0.0, 0.0);					// center
    pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, w2p);

    ////////////////////////////////////   marker setting
    visualization_msgs::Marker marker_pts,marker_vector;
    
    // set corner point
    marker_pts.header.frame_id = "camera_depth_optical_frame";
    marker_pts.header.stamp = ros::Time::now();
    geometry_msgs::Point pt[9];
    for(int i=0; i<9;i++)
    {
      pt[i].x=cloud_ptr->points[i].x/1.0f;  
      pt[i].y=cloud_ptr->points[i].y/1.0f;
      pt[i].z=cloud_ptr->points[i].z/1.0f;
      if(i!=8)
      marker_pts.points.push_back(pt[i]);
    }
    marker_pts.ns = "corner_points";
    marker_pts.id = idx;
    marker_pts.type = visualization_msgs::Marker::CUBE;
    marker_pts.action = visualization_msgs::Marker::ADD;
    marker_pts.pose.position.x = cloud_ptr->points[8].x/1.0f;
    marker_pts.pose.position.y = cloud_ptr->points[8].y/1.0f;
    marker_pts.pose.position.z = cloud_ptr->points[8].z/1.0f;
    marker_pts.pose.orientation.x = qfinal.x();
    marker_pts.pose.orientation.y = qfinal.y();
    marker_pts.pose.orientation.z = qfinal.z();
    marker_pts.pose.orientation.w = qfinal.w();
    marker_pts.scale.x = height;
    marker_pts.scale.y = depth;
    marker_pts.scale.z = width;
    marker_pts.color.r = 1.0f;
    marker_pts.color.g = 1.0f;
    marker_pts.color.b = 0.0f;
    marker_pts.color.a = 0.5;
    marker_pts.lifetime = ros::Duration(0.2);
    
    markers.markers.push_back(marker_pts);


  }

};

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "object_detection");
  ros::NodeHandle nh;
  Cluster cluster(nh);


  ROS_INFO("Clustering Start");
  // Spin
  ros::spin ();
}
