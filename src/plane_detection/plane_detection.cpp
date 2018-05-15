#include <ros/ros.h>
#include <quadrobot_msgs/Plane.h>
#include <quadrobot_msgs/PlaneArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
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

struct Plane
{
  double width;
  double depth;
  double height;
  pcl::PointXYZ pts[9];
  Eigen::Vector3f normal;
};

class Cluster
{
  private:
  ros::NodeHandle nh_;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber cloud_sub_;
  ros::Publisher pub_cluster,pub_marker,pub_plane,cloud_pub_;
  
  public:
  //  Init params
  double val_pass=3.0, val_vox=0.015f;
  double val_radius=0.1, val_minpt = 20, val_dist=0.5;
  double SmoothnessThresholdDegree = 15.0; // angle/(pi/2)
  double NormalDistanceWeight = SmoothnessThresholdDegree / 90.0;
  double DistanceThreshold = 0.2;
  double x_max,x_min,y_max,y_min,z_max,z_min;
  int MinClusterSize = 200;
  int minNeighbor = 20;
  bool doFILTER = true;

  public:
  Cluster(ros::NodeHandle nh)
  : nh_(nh)
  {      
    // Get params
    nh.getParam("/plane_detection/doFILTER", doFILTER);
    nh.getParam("/plane_detection/val_pass", val_pass);
    nh.getParam("/plane_detection/val_vox", val_vox);
    nh.getParam("/plane_detection/val_minpt", val_minpt);
    nh.getParam("/plane_detection/val_dist", val_dist);
    nh.getParam("/plane_detection/val_radius", val_radius);
    nh.getParam("/plane_detection/SmoothnessThresholdDegree", SmoothnessThresholdDegree);
    nh.getParam("/plane_detection/DistanceThreshold", DistanceThreshold);
    nh.getParam("/plane_detection/NormalDistanceWeight", NormalDistanceWeight);
    nh.getParam("/plane_detection/MinClusterSize", MinClusterSize);
    nh.getParam("/plane_detection/minNeighbor", minNeighbor);
    nh.getParam("/plane_detection/x_min", x_min);
    nh.getParam("/plane_detection/x_max", x_max);
    nh.getParam("/plane_detection/y_min", y_min);
    nh.getParam("/plane_detection/y_max", y_max);
    nh.getParam("/plane_detection/z_min", z_min);
    nh.getParam("/plane_detection/z_max", z_max);

    // Topics
    std::string point_cloud_topic_ = "/camera/depth/points";
    nh.getParam("point_cloud_topic", point_cloud_topic_);

    
    // Create ROS sub & publisher of point clouds
    cloud_sub_ = nh_.subscribe (point_cloud_topic_, 1, &Cluster::cloud_cb,this);
    cloud_pub_ = nh_.advertise<pcl::PointCloud<PointT>>("/cluster/output_cloud",1);
            
    // Create ROS publisher for the output model coefficients
    pub_cluster = nh.advertise<sensor_msgs::PointCloud2> ("/plane/clusters", 100);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray> ("/plane/markers", 10);
    pub_plane = nh.advertise<quadrobot_msgs::PlaneArray> ("/plane/info_array", 10);


  }
  ~Cluster()
  {

  }
  void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
  {
    iter++;
    std::string cloudname="/home/jjm/workspace/ROS/quadrobot_ws/";
    std::stringstream ss;
    ss << iter;
    cloudname += "cloud_" + ss.str() + ".pcd";
  
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // 
    pcl::fromROSMsg (*input, *cloud); 
    cloud_pub_.publish(cloud); 
       
    ROS_INFO("%s", cloudname.c_str());
    //pcl::io::savePCDFileBinaryCompressed(cloudname, *cloud);

    // All the objects needed
    pcl::PCDReader reader;
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    // Datasets
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    
    pcl::PointCloud<PointT>::Ptr cloud_voxel(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_noise(new pcl::PointCloud<PointT>);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    pcl::PointCloud<PointT>::Ptr cloud_rejected(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normal_rejected(new pcl::PointCloud<pcl::Normal>);

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

  // Estimate point normals
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_filtered);
  ne.setKSearch(50);
  ne.compute(*cloud_normals);

  int i = 0, nr_points = (int)cloud_filtered->points.size();
  int plane_found = 0;
  int reg_id = 0;
  int csize = 0;
  //std::cout << "number of points =" << nr_points << std::endl;

  int k = 0;
  visualization_msgs::MarkerArray markers;
  quadrobot_msgs::PlaneArray planes;

  while (cloud_filtered->points.size()> MinClusterSize)
  {
    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(NormalDistanceWeight);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(DistanceThreshold);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    // std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);

    if (inliers_plane->indices.size() == 0)
    {
      //std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    //    showCoefficients(coefficients_plane,region_type);
    //    std::cerr << "Model inliers: " << inliers_plane->indices.size () << std::endl;
    //    int accepted=writeToFile(region_type,outfilename,cloud_filtered,inliers_plane,coefficients_plane,min_region_size);

    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_plane);

    bool haveMinSize = true;
    csize = (int)(inliers_plane->indices.size());

    if (csize<MinClusterSize)
      haveMinSize = false;
    if (haveMinSize == false /*accepted==-1*/)
    {
      // add rejected region to rejected point cloud and normals
      (*cloud_rejected) += (*cloud_plane);
      extract_normals.setNegative(false);
      extract_normals.setInputCloud(cloud_normals);
      extract_normals.setIndices(inliers_plane);
      extract_normals.filter(*cloud_normals2);
      (*cloud_normal_rejected) += (*cloud_normals2);
    }
    else
    {
      plane_found++;
      addPointColoudToColouredPointCloud(cloud_plane, coloured_point_cloud, plane_found, false);
      getPlaneModel(cloud_plane, coefficients_plane,plane_found, planes, markers);
    }    

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals2);

    cloud_filtered.swap(cloud_filtered2);
    cloud_normals.swap(cloud_normals2);
    //    }
  } // while
  std::cerr << "Num of planes ="<<plane_found << std::endl;

  if (cloud_filtered->points.empty())
  {
    //std::cerr << "no more points remain." << std::endl;
  }
  else
  {
    //addPointColoudToColouredPointCloud(cloud_filtered, coloured_point_cloud, 0, false);
  }
  //ROS_INFO("Num of cluster=%d", plane_found );

    // Convert To ROS data type
  sensor_msgs::PointCloud2 output;
  pcl::PCLPointCloud2 cloud_p;
  pcl::toPCLPointCloud2(coloured_point_cloud, cloud_p);

  pcl_conversions::fromPCL(cloud_p, output);
  output.header.frame_id = "camera_depth_optical_frame";
  pub_cluster.publish(output);


  // Publish markers
  pub_marker.publish(markers);

  // Publish plane info
  pub_plane.publish(planes);
  }

  void addPointColoudToColouredPointCloud(pcl::PointCloud<PointT>::Ptr cloud_plane, pcl::PointCloud<pcl::PointXYZI> &coloured_point_cloud, int idx, bool tmp)
{
  for (int i=0;i<cloud_plane->size();i++)
  {
    pcl::PointXYZ pt = cloud_plane->points[i];
    pcl::PointXYZI pt2;
    pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z;
    pt2.intensity = float(idx);

    coloured_point_cloud.push_back(pt2);
  }
}
int getPlaneModel(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::ModelCoefficients::Ptr coefficients_plane, int idx, quadrobot_msgs::PlaneArray &planes, visualization_msgs::MarkerArray &markers)
{

    // compute principal direction
    Eigen::Vector4f centroid;
    Eigen::Matrix3f covariance;
    pcl::compute3DCentroid(*point_cloud_ptr, centroid);
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


    double width = max_pt.z-min_pt.z; // longest line of cube
    double height= max_pt.y-min_pt.y; 
    double depth = max_pt.x-min_pt.x; // shortest line of cube

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_ptr->resize(5);
    cloud_ptr->points[0] = pcl::PointXYZ(0, max_pt.y, max_pt.z);		// corner
    cloud_ptr->points[1] = pcl::PointXYZ(0, max_pt.y, min_pt.z);
    cloud_ptr->points[2] = pcl::PointXYZ(0, min_pt.y, max_pt.z);
    cloud_ptr->points[3] = pcl::PointXYZ(0, min_pt.y, min_pt.z);
    cloud_ptr->points[4] = pcl::PointXYZ(0.0, 0.0, 0.0);					  // center
    pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, w2p);

    

    ////////////////////////////////////   marker setting
    visualization_msgs::Marker marker_pt,marker_pts,marker_vector,marker_text;
    
    // Marker for corner points
    marker_pts.header.frame_id = "camera_depth_optical_frame";
    marker_pts.header.stamp = ros::Time::now();
    geometry_msgs::Point pt[5];
    for(int i=0; i<5;i++)
    {
      pt[i].x=cloud_ptr->points[i].x/1.0f;  
      pt[i].y=cloud_ptr->points[i].y/1.0f;
      pt[i].z=cloud_ptr->points[i].z/1.0f;
      if(i!=4)
            marker_pts.points.push_back(pt[i]);
    }
    marker_pts.ns = "corner_points";
    marker_pts.id = idx;
    marker_pts.type = visualization_msgs::Marker::POINTS;
    marker_pts.action = visualization_msgs::Marker::ADD;
    marker_pts.pose.position.x = 0.0;
    marker_pts.pose.position.y = 0.0;
    marker_pts.pose.position.z = 0.0;
    marker_pts.pose.orientation.x = 0.0;
    marker_pts.pose.orientation.y = 0.0;
    marker_pts.pose.orientation.z = 0.0;
    marker_pts.pose.orientation.w = 1.0;
    marker_pts.scale.x = 0.01f;
    marker_pts.scale.y = 0.01f;
    marker_pts.scale.z = 0.01f;
    marker_pts.color.r = 1.0f;
    marker_pts.color.g = 1.0f;
    marker_pts.color.b = 0.0f;
    marker_pts.color.a = 1.0;
    marker_pts.lifetime = ros::Duration(0.2);
    
    markers.markers.push_back(marker_pts);

    // Marker for center point
    marker_pt.header.frame_id = "camera_depth_optical_frame";
    marker_pt.header.stamp = ros::Time::now();
    marker_pt.ns = "center_point";
    marker_pt.id = idx*100;
    marker_pt.type = visualization_msgs::Marker::SPHERE;
    marker_pt.action = visualization_msgs::Marker::ADD;
    marker_pt.pose.position.x = pt[4].x;
    marker_pt.pose.position.y = pt[4].y;
    marker_pt.pose.position.z = pt[4].z;
    marker_pt.pose.orientation.x = 0.0;
    marker_pt.pose.orientation.y = 0.0;
    marker_pt.pose.orientation.z = 0.0;
    marker_pt.pose.orientation.w = 1.0;
    marker_pt.scale.x = 0.01f;
    marker_pt.scale.y = 0.01f;
    marker_pt.scale.z = 0.01f;
    marker_pt.color.r = 1.0f;
    marker_pt.color.g = 1.0f;
    marker_pt.color.b = 0.0f;
    marker_pt.color.a = 1.0;
    marker_pt.lifetime = ros::Duration(0.2);

    markers.markers.push_back(marker_pt);

    // Marker for normal vector
    geometry_msgs::Point end_pt;
    end_pt.x =pt[4].x-coefficients_plane->values[0]/10.0;
    end_pt.y =pt[4].y-coefficients_plane->values[1]/10.0;
    end_pt.z =pt[4].z-coefficients_plane->values[2]/10.0;

    marker_vector.header.frame_id = "camera_depth_optical_frame";
    marker_vector.header.stamp = ros::Time::now();
    marker_vector.ns = "normal_vector";
    marker_vector.id = idx*1000;
    marker_vector.type = visualization_msgs::Marker::ARROW;
    marker_vector.action = visualization_msgs::Marker::ADD;
    marker_vector.pose.position.x = 0.0;
    marker_vector.pose.position.y = 0.0;
    marker_vector.pose.position.z = 0.0;
    marker_vector.pose.orientation.x = 0.0;
    marker_vector.pose.orientation.y = 0.0;
    marker_vector.pose.orientation.z = 0.0;
    marker_vector.pose.orientation.w = 1.0;
    marker_vector.scale.x = 0.01f;
    marker_vector.scale.y = 0.01f;
    marker_vector.scale.z = 0.01f;
    marker_vector.color.r = 1.0f;
    marker_vector.color.g = 1.0f;
    marker_vector.color.b = 0.0f;
    marker_vector.color.a = 1.0;
    marker_vector.lifetime = ros::Duration(0.2);
    marker_vector.points.push_back(pt[4]);
    marker_vector.points.push_back(end_pt);

    markers.markers.push_back(marker_vector);

    // Marker for text visualization
    std::string msg="";
    std::stringstream ss;
    ss << width;
    msg += "Width: " + ss.str() + " m\n";
    ss.str("");
    ss << height;
    msg += "Height: " + ss.str() + " m";


    marker_text.header.frame_id = "camera_depth_optical_frame";
    marker_text.header.stamp = ros::Time::now();
    marker_text.ns = "text";
    marker_text.id = idx*10000;
    marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_text.action = visualization_msgs::Marker::ADD;
    marker_text.pose.position.x = end_pt.x;
    marker_text.pose.position.y = end_pt.y;
    marker_text.pose.position.z = end_pt.z;
    marker_text.pose.orientation.x = 0.0;
    marker_text.pose.orientation.y = 0.0;
    marker_text.pose.orientation.z = 0.0;
    marker_text.pose.orientation.w = 1.0;
    marker_text.scale.x = 0.05f;
    marker_text.scale.y = 0.05f;
    marker_text.scale.z = 0.05f;
    marker_text.color.r = 0.0f;
    marker_text.color.g = 0.0f;
    marker_text.color.b = 0.0f;
    marker_text.color.a = 1.0;
    marker_text.lifetime = ros::Duration(0.2);
    marker_text.text = msg;
    markers.markers.push_back(marker_text);

    ////////////////////////////////////  set plane model
    quadrobot_msgs::Plane plane;
    plane.header.frame_id = "camera_depth_optical_frame";
    plane.header.stamp = ros::Time::now();
    plane.width = width;
    plane.height = height;
    plane.depth = depth;
    
    for(int i=0;i<4;i++)
    {
      plane.coef[i] = coefficients_plane->values[i];
    }
    for(int i=0;i<4;i++)
    {
      plane.corners[i] = pt[i];
    }
    plane.center = pt[4];

    planes.planes.push_back(plane);

    return(0);
  }
};

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "plane_detection");
  ros::NodeHandle nh;
  Cluster cluster(nh);


  ROS_INFO("Clustering Start");
  // Spin
  ros::spin ();
}
