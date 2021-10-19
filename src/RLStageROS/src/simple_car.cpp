#include <stage.hh>
#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <dqn_stage_ros/stage_message.h>
#include <geometry_msgs/Twist.h>
#include <std_srvs/Empty.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace Stg;

struct ModelRobot
{
  ModelPosition* pos;
  ModelRanger* laser;
  Pose resetPose;
};

ModelRobot* robot;
usec_t stgSpeedTime;

ros::NodeHandle* n;
ros::Publisher pub_state_, pub_reset_pose_;
ros::Subscriber sub_vel_;
ros::ServiceServer reset_srv_;

geometry_msgs::PoseStamped rosCurPose;
sensor_msgs::LaserScan rosLaserData;
bool collision = false;
bool allowNewMsg = true;
double minFrontDist;
ros::Time lastSentTime;

float reset_x_set [] = { 4, 5, 8, 10,  15,  17};
float reset_y_set [] = { 3, 4, 2, 2, 1.5, 2};
float reset_angle [] = {0., 0.1, -0.1, 0, 0.2, 0.15};

void stgPoseUpdateCB( Model* mod, ModelRobot* robot)
{
  geometry_msgs::PoseStamped positionMsg;
  positionMsg.pose.position.x = robot->pos->GetPose().x;
  positionMsg.pose.position.y = robot->pos->GetPose().y;
  positionMsg.pose.position.z = robot->pos->GetPose().z;
  positionMsg.pose.orientation = tf::createQuaternionMsgFromYaw( robot->pos->GetPose().a);
  positionMsg.header.stamp = ros::Time::now();
  rosCurPose = positionMsg;

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(2.0 - robot->resetPose.x, 2.0 - robot->resetPose.y, 0.0) );
  tf::Quaternion q;
  q.setRPY(0, 0, 0);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "slam_map", "map"));
}

void stgLaserCB( Model* mod, ModelRobot* robot)
{
  sensor_msgs::LaserScan laserMsgs;
  const Stg::ModelRanger::Sensor& sensor = robot->laser->GetSensors()[0];
  double minDist = sensor.range.max;
  if( sensor.ranges.size() )
    {
      // Translate into ROS message format and publish
      laserMsgs.angle_min = -sensor.fov/2.0;
      laserMsgs.angle_max = +sensor.fov/2.0;
      laserMsgs.angle_increment = sensor.fov/(double)(sensor.sample_count-1);
      laserMsgs.range_min = sensor.range.min;
      laserMsgs.range_max = sensor.range.max;
      laserMsgs.ranges.resize(sensor.ranges.size());
      laserMsgs.intensities.resize(sensor.intensities.size());

      collision = false;
      minFrontDist = sensor.range.max;
      // added by sepehr for random position init:
      //        double min_laser_val = 99;
      for(unsigned int i = 0; i < sensor.ranges.size(); i++)
        {
          laserMsgs.ranges[i] = sensor.ranges[i];
          if(sensor.ranges[i] < 0.45)
            collision = true;
          if( i > (sensor.fov*180.0/M_PI - 45)/2 && i < (sensor.fov*180.0/M_PI + 45)/2 && sensor.ranges[i]  < minFrontDist)
            minFrontDist = sensor.ranges[i];
          if( sensor.ranges[i] < minDist)
            minDist = sensor.ranges[i];
          //            if(sensor.ranges[i] < min_laser_val)
          //                min_laser_val = sensor.ranges[i];
          laserMsgs.intensities[i] = sensor.intensities[i];
        }

      //        if( min_laser_val > 3.3 && rand()/float(RAND_MAX) > 0.1)
      //        {
      //            initial_poses.clear();
      //            initial_poses.push_back(robotmodel->positionmodel->GetGlobalPose());
      //        }
      laserMsgs.header.stamp = ros::Time::now();
      rosLaserData = laserMsgs;
    }

  if( robot->pos->GetWorld()->SimTimeNow() - stgSpeedTime > 100000)
    robot->pos->SetSpeed( 0, 0, 0);


  //temp, just to check publish rate:
  if( allowNewMsg
      && laserMsgs.header.stamp > lastSentTime
      && rosCurPose.header.stamp > lastSentTime)
    {
      allowNewMsg = false;
      dqn_stage_ros::stage_message msg;
      msg.header.stamp = ros::Time::now();
      msg.collision = collision;
      msg.minFrontDist = minFrontDist;
      msg.position = rosCurPose;
      msg.laser = rosLaserData;
      pub_state_.publish( msg);
    }

  geometry_msgs::PoseStamped reset_pose;
  reset_pose.pose.position.x = robot->pos->GetPose().x;
  reset_pose.pose.position.y = robot->pos->GetPose().y;
  reset_pose.pose.position.z = robot->pos->GetPose().z;
  reset_pose.pose.orientation = tf::createQuaternionMsgFromYaw(0);
  reset_pose.header.stamp = ros::Time::now();
  pub_reset_pose_.publish(reset_pose);
}

void rosVelocityCB( const geometry_msgs::TwistConstPtr vel)
{
// ROS_WARN("Vel recieved");
  robot->pos->SetXSpeed( vel->linear.x);
  robot->pos->SetTurnSpeed( vel->angular.z);
  lastSentTime = ros::Time::now();
  stgSpeedTime = robot->pos->GetWorld()->SimTimeNow();
  allowNewMsg = true;
}

bool rosResetSrvCB(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  ROS_INFO("Resetting stage!");
  srand (time(NULL));
  int idx = rand() % 5;
  robot->resetPose.x = reset_x_set[idx];
  robot->resetPose.y = reset_y_set[idx];
  robot->pos->SetPose( robot->resetPose);
  Pose reset_odom;
  reset_odom.x = 0;
  reset_odom.y = 0;
  reset_odom.z = 0;
  robot->pos->SetOdom(reset_odom);
  return true;
}


extern "C" int Init( Model* mod )
{ 
  int argc = 0;
  char** argv;
  ros::init( argc, argv, "target_controller_node");
  n = new ros::NodeHandle();
  lastSentTime = ros::Time::now();
  pub_state_ = n->advertise<dqn_stage_ros::stage_message>("input_data", 15);
  pub_reset_pose_ = n->advertise<geometry_msgs::PoseStamped>("/reset_pose", 10);
  sub_vel_ = n->subscribe( "cmd_vel", 15, &rosVelocityCB);
  reset_srv_ = n->advertiseService("/stage/reset_simulation", &rosResetSrvCB);
  robot = new ModelRobot;
  robot->pos = (ModelPosition*) mod;
  robot->pos->AddCallback( Model::CB_UPDATE, (model_callback_t)stgPoseUpdateCB, robot);
  robot->pos->Subscribe();
  robot->resetPose = robot->pos->GetPose();
  //    robot->pos->GetChild("ranger:0")->Subscribe();
  robot->laser = (ModelRanger*)mod->GetChild("ranger:0");
  robot->laser->AddCallback( Model::CB_UPDATE, (model_callback_t)stgLaserCB, robot);
  robot->laser->Subscribe();
  return 0; //ok
}

