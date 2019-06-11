
/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "wf_simulator/wfsim_model_forklift_rls.hpp"

WFSimModelIdealForkliftRLS::WFSimModelIdealForkliftRLS(double vx_lim, double angvel_lim, double wheelbase, double tread)
    : WFSimModelInterface(3 /* dim x */, 2 /* dim u */), vx_lim_(vx_lim), angvel_lim_(angvel_lim),
      wheelbase_(wheelbase), tread_(tread)
{

    ROS_INFO("running IDEAL_FORKLIFT_RLS model. \n parameters :");
    ROS_INFO("vx_lim : %f [m/s]", vx_lim_);
    ROS_INFO("angvel_lim : %f [rad/s]", angvel_lim_);
    ROS_INFO("wheelbase : %f [m]", wheelbase_);
    ROS_INFO("tread[ : %f [m]", tread_);
};

double WFSimModelIdealForkliftRLS::getX() { return state_(IDX::X); };
double WFSimModelIdealForkliftRLS::getY() { return state_(IDX::Y); };
double WFSimModelIdealForkliftRLS::getYaw() { return state_(IDX::YAW); };
double WFSimModelIdealForkliftRLS::getVx() { return input_(IDX_U::VX_DES); };
double WFSimModelIdealForkliftRLS::getWz() { return input_(IDX_U::WZ_DES); };
double WFSimModelIdealForkliftRLS::getSteer() 
{
    const double den = -2 * input_(IDX_U::VX_DES) + input_(IDX_U::WZ_DES) * tread_;
    const double steer = (input_(IDX_U::VX_DES) ==0 && input_(IDX_U::WZ_DES) == 0) ? 0 : std::atan(2 * input_(IDX_U::WZ_DES) * wheelbase_ / den);
    return (steer > std::atan(2 * wheelbase_ /tread_)) ? steer - std::asin(1) * 2 : steer;
};

Eigen::VectorXd WFSimModelIdealForkliftRLS::calcModel(const Eigen::VectorXd &state, const Eigen::VectorXd &input)
{
    const double yaw = state(IDX::YAW);
    const double vx = std::max(std::min(input_(IDX_U::VX_DES), vx_lim_), -vx_lim_);
    const double wz = std::max(std::min(input(IDX_U::WZ_DES), angvel_lim_), -angvel_lim_);

    Eigen::VectorXd d_state = Eigen::VectorXd::Zero(dim_x_);
    d_state(IDX::X) = vx * cos(yaw);
    d_state(IDX::Y) = vx * sin(yaw);
    d_state(IDX::YAW) = wz;

    return d_state;
};

WFSimModelTimeDelayForkliftRLS::WFSimModelTimeDelayForkliftRLS(double vx_lim, double steer_lim, double wheelbase, double tread, double dt,
                                   double vx_delay, double vx_time_constant, double steer_delay, double steer_time_constant)
    : WFSimModelInterface(5 /* dim x */, 2 /* dim u */), vx_lim_(vx_lim), steer_lim_(steer_lim), wheelbase_(wheelbase), tread_(tread), vx_delay_(vx_delay),
      vx_time_constant_(vx_time_constant), steer_delay_(steer_delay), steer_time_constant_(steer_time_constant)
{
    initializeInputQueue(dt);

    ROS_INFO("running DELAY_FORKLIFT_RLS model. \n parameters :");
    ROS_INFO("vx_lim : %f [m/s]", vx_lim_);
    ROS_INFO("steer_lim : %f [rad]", steer_lim_);
    ROS_INFO("wheelbase : %f [m]", wheelbase_);
    ROS_INFO("tread : %f [m]", tread_);
    ROS_INFO("dt : %f [s]", dt);
    ROS_INFO("vx_delay : %f [s]", vx_delay_);
    ROS_INFO("vx_time_constant : %f [s]", vx_time_constant_);
    ROS_INFO("steer_delay : %f [s]", steer_delay_);
    ROS_INFO("steer_time_constant : %f [s]", steer_time_constant_);
};

double WFSimModelTimeDelayForkliftRLS::getX() { return state_(IDX::X); };
double WFSimModelTimeDelayForkliftRLS::getY() { return state_(IDX::Y); };
double WFSimModelTimeDelayForkliftRLS::getYaw() { return state_(IDX::YAW); };
double WFSimModelTimeDelayForkliftRLS::getVx() { return state_(IDX::VX); };
double WFSimModelTimeDelayForkliftRLS::getWz() { return state_(IDX::VX) * convertSteerToCurvature(state_(IDX::STEER)); };
double WFSimModelTimeDelayForkliftRLS::getSteer() { return state_(IDX::STEER); };

void WFSimModelTimeDelayForkliftRLS::initializeInputQueue(const double &dt)
{
    size_t vx_input_queue_size = static_cast<size_t>(round(vx_delay_ / dt));
    for (size_t i = 0; i < vx_input_queue_size; i++)
    {
        vx_input_queue_.push_back(0.0);
    }
    size_t steer_input_queue_size = static_cast<size_t>(round(steer_delay_ / dt));
    for (size_t i = 0; i < steer_input_queue_size; i++)
    {
        steer_input_queue_.push_back(0.0);
    }
}

double WFSimModelTimeDelayForkliftRLS::convertSteerToCurvature(const double &steer)
{
    const double sign_steer = steer > 0.0 ? 1.0 : steer < 0.0 ? -1.0 : 0.0;
    const double den = std::max(2.0 * wheelbase_ - sign_steer * tread_ * std::tan(steer), 1.0E-8 /* avoid 0 divide */);
    return 2.0 * std::tan(steer) / den;
}

Eigen::VectorXd WFSimModelTimeDelayForkliftRLS::calcModel(const Eigen::VectorXd &state, const Eigen::VectorXd &input)
{
    const double delay_input_vx = vx_input_queue_.front();
    vx_input_queue_.pop_front();
    vx_input_queue_.push_back(input_(IDX_U::VX_DES));
    const double delay_input_steer = steer_input_queue_.front();
    steer_input_queue_.pop_front();
    steer_input_queue_.push_back(input_(IDX_U::STEER_DES));

    const double vx = state(IDX::VX);
    const double yaw = state(IDX::YAW);
    const double steer = state(IDX::STEER);
    const double delay_vx_des = std::max(std::min(delay_input_vx, vx_lim_), -vx_lim_);
    const double delay_steer_des = std::max(std::min(delay_input_steer, steer_lim_), -steer_lim_);

    Eigen::VectorXd d_state = Eigen::VectorXd::Zero(dim_x_);
    d_state(IDX::X) = vx * cos(yaw);
    d_state(IDX::Y) = vx * sin(yaw);
    d_state(IDX::YAW) = vx * convertSteerToCurvature(steer);
    d_state(IDX::VX) = -(vx - delay_vx_des) / vx_time_constant_;
    d_state(IDX::STEER) = -(steer - delay_steer_des) / steer_time_constant_;

    return d_state;
};