// This file is part of snark, a generic and flexible library for robotics research
// Copyright (c) 2011 The University of Sydney
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the The University of Sydney.
// 4. Neither the name of the The University of Sydney nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
// IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "commands_handler.h"
#include "auto_initialization.h"
#include <fstream>

namespace snark { namespace ur { namespace robotic_arm { namespace handlers {

static const char* name() {
    return "robot-arm-daemon: ";
}

void commands_handler::handle( arm::power& p )
{
    if( p.is_on && !status_.is_powered_off() ) {
        ret = result( "cannot execute power on command as current state is not 'no_power'", result::error::invalid_robot_state );
        return;
    }
    std::cerr << name() << "powering robot arm " << ( p.is_on ? "on" : "off" ) << std::endl;
    os << "power " << ( p.is_on ? "on" : "off" ) << std::endl;
    os.flush();
    ret = result();
}

void commands_handler::handle( arm::brakes& b )
{
    std::cerr << name() << "running brakes: " << b.enable  << std::endl;
    if( !b.enable ) {
        os << "set robotmode run" <<std::endl;
    }
    else {
        os << "stopj([0.1,0.1,0.1,0.1,0.1,0.1])" << std::endl;
    }
    os.flush();
    ret = result();
}
static const plane_angle_degrees_t max_pan = 90.0 * degree;
static const plane_angle_degrees_t min_pan = -90.0 * degree;
static const plane_angle_degrees_t max_tilt = 90.0 * degree;
static const plane_angle_degrees_t min_tilt = -90.0 * degree;

void commands_handler::handle( arm::move_cam& cam )
{
    std::cerr << name() << " running move_cam" << std::endl; 

    if( !status_.is_running() ) {
        ret = result( "cannot move (camera) as rover is not in running mode", result::error::invalid_robot_state );
        return;
    }

    static const length_t min_height = 0.1 * meter;
    static const length_t max_height = 1.0 * meter;
    
    
    if( cam.pan < min_pan ) { ret = result( "pan angle is below minimum limit of -45.0", result::error::invalid_input ); return; }
    if( cam.pan > max_pan ) { ret = result( "pan angle is above minimum limit of 45.0", result::error::invalid_input ); return; }
    if( cam.tilt < min_tilt ) { ret = result( "tilt angle is below minimum limit of -90.0", result::error::invalid_input ); return; }
    if( cam.tilt > max_tilt ) { ret = result( "tilt angle is above minimum limit of 90.0", result::error::invalid_input ); return; }
    if( cam.height < min_height ) { ret = result( "height value is below minimum limit of 0.1m", result::error::invalid_input ); return; }
    if( cam.height > max_height ) { ret = result( "height value is above minimum limit of 1.0m", result::error::invalid_input ); return; }
    
    inputs_.motion_primitive = real_T( input_primitive::move_cam );
    inputs_.Input_1 = cam.pan.value();
    // inputs_.Input_2 = cam.height.value() != 1.0 ? -cam.tilt.value() : zero_tilt - cam.tilt.value();
    inputs_.Input_2 = cam.tilt.value();
    inputs_.Input_3 = cam.height.value();
    
    fs::remove( home_filepath_ );
    
    move_cam_height_ = cam.height;
    ret = result();
}

void commands_handler::handle(sweep_cam& s)
{
    std::cerr << name() << " running sweep_cam" << std::endl; 

    if( !status_.is_running() ) { ret = result( "cannot sweep (camera) as rover is not in running mode", result::error::invalid_robot_state ); return; }
    if( !move_cam_height_ ) { ret = result( "robotic_arm is not in move_cam position", result::error::invalid_robot_state ); return; }
    
    ret = sweep_.run( *move_cam_height_, this->os );
}


void commands_handler::handle( arm::move_joints& joints )
{
//     if( !status_.is_running() ) {
//         ret = result( "cannot move (joints) as rover is not in running mode", result::error::invalid_robot_state );
//         return;
//     }
// 
//     std::cerr << name() << " running move joints"  << std::endl; 
//     static const plane_angle_degrees_t min = 0.0 * degree;
//     static const plane_angle_degrees_t max = 360.0 * degree;
//     for( std::size_t i=0; i<joints.joints.size(); ++i )
//     {
//         if( joints.joints[i] < min || joints.joints[0] > max ) { 
//             ret = result( "joint angle must be 0-360 degrees", result::error::invalid_input ); 
//             return;
//         }
//     }
//     
//     inputs_.motion_primitive = real_T( input_primitive::movej );
//     inputs_.Input_1 = joints.joints[0].value();
//     inputs_.Input_2 = joints.joints[1].value();
//     inputs_.Input_3 = joints.joints[2].value();
//     inputs_.Input_4 = joints.joints[3].value();
//     inputs_.Input_5 = joints.joints[4].value();
//     inputs_.Input_6 = joints.joints[5].value();
    ret = result();
}

void commands_handler::handle( arm::joint_move& joint )
{
    if( !is_initialising() ) {
        ret = result( "cannot initialise joint as rover is not in initialisation mode", result::error::invalid_robot_state );
        return;
    }
    move_cam_height_.reset(); // no longer in move_cam position
    
    /// command can be use if in running or initialising mode
    int index = joint.joint_id;
    if( status_.robot_mode != robotmode::initializing && 
        status_.joint_modes[index] != jointmode::initializing )
    { 
        std::ostringstream ss;
        ss << "robot and  joint (" << index << ") must be initializing state. However current robot mode is '" 
           << status_.mode_str() << "' and joint mode is '" << status_.jmode_str(index)  << '\'' << std::endl;
        ret = result( ss.str(), result::error::invalid_robot_state );
        return; 
    }

    static const unsigned char min_id = 0;
    static const unsigned char max_id = 5;
    std::cerr << name() << "move joint: " << int(joint.joint_id) << " dir: " << joint.dir << std::endl;
    static const angular_velocity_t velocity = 0.1 * rad_per_sec;
    static const angular_acceleration_t acceleration = 0.05 * rad_per_s2;
    static const boost::posix_time::time_duration duration = boost::posix_time::milliseconds( 20 );
    
    if( joint.joint_id < min_id || joint.joint_id > max_id ) {
        ret = result( "joint id must be 0-5", result::error::invalid_input );
        return;
    }
    
    double vel = ( joint.dir ? velocity.value() : -velocity.value() );
    if( joint.joint_id == 2 ) { vel /= 2; }
    else if( joint.joint_id == 1 ) { vel /= 2.5; }
    else if( joint.joint_id < 1 ) { vel /= 3; }
    
    std::ostringstream ss;
    ss << "speedj_init([";
    for( std::size_t i=min_id; i<=max_id; ++i )
    {
        ss << (i == joint.joint_id ? vel : 0);
        if( i != max_id ) { ss << ','; };
    }
    ss << "],"  << acceleration.value() << ',' << (duration.total_milliseconds()/1000.0) << ')' << std::endl;
    os << ss.str();
    os.flush(); 
    
    fs::remove( home_filepath_ );
    
    ret = result();
}

void commands_handler::handle( arm::set_home& h )
{
    inputs_.motion_primitive = input_primitive::set_home;
    ret = result();
}

void commands_handler::handle( arm::set_position& pos )
{
    if( !status_.is_running() ) {
        ret = result( "cannot set position as rover is not in running mode", result::error::invalid_robot_state );
        return;
    }
    
    move_cam_height_.reset(); // no longer in move_cam position

    inputs_.motion_primitive = input_primitive::set_position;

    if( pos.position == "giraffe" ) { inputs_.Input_1 = set_position::giraffe; fs::remove( home_filepath_ ); }
    else if( pos.position == "home" ) { inputs_.Input_1 = set_position::home; }
    else { ret = result("unknown position type", int(result::error::invalid_input) ); return; }

    inputs_.Input_2 = 0;    // zero pan for giraffe
    inputs_.Input_3 = 0;    // zero tilt for giraffe
    
    ret = result();
}


void commands_handler::handle( arm::move_effector& e )
{
}


bool commands_handler::is_initialising() const 
{
    if( status_.robot_mode != robotmode::initializing ) { 
        // std::cerr << "robot mode " << status_.robot_mode << " expected: " << robotmode::initializing << std::endl;
        return false; 
    }

    for( std::size_t i=0; i<joints_num; ++i ) 
    {
        // std::cerr << "joint " << i << " mode " << status_.joint_modes[i] << " expected: " << jointmode::running << std::endl;
        if( status_.jmode(i) != jointmode::initializing && 
            status_.jmode(i) != jointmode::running ) { 
            return false; 
        }
    }

    return true;
}

void commands_handler::handle(auto_init& a)
{
    ret = init_.run( false );
    if( ret.is_success() ) { 
        std::cerr << name() << "going to home position." << std::endl;
        arm::set_position home; handle( home ); 
    } // set to home

}
void commands_handler::handle( arm::auto_init_force& init )
{
    ret = init_.run( init.force );
    if( ret.is_success() ) { 
        std::cerr << name() << "going to home position." << std::endl;
        arm::set_position home; handle( home ); 
    } // set to home
}


} } } } // namespace snark { namespace ur { namespace robotic_arm { namespace handlers {
