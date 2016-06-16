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
// 3. Neither the name of the University of Sydney nor the
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

#include <cmath>
#include <boost/array.hpp>
#include <boost/graph/graph_concepts.hpp>
#include "calculator.h"
#include "packet.h"
#include "../../las/packets.h"

namespace snark { namespace velodyne { namespace puck {

static boost::array< double, 16 > elevation_ = {{ -15.0 * M_PI / 180
                                                , 1 * M_PI / 180
                                                , -13 * M_PI / 180
                                                , -3 * M_PI / 180
                                                , -11 * M_PI / 180
                                                , 5 * M_PI / 180
                                                , -9 * M_PI / 180
                                                , 7 * M_PI / 180
                                                , -7 * M_PI / 180
                                                , 9 * M_PI / 180
                                                , -5 * M_PI / 180
                                                , 11 * M_PI / 180
                                                , -3 * M_PI / 180
                                                , 13 * M_PI / 180
                                                , -1 * M_PI / 180
                                                , 15 * M_PI / 180 }};

namespace timing {
    
static double firing_interval = ( 2.304 / 1000000 );

static double recharge_interval = ( 18.43 / 1000000 );

} // namespace timing {

struct laser
{
    double sin;
    double cos;
    
    laser() {}
    laser( unsigned int index )
        : sin( std::sin( elevation_[ index % puck::packet::number_of_lasers ] ) )
        , cos( std::cos( elevation_[ index % puck::packet::number_of_lasers ] ) )
    {
    }
};

struct channel
{
    double single_mode_delay;
    double dual_mode_delay;
    
    channel() {}
    channel( unsigned int block, unsigned int index )
        : single_mode_delay( timing::firing_interval * index + timing::recharge_interval * ( index / puck::packet::number_of_lasers ) + timing::recharge_interval * block * 2 )
        , dual_mode_delay( timing::firing_interval * ( index % puck::packet::number_of_lasers ) + timing::recharge_interval * block )
    {
    }
};

typedef boost::array< boost::array< channel, puck::packet::number_of_lasers * 2 >, puck::packet::number_of_blocks > channels_t;

static channels_t init_channels()
{
    channels_t channels;
    for( unsigned int i = 0; i < puck::packet::number_of_blocks; ++i )
    {
        for( unsigned int j = 0; j < puck::packet::number_of_lasers * 2; ++j ) { channels[i][j] = channel( i, j ); }
    }
    return channels;
}

typedef boost::array< laser, puck::packet::number_of_lasers > lasers_t;

static lasers_t init_lasers()
{
    lasers_t lasers;
    for( unsigned int j = 0; j < puck::packet::number_of_lasers; ++j ) { lasers[j] = laser( j ); }
    return lasers;
}

static channels_t channels = init_channels();
static lasers_t lasers = init_lasers();

std::pair< ::Eigen::Vector3d, ::Eigen::Vector3d > calculator::ray( unsigned int laser, double range, double angle ) const { return std::make_pair( ::Eigen::Vector3d::Zero(), point( laser, range, angle ) ); }

::Eigen::Vector3d calculator::point( unsigned int laser, double range, double angle ) const
{
    laser %= 16; // todo: fix it!
    return ::Eigen::Vector3d( range * lasers[laser].cos * std::sin( angle ), range * lasers[laser].cos * std::cos( angle ), range * lasers[laser].sin );
}

double calculator::range( unsigned int, double range ) const { return range; }

double calculator::azimuth( unsigned int, double azimuth ) const { return azimuth; }

double calculator::intensity( unsigned int, unsigned char intensity, double ) const { return intensity; }

} } } // namespace snark { namespace velodyne { namespace puck {
