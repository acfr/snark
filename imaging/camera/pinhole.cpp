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

#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/bind.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <comma/application/verbose.h>
#include <comma/base/exception.h>
#include <comma/base/types.h>
#include <comma/math/compare.h>
#include "pinhole.h"

namespace snark { namespace camera {
    
pinhole::config_t::config_t() : focal_length( 0 ), image_size( Eigen::Vector2i::Zero() ) {}

void pinhole::config_t::validate()
{
    if( !comma::math::less( 0, focal_length ) ) { COMMA_THROW( comma::exception, "pinhole config: expected positive focal length, got: " << focal_length ); }
    if( sensor_size && ( !comma::math::less( 0, sensor_size->x() ) || !comma::math::less( 0, sensor_size->y() ) ) ) { COMMA_THROW( comma::exception, "pinhole config: expected positive sensor size, got: " << sensor_size->x() << "," << sensor_size->y() ); }
    if( image_size.x() <= 0 || image_size.y() <= 0 ) { COMMA_THROW( comma::exception, "pinhole config: expected positive image size, got: " << image_size.x() << "," << image_size.y() ); }
}

template < typename V > V pinhole::config_t::distortion_t::as() const
{ 
    V v;
    v[0] = radial.k1;
    v[1] = radial.k2;
    v[2] = tangential.p1;
    v[3] = tangential.p2;
    v[4] = radial.k3;
    return v;
}

template Eigen::Matrix< double, 5, 1 > pinhole::config_t::distortion_t::as< Eigen::Matrix< double, 5, 1 > >() const;

template cv::Vec< double, 5 > pinhole::config_t::distortion_t::as< cv::Vec< double, 5 > >() const;

bool pinhole::config_t::distortion_t::radial_t::empty() const { return k1 == 0 && k2 == 0 && k3 == 0; }

bool pinhole::config_t::distortion_t::tangential_t::empty() const { return p1 == 0 && p2 == 0; }

bool pinhole::config_t::distortion_t::empty() const { return map_filename.empty() && radial.empty() && tangential.empty(); }

//quick and dirty: if sensor_size is empty we are taking pixel size to be 1 meter !?
Eigen::Vector2d pinhole::config_t::pixel_size() const { return sensor_size ? Eigen::Vector2d( sensor_size->x() / image_size.x(), sensor_size->y() / image_size.y() ) : Eigen::Vector2d( 1, 1 ); }

static double squared_radius( const Eigen::Vector2d& p, const snark::camera::pinhole::config_t& c )
{
    return ( p - c.image_centre() ).squaredNorm();
}

Eigen::Vector2d pinhole::config_t::radially_corrected( const Eigen::Vector2d& p ) const
{
    if( !distortion ) { return p; }
    double r2 = squared_radius( p, *this );
    double k = 1 + distortion->radial.k1 * r2 + distortion->radial.k2 * r2 * r2 + distortion->radial.k3 * r2 * r2 * r2;
    return p * k;
}

Eigen::Vector2d pinhole::config_t::tangentially_corrected( const Eigen::Vector2d& p ) const
{
    if( !distortion ) { return p; }
    double r2 = squared_radius( p, *this );
    double xy = p.x() * p.y();
    return p + Eigen::Vector2d( distortion->tangential.p1 * 2 * xy + distortion->tangential.p2 * ( r2 + p.x() * p.x() * 2 )
                              , distortion->tangential.p2 * 2 * xy + distortion->tangential.p1 * ( r2 + p.y() * p.y() * 2 ) );
}

static double reverse_map( std::vector< float >::const_iterator first, unsigned int width, double value )
{
    std::vector< float >::const_iterator last = first + width;
    std::vector< float >::const_iterator right = std::upper_bound( first, last, value );
    if( right == last ) { --right; }
    if( right == first ) { ++right; }
    std::vector< float >::const_iterator left = right - 1;
    //std::cerr<<"extrapolate_reverse_map: left [" <<int(left - first)<<"]: " << *left<< " right ["<<int(right - first)<<"]:"<<*right<<" /"<<int(last-first)<<" value "<<value<<std::endl;
    double diff = *right - *left;
    //std::cerr << "==> value: " << value << " *left: " << *left << " right: " << *right << " diff: " << diff << " int( left - first ): " << int( left - first ) << std::endl;
    if( diff > 0 ) { return ( value - double( *left ) ) / diff + int( left - first ); }
    if( diff == 0 ) { return int( left - first ); }
    COMMA_THROW( comma::exception, "on pixel coordinate " << value << ": expected elements of undistort map ordered, got element " << int( left - first ) << " equal to " << *left << " element " << int( right - first ) << " equal to " << *right );
}

static double extrapolate_map( const float* first, unsigned int width, double value )
{
    const float* left=first + int(value);
    if( value < 0 ) { left=first; }
    else { if( value >= width - 1 ) { left = first + width - 2; } }
    const float* right=left+1;
    double frac=value - (left - first);
    double step = *right - *left;
    return double(frac * step + *left);
}

Eigen::Vector2d pinhole_to_distorted( const camera::pinhole& pinhole, const Eigen::Vector2d& p )
{
    return pinhole.config().tangentially_corrected( pinhole.config().radially_corrected( p ) ); 
}

// todo: temporarily parametrize on the frame required
Eigen::Vector3d pinhole::to_cartesian( const Eigen::Vector2d& p, bool undistort ) const
{
    Eigen::Vector2d q = ( undistort ? undistorted( p ) : p ) - image_centre_;
    Eigen::Vector2d s = pixel_size_;
    return Eigen::Vector3d( -q.x() * s.x(), -q.y() * s.y(), -config_.focal_length );
}

Eigen::Vector2d pinhole::to_pixel( const Eigen::Vector2d& p ) const
{
    Eigen::Vector2d s = config_.pixel_size();
    return Eigen::Vector2d( -p.x() / s.x(), -p.y() / s.y() ) + image_centre_;
}

Eigen::Vector2d pinhole::config_t::image_centre() const { return principal_point ? *principal_point : Eigen::Vector2d( double( image_size.x() ) / 2, double( image_size.y() ) / 2 ); }

cv::Mat pinhole::config_t::camera_matrix() const
{
    cv::Mat m = cv::Mat::eye( 3, 3, CV_64F );
    const Eigen::Vector2d& c = image_centre();
    if( sensor_size )
    {
        m.at< double >( 0, 0 ) = focal_length * image_size.x() / sensor_size->x();
        m.at< double >( 1, 1 ) = focal_length * image_size.y() / sensor_size->y();
    }
    else
    {
        m.at< double >( 0, 0 ) = focal_length;
        m.at< double >( 1, 1 ) = focal_length;
    }
    m.at< double >( 0, 2 ) = c.x();
    m.at< double >( 1, 2 ) = c.y();
    return m;
}

static void load( const std::string& filename, const Eigen::Vector2i& image_size, cv::Mat& map_x, cv::Mat& map_y )
{
    std::ifstream ifs( &filename[0], std::ios::binary );
    if( !ifs.is_open() ) { COMMA_THROW( comma::exception, "failed to open " << filename ); }
    std::vector< char > buffer( image_size.x() * image_size.y() * 4 );
    ifs.read( &buffer[0], buffer.size() );
    if( ifs.gcount() < int( buffer.size() ) ) { COMMA_THROW( comma::exception, "expected to read " << buffer.size() << " bytes, got " << ifs.gcount() ); }
    map_x = cv::Mat( image_size.y(), image_size.x(), CV_32FC1, &buffer[0] ).clone();
    ifs.read( &buffer[0], buffer.size() );
    if( ifs.gcount() < int( buffer.size() ) ) { COMMA_THROW( comma::exception, "expected to read " << buffer.size() << " bytes, got " << ifs.gcount() ); }
    map_y = cv::Mat( image_size.y(), image_size.x(), CV_32FC1, &buffer[0] ).clone();
}

static void make_distortion_map( const pinhole::config_t& config, cv::Mat& map_x,cv::Mat& map_y )
{
    if( !config.distortion ) { COMMA_THROW( comma::exception, "distortion parameters or map filename not defined" ); }
//     int width=image_size.x();
//     int heigth=image_size.y();
//     i:{0..w},j:{0..h} undistorted(i,j)
    cv::Mat camera = cv::Mat_< double >::zeros( 3, 3 );
    camera.at< double >( 0, 0 ) = config.focal_length * ( config.sensor_size ? double( config.image_size.x() ) / config.sensor_size->x() : 1.0 );
    camera.at< double >( 1, 1 ) = config.focal_length * ( config.sensor_size ? double( config.image_size.y() ) / config.sensor_size->y() : 1.0 );
    Eigen::Vector2d c = config.image_centre();
    camera.at< double >( 0, 2 ) = c.x();
    camera.at< double >( 1, 2 ) = c.y();
    camera.at< double >( 2, 2 ) = 1;
    cv::Vec< double, 5 > distortion_coeff = config.distortion->as< cv::Vec< double, 5 > >();
    cv::Size size( config.image_size.x(), config.image_size.y() );
    cv::initUndistortRectifyMap( camera, distortion_coeff, cv::Mat(), camera, size, CV_32FC1, map_x, map_y );
}

pinhole::distortion_map_t::distortion_map_t( const pinhole::config_t& config )
{
    if( config.distortion->map_filename.empty() ) { make_distortion_map( config, map_x, map_y ); }
    else if( !config.distortion->empty() ) { load( config.distortion->map_filename, config.image_size, map_x, map_y ); }
    std::size_t size = map_x.rows * map_x.cols;
    x_rows_.resize( size );
    y_cols_.resize( size );
    for( int i = 0; i < map_x.rows; ++i ) // todo: phase out x_rows_, y_cols_
    {
        cv::Mat mat_row = map_x.row( i );
        float* ptr = mat_row.ptr< float >();
        std::memcpy( &x_rows_[ map_x.cols * i ], ptr, map_x.cols * sizeof( float ) );
    }
    for( int i = 0; i < map_y.cols; ++i ) // todo: phase out x_rows_, y_cols_
    {
        cv::Mat mat_col = map_y.col( i ).t();
        float* ptr = mat_col.ptr< float >();
        std::memcpy( &y_cols_[ map_y.rows * i ], ptr, map_y.rows * sizeof( float ) );
    }
}

static void save( std::ostream& os, const cv::Mat& mat )
{
    std::streamsize len=(std::streamsize)mat.cols*mat.elemSize();
    for(int i=0;i<mat.rows;i++) { os.write((const char*)mat.ptr(i),len); }
}

void pinhole::distortion_map_t::write( std::ostream& os ) const
{
    save( os, map_x );
    save( os, map_y );
}

Eigen::Vector2d pinhole::undistorted( const Eigen::Vector2d& p ) const
{
    if( !distortion_map() ) { return p; }
    int y_index = p.y() < 0 ? 0 : ( p.y() >= config().image_size.y() ? config().image_size.y() - 1 : p.y() );
    if(distortion_map()->x_rows_.begin() + y_index * config_.image_size.x() == distortion_map()->x_rows_.end()) { COMMA_THROW( comma::exception, "index out of range"); }
    double ox = reverse_map( distortion_map()->x_rows_.begin() + y_index * config_.image_size.x(), static_cast< unsigned int >( config_.image_size.x() ), p.x() );
    int x_index = p.x() < 0 ? 0 : (p.x() >= config_.image_size.x()?config_.image_size.x()-1:p.x());
    if(distortion_map()->y_cols_.begin() + x_index * config_.image_size.y()==distortion_map()->y_cols_.end()) { COMMA_THROW( comma::exception, "index out of range"); }
    double oy = reverse_map( distortion_map()->y_cols_.begin() + x_index * config_.image_size.y(), static_cast< unsigned int >( config_.image_size.y() ), p.y() );
    return Eigen::Vector2d( ox, oy );
}

Eigen::Vector2d pinhole::distort( const Eigen::Vector2d& p ) const
{
    if( !distortion_map() ) { return p; }
    Eigen::Vector2d distorted;
    int y_index = p.y() < 0 ? 0 : p.y() > config_.image_size.y() ? config_.image_size.y() - 1 : p.y();
    distorted[0] = extrapolate_map( &distortion_map()->x_rows_[ y_index * config_.image_size.x() ]
                                  , static_cast< unsigned int >( config_.image_size.x() ), p.x() );
    int x_index = p.x() < 0 ? 0 : p.x() > config_.image_size.x() ? config_.image_size.x() - 1 : p.x();
    distorted[1] = extrapolate_map( &distortion_map()->y_cols_[ x_index * config_.image_size.y() ]
                                  , static_cast< unsigned int >( config_.image_size.y() ), p.y() );
    return distorted;
}

static boost::optional< pinhole::distortion_map_t > make_distortion_map( const pinhole::config_t& config )
{
    if( !config.distortion || config.distortion->empty() ) { return boost::none; }
    return pinhole::distortion_map_t( config );
}

pinhole::pinhole( const pinhole::config_t& config )
    : config_( config )
    , pixel_size_( config.pixel_size() )
    , image_centre_( config.image_centre() )
    , distortion_map_( boost::bind( &make_distortion_map, boost::cref( config_ ) ) )
{
}

const boost::optional< pinhole::distortion_map_t >& pinhole::distortion_map() const { return *distortion_map_; }

std::string pinhole::usage()
{
    std::ostringstream oss;
    oss << "camera config: for a sample, try image-pinhole --output-config" << std::endl;
    oss << "    sensor_size: sensor dimensions; optional; if absent, focal_length should be effectively defined in pixels" << std::endl;
    oss << "        sensor_size/x: sensor width, meters" << std::endl;
    oss << "        sensor_size/y: sensor height, m, meters" << std::endl;
    oss << "    image_size: image dimensions, pixels" << std::endl;
    oss << "        image_size/x: image width, pixels" << std::endl;
    oss << "        image_size/y: image height, pixels" << std::endl;
    oss << "    focal_length: focal length, meters" << std::endl;
    oss << "    principal_point: image centre, pixels, default: geometrical image centre" << std::endl;
    oss << "        principal_point/x" << std::endl;
    oss << "        principal_point/y" << std::endl;
    oss << "    distortion: optional" << std::endl;
    oss << "        distortion/radial/k1" << std::endl;
    oss << "        distortion/radial/k2" << std::endl;
    oss << "        distortion/radial/k3" << std::endl;
    oss << "        distortion/tangential/p1" << std::endl;
    oss << "        distortion/tangential/p2" << std::endl;
    oss << "        distortion/map_filename=<filename>: distortion map as two concatenated binary distortion maps for x and y" << std::endl;
    oss << "                                   each map is matrix of floats of size image_size/x X image_size/y" << std::endl;
    oss << "    calculated values" << std::endl;
    oss << "        pixel size: calculated as sensor_size/image_size; this is used to convert between cartesian frame (meter) and image frame (pixels)" << std::endl;
    oss << "            if sensor_size is empty (not specified) then pixel_size is assumed to be 1 meter; which means focal_length is effectively in pixels and no scaling when converting between cartesian/image frame" << std::endl;
    return oss.str();
}

} } // namespace snark { namespace camera {
