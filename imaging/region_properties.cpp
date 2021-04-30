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
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/foreach.hpp>
#include <Eigen/Core>
#include "region_properties.h"
#include <comma/base/exception.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS( cs::cartesian )

namespace snark{ namespace imaging {

/// compute the area of a polygon and its convex hull
/// @param points polygon points
/// @param area area of the polygon
/// @param convexArea area of the convex hull
void compute_area( const std::vector< cv::Point >& points, double& area, double& convexArea )
{
    boost::geometry::model::polygon< boost::tuple< double, double > > polygon;
    for( unsigned int i = 0; i < points.size(); i++ )
    {
        boost::geometry::append( polygon, boost::make_tuple( points[i].x, points[i].y ) );
    }
    boost::geometry::append( polygon, boost::make_tuple( points[0].x, points[0].y ) ); // close polygon

    area = boost::geometry::area( polygon );

    boost::geometry::model::polygon< boost::tuple<double, double> > hull;
    boost::geometry::convex_hull( polygon, hull );

    convexArea = boost::geometry::area( hull );
}

/// constructor
/// @param image input image, is considered as a binary image ( all non-zero pixels are 1 )
region_properties::region_properties ( const cv::Mat& image, double minArea ):
    m_minArea( minArea )
{
    cv::Mat binary;
    if( image.channels() == 3 )
    {
        cv::cvtColor( image, binary, cv::COLOR_RGB2GRAY );
    }
    else if( image.channels() == 1 )
    {
        binary = image;
    }
    else
    {
        COMMA_THROW( comma::exception, "incorrect number of channels, should be 1 or 3, not " << image.channels() );
    }
//     cv::Mat closed;
//     cv::morphologyEx( binary, closed, cv::MORPH_CLOSE, cv::Mat::ones( 3, 3, CV_8U) );
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours( binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
    
    for( unsigned int i = 0; i < contours.size(); i++ )
    {
        binary = cv::Scalar(0);
        cv::drawContours( binary, contours, i, cv::Scalar(0xFF), cv::FILLED );
        cv::Rect rect = cv::boundingRect( cv::Mat( contours[i]) );
        cv::Moments moments = cv::moments( binary( rect ), true );
        double x = moments.m10/moments.m00;
        double y = moments.m01/moments.m00;
        double area = moments.m00; // cv::countNonZero( binary( rect ) )
        if( area > m_minArea )
        {
            // see wikipedia, image moments
            double diff = moments.nu20 - moments.nu02;
            double a = 0.5 * ( moments.nu20 + moments.nu02 );
            double b = 0.5 * std::sqrt( 4 * moments.nu11 * moments.nu11 + diff * diff );
            double minEigenValue = a - b;
            double maxEigenValue = a + b;
    //         std::cerr << " min " << minEigenValue << " max " << maxEigenValue << std::endl;
            double theta = 0.5 * std::atan2( 2 * moments.nu11, diff );
            double eccentricity = 1;
            if( std::fabs( maxEigenValue ) > 1e-15 )
            {
                eccentricity = std::sqrt( 1 - minEigenValue / maxEigenValue );
            }

            double polygonArea;
            double convexArea;
            compute_area( contours[i], polygonArea, convexArea );
    //         std::cerr << " area " << area << " polygon " << polygonArea << " convex " << convexArea << std::endl;

            blob blob;
            blob.majorAxis = 2 * std::sqrt( moments.m00 * maxEigenValue );
            blob.minorAxis = 2 * std::sqrt( moments.m00 * minEigenValue );
            blob.orientation = theta;
            blob.centroid = cv::Point( x + rect.x, y + rect.y );
            blob.area = area;
            blob.eccentricity = eccentricity;
            blob.solidity = 0;
            if( std::fabs( convexArea ) > 1e-15 )
            {
                blob.solidity = polygonArea / convexArea;
            }
            m_blobs.push_back( blob );
        }
    }    
}

/// draw debug information on the image
void region_properties::show( cv::Mat& image, bool text )
{
    for( unsigned int i = 0; i < m_blobs.size(); i++ )
    {
        cv::Point centroid = m_blobs[i].centroid;
        cv::circle( image, centroid, 3, cv::Scalar( 0, 0, 255 ), 2 );
        std::stringstream s;
        s << i;
        if( text )
        {
            cv::putText( image, s.str(), centroid + cv::Point( 2, 2 ), cv::FONT_HERSHEY_PLAIN ,1, cv::Scalar( 0, 0, 255 ) );
        }
        cv::ellipse( image, centroid, cv::Size( m_blobs[i].majorAxis, m_blobs[i].minorAxis ), m_blobs[i].orientation * 180.0 / M_PI, 0, 360, cv::Scalar( 0, 255, 0 ) );
//         std::cerr << i << ": area " << m_blobs[i].area << " eccentricity " << m_blobs[i].eccentricity << " solidity " << m_blobs[i].solidity << std::endl;
    }
}

} } 


