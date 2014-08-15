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

/// @author vsevolod vlaskine

#ifdef WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif

#include <comma/application/signal_flag.h>
#include <comma/base/exception.h>
#include <comma/csv/stream.h>
#include <comma/csv/impl/program_options.h>
#include <comma/math/compare.h>
#include <comma/visiting/traits.h>
#include <snark/visiting/eigen.h>

// #include <boost/random/mersenne_twister.hpp>
// #include <boost/random/uniform_real.hpp>
// #include <boost/random/variate_generator.hpp>
// #include <Eigen/Geometry>
// boost::mt19937 generator;
// boost::uniform_real< double > distribution( 0, 1 );
// boost::variate_generator< boost::mt19937&, boost::uniform_real< double > > r( generator, distribution );

int main( int argc, char** argv )
{
//     for( unsigned int i = 0; i < boost::lexical_cast< unsigned int >( argv[1] ); ++i )
//     {
//         double x = r() * 20 - 10;
//         double y = r() * 20 - 10;
//         double z = r() * 20 - 10;
//         std::cout << x << ',' << y << ',' << z << ',' << int( std::max( std::abs( x ), std::max( std::abs( y ), std::abs( z ) ) ) ) << std::endl;
//     }
//     return 0;

    try
    {
        std::string normal_string;
        std::string points_string;
        std::string point_outside;
        boost::program_options::options_description description( "options" );
        description.add_options()
            ( "help,h", "display help message" )
            ( "points,p", boost::program_options::value< std::string >( &points_string )->default_value( "0,0,0" ), "point(s) belonging to the plane, either 3 points, or 1 point, if --normal defined" )
            ( "point-outside", boost::program_options::value< std::string >( &point_outside ), "point on the side of the plane where the normal would point, a convenience option; 3 points are enough" )
            ( "normal,n", boost::program_options::value< std::string >( &normal_string ), "normal to the plane" )
            ( "intersections", "assume the input represents a trajectory, find all its intersections with the plane");
        description.add( comma::csv::program_options::description( "x,y,z" ) );
        boost::program_options::variables_map vm;
        boost::program_options::store( boost::program_options::parse_command_line( argc, argv, description), vm );
        boost::program_options::notify( vm );
        if ( vm.count( "help" ) )
        {
            std::cerr << "take points on stdin, append distance from a given plane," << std::endl;
            std::cerr << "if --intersection is specified, assume the input represents a trajectory, find its intersections with the plane," << std::endl;
            std::cerr << "and output adjacent points of the trajectory, where the intersection occurs, direction of intersection, and intersection point" << std::endl;
            std::cerr << std::endl;
            std::cerr << "usage: cat points.csv | points-slice [options] > points_with_distance.csv" << std::endl;
            std::cerr << std::endl;
            std::cerr << "input: --points x1,y1,z1,x2,y2,z2,x3,y3,x3    (\"--normal\" is ignored, normal direction is based on \"--points\")" << std::endl;
            std::cerr << "input: --points x1,y1,z1,x2,y2,z2,x3,y3,x3 --point-outside x4,y4,z4    (\"--normal\" is ignored, normal direction is based on \"--point-outside\")" << std::endl;
            std::cerr << "input: --points x,y,z --normal n1,n2,n3" << std::endl;
            std::cerr << "input: --points x1,y1,z1,x2,y2,z2,x3,y3,x3 --normal n1,n2,n3    (\"x2,y2,z2,x3,y3,x3\" are ignored)" << std::endl;
            std::cerr << "input: --normal n1,n2,n3    (using default \"--points 0,0,0\")" << std::endl;
            std::cerr << "output: x,y,z,distance" << std::endl;
            std::cerr << "output if --intersections is specified: x1,y1,z1,x2,y2,z2,p1,p2,p3,i" << std::endl;
            std::cerr << "    where \"x1,y1,z1,x2,y2,z2\" are the points of an intersecting interval, \"p1,p2,p3\" is the intersection point, \"i\" is the direction of intersection" << std::endl;
            std::cerr << "binary output format: <input_format>,d" << std::endl;
            std::cerr << std::endl;
            std::cerr << description << std::endl;
            std::cerr << "examples:" << std::endl;
            std::cerr << "   echo -e \"0,0,-1\\n0,0,0\\n0,0,1\" | points-slice --points 0,0,0,1,0,0,0,1,0" << std::endl;
            std::cerr << "   echo -e \"0,0,-1\\n0,0,0\\n0,0,1\" | points-slice --points 0,0,0,1,0,0,0,1,0 --point-outside 0,0,1" << std::endl;
            std::cerr << "   echo -e \"0,0,-1\\n0,0,0\\n0,0,1\" | points-slice --points 0,0,0 --normal 0,0,1" << std::endl;
            std::cerr << "   echo -e \"0,0,-1\\n0,0,0\\n0,0,1\" | points-slice --normal 0,0,1" << std::endl;
            return 1;
        }
        if( vm.count( "points" ) == 0 ) { std::cerr << "points-slice: please specify --points" << std::endl; return 1; }
        comma::csv::options csv = comma::csv::program_options::get( vm );
        Eigen::Vector3d normal;
        Eigen::Vector3d point;
        if( vm.count( "normal" ) )
        {
            normal = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( normal_string );
            point = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( points_string );
        }
        else
        {
            boost::array< Eigen::Vector3d, 3 > points; // quick and dirty
            std::vector< std::string > v = comma::split( points_string, ',' );
            if( v.size() != 9 ) { std::cerr << "points-slice: expected 3 points, got: \"" << points_string << "\"" << std::endl; return 1; }
            point = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( v[0] + ',' + v[1] + ',' + v[2] ); // quick and dirty
            Eigen::Vector3d a = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( v[3] + ',' + v[4] + ',' + v[5] ); // quick and dirty
            Eigen::Vector3d b = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( v[6] + ',' + v[7] + ',' + v[8] ); // quick and dirty
            a -= point;
            b -= point;
            if( comma::math::equal( std::abs( a.dot( b ) ), a.norm() * b.norm() ) ) { std::cerr << "points-slice: given points are not corners or a triangle: \"" << points_string << "\"" << std::endl; }
            normal = a.cross( b );
            if( vm.count( "point-outside" ) )
            {
                Eigen::Vector3d p = comma::csv::ascii< Eigen::Vector3d >( "x,y,z", ',' ).get( point_outside );
                if( comma::math::equal( ( p - point ).dot( normal ), 0 ) ) { std::cerr << "points-slice: expected a point outside of the plane, got: " << point_outside << ", which belongs to the plane" << std::endl; return 1; }
                normal *= normal.dot( p - point ) > 0 ? 1 : -1;
            }
        }
        normal.normalize();
        #ifdef WIN32
            _setmode( _fileno( stdout ), _O_BINARY ); /// @todo move to a library
        #endif
        comma::csv::input_stream< Eigen::Vector3d > istream( std::cin, csv );
        comma::signal_flag is_shutdown;
        if( vm.count("intersections") )
        {
            Eigen::Hyperplane< double, 3 > plane( normal, point );
            boost::optional< Eigen::Vector3d > last;
            double d_last = 0;
            while( !is_shutdown && ( istream.ready() || ( !std::cin.eof() && std::cin.good() ) ) )
            {
                const Eigen::Vector3d* p = istream.read();
                if( !p ) { break; }
                double d = ( *p - point ).dot( normal );
                if( last )
                {
                    bool intersects = ( d * d_last <= 0 );
                    if( intersects )
                    {
                        Eigen::Vector3d intersection_point;
                        bool lies_on_plane = ( d == 0 && d_last == 0 );
                        if( lies_on_plane ) 
                        { 
                            intersection_point = *last; 
                        }
                        else
                        {
                            Eigen::ParametrizedLine< double, 3 > line = Eigen::ParametrizedLine< double, 3 >::Through( *last, *p );
                            intersection_point = line.intersectionPoint( plane );
                        }
                        int direction;
                        if( d_last != 0 ) { direction = ( d_last > 0 ) ? 1 : -1; }
                        else if( d != 0 ) { direction = ( d < 0 ) ? 1 : -1; }
                        else { direction = 0; }
                        if( csv.binary() )
                        {
                            std::cout.write( reinterpret_cast< const char* >( &( (*last).x() ) ), sizeof( double ) );
                            std::cout.write( reinterpret_cast< const char* >( &( (*last).y() ) ), sizeof( double ) );
                            std::cout.write( reinterpret_cast< const char* >( &( (*last).z() ) ), sizeof( double ) );
                            std::cout.write( istream.binary().last(), istream.binary().binary().format().size() );
                            std::cout.write( reinterpret_cast< const char* >( &( intersection_point.x() ) ), sizeof( double ) );
                            std::cout.write( reinterpret_cast< const char* >( &( intersection_point.y() ) ), sizeof( double ) );
                            std::cout.write( reinterpret_cast< const char* >( &( intersection_point.z() ) ), sizeof( double ) );
                            std::cout.write( reinterpret_cast< const char* >( &direction ), sizeof( int ) );
                        }
                        else
                        {
                            std::cout << ( *last ).x() << csv.delimiter << ( *last ).y() << csv.delimiter << ( *last ).z() << csv.delimiter
                            << comma::join( istream.ascii().last(), csv.delimiter ) << csv.delimiter 
                            << intersection_point.x() << csv.delimiter << intersection_point.y() << csv.delimiter << intersection_point.z() << csv.delimiter 
                            << direction << std::endl;
                        }
                    }
                }
                last.reset( *p );
                d_last = d;
            }
        }
        else
        {
            while( !is_shutdown && ( istream.ready() || ( !std::cin.eof() && std::cin.good() ) ) )
            {
                const Eigen::Vector3d* p = istream.read();
                if( !p ) { break; }
                double d = ( *p - point ).dot( normal );
                if( csv.binary() )
                {
                    std::cout.write( istream.binary().last(), istream.binary().binary().format().size() );
                    std::cout.write( reinterpret_cast< const char* >( &d ), sizeof( double ) );
                }
                else
                {
                    std::cout << comma::join( istream.ascii().last(), csv.delimiter ) << csv.delimiter << d << std::endl;
                }
            }
        }
        return 0;
    }
    catch( std::exception& ex )
    {
        std::cerr << "points-slice: " << ex.what() << std::endl;
    }
    catch( ... )
    {
        std::cerr << "points-slice: unknown exception" << std::endl;
    }
    return 1;
}
