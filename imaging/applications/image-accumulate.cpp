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


#ifdef WIN32
#include <winsock2.h>
#endif
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <comma/application/command_line_options.h>
#include <comma/application/signal_flag.h>
#include <comma/base/exception.h>
#include <comma/base/types.h>
#include <comma/csv/ascii.h>
#include <comma/csv/stream.h>
#include <comma/math/compare.h>
#include <comma/math/cyclic.h>
#include <comma/name_value/parser.h>
#include <comma/string/string.h>
#include <comma/visiting/traits.h>
#include "../cv_mat/serialization.h"

void usage()
{
    std::cerr << std::endl;
    std::cerr << "take arrays of values on stdin, output them as images on stdout" << std::endl;
    std::cerr << std::endl;
    std::cerr << "usage: cat arrays.bin | image-accumulate <options> | cv-cat view > /dev/null" << std::endl;
    std::cerr << std::endl;
    std::cerr << "options" << std::endl;
    std::cerr << "    --binary,-b=<format>: binary input format" << std::endl;
    std::cerr << "    --clockwise: if polar, clockwise rotation; default: counter-clockwise" << std::endl;
    std::cerr << "    --colour-map,--colourmap,--colour=<which>: currently 'hot', 'jet', 'green', 'red', or <r>,<g>,<b>; default green (0,255,0); use --output=\"type=3ub\" to activate colour map" << std::endl;
    std::cerr << "    --delimiter,-d: csv delimiter; default ','" << std::endl;
    std::cerr << "    --degrees: if angle field present, angle as degrees; default radians" << std::endl;
    std::cerr << "    --dirty,--do-not-clean: do not clean the current image from the rows from previous blocks" << std::endl;
    std::cerr << "                            (useful, e.g. when sparce data accumulate over time)" << std::endl;
    std::cerr << "    --fields,-f=<fields>: " << std::endl;
    std::cerr << "        t: timestamp" << std::endl;
    std::cerr << "        block: block id (e.g. revolution)" << std::endl;
    std::cerr << "             if absent, block size from --size will be used" << std::endl;
    std::cerr << "             to identify blocks" << std::endl;
    std::cerr << "        id: channel id" << std::endl;
    std::cerr << "        row: row number in block" << std::endl;
    std::cerr << "             if absent, all rows in block present" << std::endl;
    std::cerr << "        angle: in radians; used for polar; if present, row number is ignored" << std::endl;
    std::cerr << "             note: currently, for simplicity's sake, block size" << std::endl;
    std::cerr << "                   specifies the angle resolution" << std::endl;
    std::cerr << "                   e.g. for block size 1000, angle precision will be 2 * pi / 1000 in the image" << std::endl;
    std::cerr << "                   see --size option" << std::endl;
    std::cerr << "        values: vector values" << std::endl;
    std::cerr << "        default: values" << std::endl;
    std::cerr << "    --fps=<value>: if present, update image <value> frames per second" << std::endl;
    std::cerr << "                   if absent, update on every new revolution (then revolution and sweep fields have to be present)" << std::endl;
    std::cerr << "    --id=<id>[,<options>]: if id field present, channel id to output, e.g: --id=1 --id=\"3;scale=1,5\" --id=4" << std::endl;
    std::cerr << "        options: scale=<min>,<max>: see --scale" << std::endl;
    std::cerr << "                 colourmap=<value>: see --colourmap" << std::endl;
    std::cerr << "                 dial-colour=<value>: see --dial-colour" << std::endl;
    std::cerr << "                 default: values defined in command line options" << std::endl;
    std::cerr << "    --row-number-offset,--offset=<value>: row number offset; default: 0" << std::endl;
    std::cerr << "    --offset-from-center=<pixels>: polar only: offset all values by <pixels> from center" << std::endl;
    std::cerr << "    --output=<options>: see cv-cat --help for output options; default: no-header;rows=<r>;cols=<v>;type=3ub" << std::endl;
    std::cerr << "        note: type=3ub instructs image-accumulate to apply a colour map to the 8-bit gray-scale pixel values (use type=ub to output gray-scale images)" << std::endl;
    std::cerr << "    --polar: if present, convert to polar coordinates; default: output waterfall" << std::endl;
    std::cerr << "    --scale=<from>,<to>: min and max for black-to-white gradation; default: 0-255" << std::endl;
    std::cerr << "    --dial,--dial-size=<pixels>: highlight the current position (if --fps present); default: 1" << std::endl;
    std::cerr << "    --dial-colour=<colour> | <r>,<g>,<b>: dial colour; default: white" << std::endl;
    std::cerr << "    --size=<rows>,<cols>: rows is the number of input rows to accumulate and cols is the width of each row" << std::endl;
    std::cerr << "    --output-options: print output image options and exit" << std::endl;
    std::cerr << "    --verbose,-v: more output" << std::endl;
    std::cerr << "    --z=<value>: 'up' or 'down': in polar direction of z axis; default: up" << std::endl;
    std::cerr << std::endl;
    std::cerr << "examples" << std::endl;
    std::cerr << "    accumulate and visualise the last 100 frames from the output of a slit-scan camera with 640 pixels of 8-bit gray-scale values:" << std::endl;
    std::cerr << "        cat slit-scan.bin | image-accumulate --fields=,,,,values --binary=t,3ui,640ub --size=100,640 --output=\"type=ub\" | cv-cat \"view;null\"" << std::endl;
    std::cerr << "    the same as above but also applies jet colour map to gray-scale pixel values:" << std::endl;
    std::cerr << "        cat slit-scan.bin | image-accumulate --fields=,,,,values --binary=t,3ui,640ub --size=100,640 --output=\"type=3ub\" --colourmap=jet | cv-cat \"view;null\"" << std::endl;
    std::cerr << "    todo: more examples" << std::endl;
    std::cerr << std::endl;
    exit( 1 );
}

struct input
{
    boost::posix_time::ptime t;
    unsigned int block;
    unsigned int row;
    double angle;
    unsigned int id;
    std::vector< double > values;
    input() : block( 0 ), row( 0 ), angle( 0 ), id( 0 ) 
    {
        t=boost::posix_time::microsec_clock::universal_time();
    }
};

namespace comma { namespace visiting {
    
template <> struct traits< input >
{
    template< typename K, typename V >
    static void visit( const K& k, const input& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "block", t.block );
        v.apply( "row", t.row );
        v.apply( "angle", t.angle );
        v.apply( "id", t.id );
        v.apply( "values", t.values ); // todo: visiting may be way too slow! use memcpy?
    }    
    
    template< typename K, typename V >
    static void visit( const K& k, input& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "block", t.block );
        v.apply( "row", t.row );
        v.apply( "angle", t.angle );
        v.apply( "id", t.id );
        v.apply( "values", t.values ); // todo: visiting may be way too slow! use memcpy?
    }
};
    
} } // namespace comma { namespace visiting {

template < typename T >
class scaled // quick and dirty
{
    public:
        scaled( std::pair< double, double > from
              , std::pair< double, double > to = std::pair< double, double >( std::numeric_limits< T >::min(), std::numeric_limits< T >::max() ) )
            : from_( from )
            , to_( to )
        {
            if( !comma::math::less( from.first, from.second ) ) { COMMA_THROW( comma::exception, "expected scale, got \"" << from.first << "," << from.second << "\"" ); }
            factor_ = ( to.second - to.first ) / ( from.second - from.first );
        }
        
        T operator()( double v ) const
        {
            if( v <= from_.first ) { return to_.first; }
            if( v >= from_.second ) { return to_.second; }
            return to_.first + ( v - from_.first ) * factor_;
        } 
        
    private:
        std::pair< double, double > from_;
        std::pair< double, double > to_;
        double factor_;
};

struct color_map // quick and dirty
{
    typedef boost::array< unsigned char, 3 > pixel;
    
    typedef boost::array< pixel, 256 > values;
    
    enum { red = 2, green = 1, blue = 0 };
    
    static values constant( unsigned char r, unsigned char g, unsigned char b )
    {
        values v;
        //for( unsigned int i = 0; i < 255; ++i ) { v[i][red] = ( i * r ) / 256; v[i][green] = ( i * g ) / 256; v[i][blue] = ( i * b ) / 256; }
        for( unsigned int i = 0; i < 256; ++i )
        { 
            v[i][red] = ( i * r ) / 255; v[i][green] = ( i * g ) / 255; v[i][blue] = ( i * b ) / 255;
        }
        return v;
    }
    
    static values temperature( unsigned char offset_r, unsigned char offset_g )
    {
        values v;
        for( unsigned int i = 0; i < offset_r; ++i )
        {
            v[i][red] = ( i * 255 ) / offset_r;
            v[i][green] = 0;
            v[i][blue] = 0;
        }
        for( unsigned int i = offset_r; i < ( offset_r + offset_g ); ++i )
        {
            v[i][red] = 255;
            v[i][green] = ( ( i - offset_r ) * 255 ) / offset_g;
            v[i][blue] = 0;
        }
        for( unsigned int i = offset_r + offset_g; i < 256; ++i )
        {
            v[i][red] = 255;
            v[i][green] = 255;
            v[i][blue] = ( ( i - offset_r - offset_g ) * 255 ) / ( 256 - offset_r - offset_g );
        }
        return v;
    }
    
    static values jet()
    {
        values v;
        jet_impl_( v, red, 32 + 64 );
        jet_impl_( v, green, 32 );
        jet_impl_( v, blue, -32 );
        for( unsigned int i = 0; i < 64; ++i ) { v[i][red] = 0; }
        for( unsigned int i = 256 - 64; i < 256; ++i ) { v[i][blue] = 0; }
        return v;
    }
    
    static pixel contrast_to( const values& v )
    {
        pixel p;
        for( unsigned int i = 0; i < 3; ++i )
        {
            unsigned int average = 0;
            for( unsigned int k = 0; k < 256; ++k ) { average += v[k][i]; }
            average /= 256;
            p[i] = 255 - average;
        }
        return p;
    }
    
    private:
        static void jet_impl_( values& v, unsigned int channel, int offset )
        {
            for( unsigned int i = 0; i < 256; ++i ) { v[i][channel] = 0; }
            comma::math::cyclic< unsigned int > c( 0, 256 );
            c += offset;
            for( unsigned int i = 1; i < 64; ++i, ++c ) { v[ c() ][channel] = i * 4; }
            for( unsigned int i = 0; i < 65; ++i, ++c ) { v[ c() ][channel] = 255; }
            for( unsigned int i = 1; i < 64; ++i, ++c ) { v[ c() ][channel] = 255 - i * 4; }
        }
};

static bool verbose;
static comma::csv::options csv;
static bool polar = false;
static unsigned int block_size;
static unsigned int row_size;
static unsigned int row_number_offset;
static bool as_radians;
static bool clockwise;
static int sign;
static unsigned int dial_size = 0;
static bool z_up;
static bool has_block;
static bool has_row;
static bool has_angle;

static input in;
static std::pair< double, double > scale;
static boost::optional< double > fps;
static std::vector< std::pair< double, double > > sin_cos;
static cv::Mat image;
typedef std::map< unsigned int, unsigned int > Ids;
static Ids ids;
static boost::scoped_ptr< snark::cv_mat::serialization > serialization;
static boost::scoped_ptr< boost::thread > output_thread;
static bool done = false;
static comma::signal_flag is_shutdown;
static snark::cv_mat::serialization::options output_options;
static unsigned int pixel_size;
static unsigned int offset_from_center; // quick and dirty

static std::vector< std::pair< double, double > > precomputed_sin_cos_()
{
    if( !polar ) { return std::vector< std::pair< double, double > >(); }
    std::vector< std::pair< double, double > > v( block_size ); // todo: quick and dirty; apparently easy to optimize
    double step = ( M_PI * 2 ) / block_size;
    double angle = 0;
    for( std::size_t i = 0; i < v.size(); ++i, angle += step )
    {
        v[i].first = std::sin( -angle );
        v[i].second = std::cos( -angle );
    }
    return v;
}

static void output_once_( const boost::posix_time::ptime& t )
{
    std::pair< boost::posix_time::ptime, cv::Mat > output;
    if( polar )
    {
        static unsigned int polar_size = ( row_size + offset_from_center ) * 2 + 1;
        static unsigned int type = output_options.get_header().type;
        static cv::Mat polar_image( polar_size, polar_size * ids.size(), type );
        ::memset( const_cast< unsigned char* >( polar_image.datastart ), 0, polar_image.dataend - polar_image.datastart );
        for( unsigned int i = 0; i < ids.size(); ++i )
        {
            unsigned int offset = row_size * i;
            unsigned int polar_offset = polar_size * i;
            for( unsigned int row = 0; row < block_size; ++row )
            {
                double x_step = sin_cos[row].second; 
                double y_step = sin_cos[row].first * sign;
                double x = row_size + offset_from_center + polar_offset; 
                double y = row_size + offset_from_center;
                x += sin_cos[row].second * offset_from_center;
                y += sin_cos[row].first * sign * offset_from_center;
                for( unsigned int column = 0; column < row_size; ++column, x += x_step, y += y_step )
                {
                    ::memcpy( const_cast< unsigned char* >( polar_image.datastart ) + ( int( y ) * ( polar_image.cols ) + int( x ) ) * pixel_size
                            , image.datastart + ( int( row ) * image.cols + offset + column ) * pixel_size
                            , pixel_size );
                }
            }
        }
        output.second = polar_image;
    }
    else
    {
        output.second = image;
    }
    output.first = t;
    serialization->write( std::cout, output );
}

static void output_()
{
    static const boost::posix_time::time_duration period = boost::posix_time::milliseconds( (long)(1000.0 / *fps) );
    static const boost::posix_time::time_duration timeout = boost::posix_time::milliseconds( 10 );
    boost::posix_time::ptime time_to_output = boost::posix_time::microsec_clock::universal_time() + period;    
    while( !is_shutdown && !done && std::cout.good() )
    {
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
        if( time_to_output <= now )
        {
            time_to_output = now + period;
            output_once_( now );
        }
        boost::thread::sleep( now + timeout );
    }
}

class channel
{
    public:
        struct options
        {
            std::pair< double, double > scale;
            std::string colourmap;
            std::string dial_colour;
            bool do_not_clean;
            //unsigned int offset_from_center; // quick and dirty
            
            options( const comma::command_line_options& options )
                : scale( comma::csv::ascii< std::pair< double, double > >().get( options.value< std::string >( "--scale", "0,255" ) ) )
                , colourmap( options.value< std::string >( "--colour-map,--colourmap,--colour", "0,255,0" ) )
                , dial_colour( options.value< std::string >( "--dial-colour", "white" ) )
                , do_not_clean( options.exists( "--dirty,--do-not-clean" ) )
                //, offset_from_center( options.value( "--offset-from-center", 0 ) )
            {
            }
        };
        
        channel( unsigned int index, const channel::options& options )
            : index_( index )
            , scaled_( options.scale )
            , row_( 0, block_size )
            , row_count_( 0 )
            , angle_step_( ( M_PI * 2 ) / block_size )
            , options_( options )
        {
            if( options.colourmap == "green" ) { colourmap_ = color_map::constant( 0, 255, 0 ); }
            else if( options.colourmap == "red" ) { colourmap_ = color_map::constant( 255, 0, 0 ); }
            else if( options.colourmap == "hot" ) { colourmap_ = color_map::temperature( 96, 96 ); }
            else if( options.colourmap == "jet" ) { colourmap_ = color_map::jet(); }
            else
            { 
                std::vector< std::string > v = comma::split( options.colourmap, ',' );
                if( v.size() != 3 ) { COMMA_THROW( comma::exception, "image-accumulate: expected colourmap, got '" << options.colourmap << "'" ); }
                colourmap_ = color_map::constant( boost::lexical_cast< unsigned int >( v[0] ), boost::lexical_cast< unsigned int >( v[1] ), boost::lexical_cast< unsigned int >( v[2] ) );
            }
            if( options.dial_colour == "white" ) { dial_colour_ = cv::Scalar( 255, 255, 255 ); }
            else if( options.dial_colour == "white" ) { dial_colour_ = cv::Scalar( 255, 255, 255 ); }
            else if( options.dial_colour == "black" ) { dial_colour_ = cv::Scalar( 0, 0, 0 ); }
            else if( options.dial_colour == "red" ) { dial_colour_ = cv::Scalar( 0, 0, 255 ); }
            else if( options.dial_colour == "green" ) { dial_colour_ = cv::Scalar( 0, 255, 0 ); }
            else if( options.dial_colour == "blue" ) { dial_colour_ = cv::Scalar( 255, 0, 0 ); }
            else if( options.dial_colour == "yellow" ) { dial_colour_ = cv::Scalar( 0, 255, 255 ); }
            else
            { 
                std::vector< std::string > v = comma::split( options.dial_colour, ',' );
                if( v.size() != 3 ) { COMMA_THROW( comma::exception, "image-accumulate: expected colour, got '" << options.dial_colour << "'" ); }
                dial_colour_ = cv::Scalar( boost::lexical_cast< unsigned int >( v[2] ), boost::lexical_cast< unsigned int >( v[1] ), boost::lexical_cast< unsigned int >( v[0] ) );
            }
        }

        bool draw( const input* p )
        {
            if( !fps )
            {   
                if(    ( !p && row_count_ > 0 )
                    || ( !has_block && row_count_ >= block_size )
                    || ( p && block_ && *block_ != p->block ) )
                {
                    output_once_( timestamp_ );
                    image.setTo( 0 );
                    row_count_ = 0;
                }
            }
            if( !p ) { return false; }
            timestamp_ = p->t;
            block_ = p->block;
            unsigned int new_row;
            if( has_angle )
            {
                double angle = as_radians ? p->angle : ( p->angle * ( M_PI / 180.0 ) );
                if( sign < 0 ) { angle = M_PI * 2 - angle; }
                new_row = ( unsigned int )( angle / angle_step_ );
            }
            else
            {
                new_row = p->row;
            }
            if( new_row >= block_size ) { std::cerr << "image-accumulate: expected row number less than " << block_size << "; got " << new_row; return false; }
            comma::math::cyclic< unsigned int > r( 0, block_size );
            r = new_row + row_number_offset;
            new_row = r();
            // todo: test row number offset
            if( has_row || has_angle )
            {
                if( fps ) { clean_( new_row, clockwise ? -1 : 1 ); }
                row_ = new_row;
            }
            else
            {
                row_ = row_count_;
            }
            ++row_count_;
            draw_line_( p, row_() );
            draw_dial_();
            return true;
        }
        
    private:
        unsigned int index_;
        scaled< unsigned char > scaled_;
        color_map::values colourmap_;
        cv::Scalar dial_colour_;
        boost::posix_time::ptime timestamp_;
        boost::optional< unsigned int > block_;
        comma::math::cyclic< unsigned int > row_;
        unsigned int row_count_;
        double angle_step_;
        options options_;
        
        void draw_dial_()
        {
            comma::math::cyclic< unsigned int > begin( row_ + 1 );
            comma::math::cyclic< unsigned int > end( begin + dial_size );
            for( comma::math::cyclic< unsigned int > row = begin; row != end; ++row )
            {
                //cv::line( image, cv::Point( index_ * row_size, row() ), cv::Point( ( index_ + 1 ) * row_size, row() ), dial_colour_ );
            }
        }
        
        void clean_( unsigned int to, unsigned int increment )
        {
            if( options_.do_not_clean ) { return; }
            comma::math::cyclic< unsigned int > r( row_ );
            unsigned int offset = index_ * row_size * pixel_size;
            for( r += increment; row_() != to && r() != to; r += increment )
            {
                ::memset( const_cast< unsigned char* >( image.datastart ) + offset + image.cols * r() * pixel_size, 0, row_size * pixel_size );
            }
        }
        
        void draw_line_( const input* p, unsigned int row )
        {
            unsigned int offset = ( image.cols * row + index_ * row_size ) * pixel_size;
            if( pixel_size == 1 ) // quick and dirty
            {
                for( unsigned int i = 0; i < row_size; ++i )
                {
                    *( const_cast< unsigned char* >( image.datastart ) + offset + i ) = scaled_( p->values[i] );
                }
            }
            else
            {
                for( unsigned int i = 0; i < row_size; ++i )
                {
                    const boost::array< unsigned char, 3 >& colour = colourmap_[ scaled_( p->values[i] ) ];
                    ::memcpy( const_cast< unsigned char* >( image.datastart ) + offset + i * 3, &colour[0], 3 );   
                }
            }
        }
};

namespace comma { namespace visiting {
    
template <> struct traits< channel::options >
{
    template< typename K, typename V >
    static void visit( const K& k, channel::options& t, V& v )
    {
        std::string scale = boost::lexical_cast< std::string >( t.scale.first ) + "," + boost::lexical_cast< std::string >( t.scale.second );
        v.apply( "scale", scale ); // quick and dirty
        t.scale = comma::csv::ascii< std::pair< double, double > >().get( scale );
        v.apply( "colourmap", t.colourmap );
        v.apply( "dial-colour", t.dial_colour );
        //v.apply( "offset-from-center", t.offset_from_center );
    }
};
    
} } // namespace comma { namespace visiting {

int main( int ac, char** av )
{
    int result = 0;
    try
    {
        comma::command_line_options options( ac, av );
        if( options.exists( "--help,-h" ) ) { usage(); }
        verbose = options.exists( "--verbose,-v" );
        csv = comma::csv::options( options, "values" );
        csv.full_xpath = true;
        row_number_offset = options.value( "--row-number-offset,--offset", 0 );
        polar = options.exists( "--polar" );
        as_radians = !options.exists( "--degrees" );
        clockwise = options.exists( "--clockwise" );
        offset_from_center = options.value( "--offset-from-center", 0 ); // quick and dirty
        z_up = options.value< std::string >( "--z", "up" ) == "up";
        sign = ( z_up && !clockwise ) || ( !z_up && clockwise ) ? 1 : -1;
        if( options.exists( "--fps" ) ) { dial_size = options.value( "--dial-size,--dial", 0 ); }
        std::string s = options.value< std::string >( "--size" );
        std::vector< std::string > v = comma::split( s, ',' );
        if( v.size() != 2 && v.size() != 3 ) { std::cerr << "image-accumulate: expected --size=<value>; got \"" << s << "\"" << std::endl; return 1; }
        block_size = boost::lexical_cast< unsigned int >( v[0] );
        row_size = boost::lexical_cast< unsigned int >( v[1] );
        scale = comma::csv::ascii< std::pair< double, double > >().get( options.value< std::string >( "--scale", "0,255" ) );
        scaled< unsigned char > scaled( scale );
        fps = options.optional< double >( "--fps" );
        sin_cos = precomputed_sin_cos_();
        channel::options default_channel_options( options );
        std::vector< channel::options > channel_options;
        if( csv.has_field( "id" ) )
        {
            const std::vector< std::string >& v = options.values< std::string >( "--id" );
            if( v.empty() ) { std::cerr << "image-accumulate: 'id' field given, please specify at least one --id" << std::endl; return 1; }
            for( unsigned int i = 0; i < v.size(); ++i )
            {
                channel_options.push_back( comma::name_value::parser( ';', '=' ).get< channel::options >( v[i], default_channel_options ) );
                ids[ boost::lexical_cast< unsigned int >( comma::split( v[i], ';' )[0] ) ] = i;
            }
        }
        else
        {
            ids[0] = 0;
            channel_options.push_back( default_channel_options );
        }
        in.values.resize( row_size );
        has_block = csv.has_field( "block" );
        has_row = csv.has_field( "row" );
        has_angle = csv.has_field( "angle" );
        if( has_row && has_angle ) { std::cerr << "image-accumulate: in input fields, expected either 'row' or 'angle'; got both" << std::endl; return 1; }
        comma::csv::input_stream< input > istream( std::cin, csv, in ); // todo: watch performance!
        std::string default_output_options = "no-header;rows=" + boost::lexical_cast< std::string >( polar ? row_size * 2 + 1 : block_size ) + ";cols=" + boost::lexical_cast< std::string >( polar ? ( row_size * 2 + 1 ) * ids.size() : row_size * ids.size() ) + ";type=3ub";
        std::string output_options_string = options.value( "--output", default_output_options );
        output_options = comma::name_value::parser( ';', '=' ).get< snark::cv_mat::serialization::options >( output_options_string );
        if( options.exists( "--output-options" ) ) { std::cout << comma::name_value::parser( ';', '=' ).put< snark::cv_mat::serialization::options >( output_options ) << std::endl; return 0; }
        serialization.reset( new snark::cv_mat::serialization( output_options ) );
        image = cv::Mat::zeros( block_size, row_size * ids.size(), output_options.get_header().type );
        pixel_size = output_options.get_header().type == CV_8UC3 ? 3 : 1; // quick and dirty
        boost::ptr_vector< channel > channels;
        for( unsigned int i = 0; i < ids.size(); ++i ) { channels.push_back( new channel( i, channel_options[i] ) ); }
        if( fps ) { output_thread.reset( new boost::thread( &output_ ) ); }
        while( !is_shutdown && std::cin.good() && !std::cin.eof() )
        {
            const input* p = istream.read();
            if( p )
            {
                Ids::const_iterator it = ids.find( p->id );
                if( it == ids.end() ) { continue; }
                if( !channels[ it->second ].draw( p ) ) { break; }
            }
            else
            {
                for( Ids::const_iterator it = ids.begin(); it != ids.end(); ++it )
                {
                    channels[ it->second ].draw( p );
                }
                break;
            }
        }
    }
    catch( std::exception& ex )
    {
        std::cerr << "image-accumulate: " << ex.what() << std::endl;
        result = 1;
    }
    catch( ... )
    {
        std::cerr << "image-accumulate: unknown exception" << std::endl;
        result = 1;
    }
    done = true;
    if( fps ) { output_thread->join(); }
    return result;
}
