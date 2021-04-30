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

#include <algorithm>
#include <fstream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/thread.hpp>
#include <boost/type_traits.hpp>
#include <boost/unordered_map.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>
#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d.hpp>
#if CV_MAJOR_VERSION <= 2
#include <opencv2/contrib/contrib.hpp>
#endif // #if CV_MAJOR_VERSION <= 2
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <comma/base/exception.h>
#include <comma/base/types.h>
#include <comma/csv/ascii.h>
#include <comma/csv/stream.h>
#include <comma/csv/traits.h>
#include <comma/csv/options.h>
#include <comma/string/string.h>
#include <comma/name_value/parser.h>
#include <comma/name_value/serialize.h>
#include <comma/application/verbose.h>
#include "../camera/pinhole.h"
#include "../camera/traits.h"
#include "../../timing/time.h"
#include "../../timing/timestamped.h"
#include "../../timing/traits.h"
#include "filters.h"
#include "serialization.h"
#include "traits.h"
#include "depth_traits.h"
#include "../vegetation/filters.h"
#include "detail/accumulated.h"
#include "detail/arithmetic.h"
#include "detail/bitwise.h"
#include "detail/colors.h"
#include "detail/file.h"
#include "detail/load.h"
#include "detail/morphology.h"
#include "detail/ratio.h"
#include "detail/utils.h"

namespace {

    struct map_input_t
    {
        typedef double value_type;
        typedef comma::int32 key_type;
        key_type key;
        value_type value;
    };

    struct normalization
    {
        typedef std::pair< int, int > key_t;            // from type, to type
        typedef std::pair< double, double > scaling_t;  // scale, offset
        typedef boost::unordered_map< key_t, scaling_t > map_t;
        template < int From, int To >
        struct factors
        {
            typedef snark::cv_mat::depth_traits< From > from;
            typedef snark::cv_mat::depth_traits< To > to;
            static double scale() {
                return ( to::max_value() - to::min_value() ) / ( from::max_value() - from::min_value() );
            }
            static double offset() {
                return ( to::min_value() * from::max_value() - to::max_value() * from::min_value() ) / ( from::max_value() - from::min_value() );
            }
        };
        template< int From >
        struct factors< From, From >
        {
            static double scale() { return 1.0; }
            static double offset() { return 0.0; }
        };
        typedef boost::mpl::list< snark::cv_mat::depth_traits< CV_8U >::value_t
                                , snark::cv_mat::depth_traits< CV_8S >::value_t
                                , snark::cv_mat::depth_traits< CV_16U >::value_t
                                , snark::cv_mat::depth_traits< CV_16S >::value_t
                                , snark::cv_mat::depth_traits< CV_32S >::value_t
                                , snark::cv_mat::depth_traits< CV_32F >::value_t
                                , snark::cv_mat::depth_traits< CV_64F >::value_t > data_types;
        template< int from >
        struct inner
        {
            inner( map_t & m ) : m_( m ) {}
            template< typename kind >
            void operator()( kind )
            {
                enum { value = snark::cv_mat::traits_to_depth< kind >::value };
                m_[ std::make_pair( from, int( value ) ) ] = std::make_pair( factors< from, int( value ) >::scale(), factors< from, int( value ) >::offset() );
            }
            map_t & m_;
        };
        struct outer
        {
            outer( map_t & m ) : m_( m ) {}
            template< typename kind >
            void operator()( kind )
            {
                enum { value = snark::cv_mat::traits_to_depth< kind >::value };
                boost::mpl::for_each< data_types >( inner< int( value ) >( m_ ) );
            }
            map_t & m_;
        };
        static map_t create_map()
        {
            map_t m;
            boost::mpl::for_each< data_types >( outer( m ) );
            return m;
        }
    };

} // anonymous

namespace snark{ namespace cv_mat {

template < typename H > struct empty;

template < > struct empty< boost::posix_time::ptime > { static bool is_empty( const boost::posix_time::ptime& t ) { return t.is_not_a_date_time(); } };
template < > struct empty< std::vector< char > > { static bool is_empty( const std::vector< char >& v ) { return v.empty(); } };

template < typename H >
static bool is_empty( typename impl::filters< H >::value_type m, const typename impl::filters< H >::get_timestamp_functor& get_timestamp ) { return ( m.second.empty() && ( empty< H >::is_empty(m.first) || get_timestamp(m.first) == boost::posix_time::not_a_date_time ) ); }

static const boost::unordered_map< std::string, int > types_ = impl::fill_types();

std::string type_as_string( int t ) { return impl::type_as_string(t); }

template < typename H >
struct canny_impl_ {
    typedef typename impl::filters< H >::value_type value_type;
    value_type operator()( value_type m, double threshold1, double threshold2, int kernel_size )
    {
        value_type n;
        n.first = m.first;
        cv::Canny( m.second, n.second, threshold1, threshold2, kernel_size );
        return n;
    }
};

template < typename H >
struct cvt_color_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type m, unsigned int which )
    {
        value_type n;
        n.first = m.first;
        cv::cvtColor( m.second, n.second, which );
        return n;
    }
};

template < typename H, unsigned int In, int InType, unsigned int Out, int OutType >
struct pixel_format_impl_
{
    static const unsigned int ElementSize = sizeof( typename depth_traits< CV_MAT_DEPTH( InType ) >::value_t );
    static const unsigned int InBytes = In * ElementSize;
    typedef typename impl::filters< H >::value_type value_type;
    typedef typename depth_traits< CV_MAT_DEPTH( OutType ) >::value_t value_out_t;
    typedef std::array< std::array< std::function< value_out_t( const unsigned char byte ) >, Out >, InBytes > transforms_t;
    typedef std::array< std::array< std::array< value_out_t, Out >, 256 >, InBytes > lookup_table_t;

    // byte operations
    struct zero { value_out_t operator()( const unsigned char byte ) const { return 0; } };
    struct identity { value_out_t operator()( const unsigned char byte ) const { return byte; } };
    struct shift_byte
    {
        unsigned int left_shift;
        shift_byte( unsigned int left_shift ) : left_shift(left_shift) {}
        value_out_t operator()( const unsigned char byte ) const { return byte << left_shift; }
    };
    struct shift_quadbit
    {
        unsigned int right_shift;
        unsigned int left_shift;
        shift_quadbit( unsigned int right_shift, unsigned int left_shift ) : right_shift(right_shift), left_shift(left_shift) {}
        value_out_t operator()( const unsigned char byte ) const { return ( byte >> right_shift & 0xF ) << left_shift; }
    };

    static std::string quadbit_characters( unsigned int quadbits )
    {
        std::string c;
        for( unsigned int i = 0; i < quadbits; ++i ) { c += 'a' + i; }
        c += '0';
        return c;
    }
    static transforms_t parse( const std::array< std::string, Out >& formats )
    {
        std::string characters = quadbit_characters( InBytes * 2 );
        transforms_t transforms;
        std::fill_n( &transforms[0][0], InBytes * Out, zero() );
        for( unsigned int i = 0; i < formats.size(); ++i )
        {
            std::string format = formats[i];
            std::size_t pos = format.find_first_not_of( characters );
            if( pos != std::string::npos) { COMMA_THROW( comma::exception, "pixel format '" << format << "' contains invalid character: " << format[pos] ); }
            unsigned int quadbit = format.size() - 1 ;
            for( unsigned j = 0; j < format.size(); ++j, --quadbit )
            {
                char c = format[j];
                if( c == '0' ) { continue; }
                unsigned int index = ( c - 'a' ) / 2;
                bool high_quadbit = ( c - 'a' ) % 2 == 0;
                if( quadbit > 0 && high_quadbit && format[j + 1] == c + 1 )
                {
                    if( quadbit == 1 ) { transforms[ index ][ i ] = identity(); }
                    else { transforms[ index ][ i ] = shift_byte( (quadbit - 1) * 4 ); }
                    ++j;
                    --quadbit;
                }
                else
                {
                    transforms[ index ][ i ] = shift_quadbit( high_quadbit ? 4 : 0, quadbit * 4 );
                }
            }
        }
        return transforms;
    }
    static lookup_table_t generate_lookup_table( const transforms_t& transforms )
    {
        lookup_table_t table;
        for( unsigned int i = 0; i < InBytes; ++i )
        {
            for( unsigned int j = 0; j < 256; ++j )
            {
                for( unsigned int k = 0; k < Out; ++k )
                {
                    table[i][j][k] = transforms[i][k]( j );
                }
            }
        }
        return table;
    }

    std::string filter_name;
    lookup_table_t table;
    pixel_format_impl_( const std::string& filter_name, const std::array< std::string, Out >& formats ) : filter_name( filter_name ) , table( generate_lookup_table( parse( formats ) ) ) {}
    value_type operator()( const value_type& m )
    {
        if( m.second.type() != InType ) { COMMA_THROW( comma::exception, filter_name << ": expected type: " << type_as_string( InType ) << " (" << InType << "), got: " << type_as_string( m.second.type() ) << " (" << m.second.type() << ")" ); }
        if( m.second.cols % In ) { COMMA_THROW( comma::exception, filter_name << ": columns: " << m.second.cols << " is not divisible by " << In ); }
        cv::Mat mat( m.second.rows, m.second.cols / In * Out, OutType, 0.0 );
        unsigned int bytes = m.second.cols * ElementSize;
        for( unsigned int i = 0; int( i ) < m.second.rows; ++i )
        {
            const unsigned char *in = m.second.ptr( i );
            value_out_t *out = mat.ptr< value_out_t >( i );
            for( unsigned int j = 0; j < bytes; j += InBytes, in += InBytes, out += Out )
            {
                for( unsigned int k = 0; k < Out; ++k )
                {
                    for( unsigned int l = 0; l < InBytes; ++l )
                    {
                        out[k] |= table[l][ in[l] ][ k ];
                    }
                }
            }
        }
        return value_type( m.first, mat );
    }
};

template < typename H >
static typename impl::filters< H >::value_type unpack12_impl_( typename impl::filters< H >::value_type m )
{
    //if(m.second.channels()!=1) { COMMA_THROW( comma::exception, "expected one channel input, got: ", m.second.channels() );}
    if(m.second.type()!=CV_8UC1 && m.second.type()!=CV_16UC1) { COMMA_THROW( comma::exception, "expected CV_8UC1("<<int(CV_8UC1)<<") or CV_16UC1("<<int(CV_16UC1)<<") , got: "<< m.second.type() );}
    //number of bytes
    int size=m.second.cols;
    if(m.second.type()==CV_16UC1){size*=2;}
    if(size%3!=0) { COMMA_THROW( comma::exception, "size is not divisible by three: "<<size);}
    cv::Mat mat(m.second.rows, (2*size)/3, CV_16UC1);
    for(int j=0;j<m.second.rows;j++)
    {
        const unsigned char* ptr=m.second.ptr(j);
        unsigned short* out_ptr=mat.ptr<unsigned short>(j);
        for(int col=0, i=0;i<size;i+=3)
        {
            out_ptr[col++] = ((unsigned(ptr[i])<<4) +  (unsigned(ptr[i+1]&0xF0) >> 4)) << 4;
            out_ptr[col++] = ((unsigned(ptr[i+2])<<4) + unsigned(ptr[i+1]&0x0F)) << 4;
            //out_ptr[col++] = (unsigned(ptr[i+2])<<4 | unsigned(ptr[i+1]&0x0F) >> 4) << 4;
        }
   }
    return typename impl::filters< H >::value_type(m.first, mat);
}

template < typename H >
static typename impl::filters< H >::value_type head_impl_( typename impl::filters< H >::value_type m, unsigned int number_of_frames )
{
    static unsigned int frame_number = 0;
    if( frame_number < number_of_frames ) { frame_number++; return m; } else { return typename impl::filters< H >::value_type(); }
}

template < typename H >
static typename impl::filters< H >::value_type crop_impl_( typename impl::filters< H >::value_type m, unsigned int x, unsigned int y, unsigned int w, unsigned int h )
{
    cv::Mat cropped;
    m.second( cv::Rect( x, y, w, h ) ).copyTo( cropped );
    return typename impl::filters< H >::value_type( m.first, cropped );
}

typedef std::pair< unsigned int, unsigned int > tile_t;

template < typename H > class crop_tile_impl_
{
    public:
        typedef typename impl::filters< H >::value_type value_type;
        value_type operator()( value_type input, unsigned int count_x, unsigned int count_y, const std::vector< tile_t >& tiles, bool vertical )
        {
            unsigned int w = input.second.cols / count_x;
            unsigned int h = input.second.rows / count_y;
            unsigned int s = tiles.empty() ? count_x * count_y : tiles.size();
            value_type output( input.first, cv::Mat( vertical ? h * s : h, vertical ? w : w * s, input.second.type() ) );
            if( tiles.empty() ) { for( unsigned int j( 0 ), n( 0 ); j < count_y; ++j ) { for( unsigned int i = 0; i < count_x; ++i, ++n ) { copy_tile_( input.second, output.second, n, i, j, w, h, vertical ); } } }
            else { for( std::size_t i = 0; i < tiles.size(); ++i ) { copy_tile_( input.second, output.second, i, tiles[i].first, tiles[i].second, w, h, vertical ); } }
            return output;
        }
        
    private:
        static void copy_tile_( const cv::Mat& input, cv::Mat& output, unsigned int n, unsigned int i, unsigned int j, unsigned int w, unsigned int h, bool vertical )
        {
            cv::Mat tile( output, cv::Rect( vertical ? 0 : n * w, vertical ? n * h: 0, w, h ) );
            cv::Mat( input, cv::Rect( i * w, j * h, w, h ) ).copyTo( tile );
        }
};

template < typename H > struct tile_impl_
{
    typename impl::filters< H >::value_type operator()( typename impl::filters< H >::value_type input, unsigned int width, unsigned int height, bool vertical ) { return crop_tile_impl_< H >()( input, input.second.cols / width, input.second.rows / height, std::vector< tile_t >(), vertical ); }
};

template < typename H > struct untile_impl_
{
    typedef typename impl::filters< H >::value_type value_type;
    value_type operator()( value_type input, unsigned int count_x, unsigned int count_y, bool vertical )
    {
        unsigned int count = count_x * count_y;
        unsigned int tile_width;
        unsigned int tile_height;
        if( vertical )
        {
            tile_width = input.second.cols;
            tile_height = input.second.rows / count;
            if( tile_height == 0 ) { COMMA_THROW( comma::exception, "untile: expected image height at least " << count << " rows; got: " << input.second.rows ); }
        }
        else
        {
            tile_width = input.second.cols / count;
            if( tile_width == 0 ) { COMMA_THROW( comma::exception, "untile: expected image width at least " << count << " cols; got: " << input.second.cols ); }
            tile_height = input.second.rows;
        }
        value_type output( input.first, cv::Mat( tile_height * count_y, tile_width * count_x, input.second.type() ) );
        for( unsigned int j( 0 ), n( 0 ); j < count_y; ++j )
        { 
            for( unsigned int i = 0; i < count_x; ++i, ++n )
            { 
                cv::Mat tile( output.second, cv::Rect( i * tile_width, j * tile_height, tile_width, tile_height ) );
                cv::Mat( input.second, cv::Rect( vertical ? 0 : n * tile_width, vertical ? n * tile_height : 0, tile_width, tile_height ) ).copyTo( tile );
            }
        }
        return output;
    }
};

typedef std::pair< unsigned int, unsigned int > stripe_t;

static unsigned int sum_up( unsigned int v, const stripe_t & s ){ return v + s.second; }

template < typename H >
struct crop_cols_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type input, const std::vector< stripe_t > & cols )
    {
        unsigned int h = input.second.rows;
        unsigned int w = std::accumulate( cols.begin(), cols.end(), 0, sum_up );
        value_type output( input.first, cv::Mat( h, w, input.second.type() ) );
        unsigned int offset = 0;
        for( std::size_t i = 0; i < cols.size(); ++i)
        {
            cv::Mat tile( output.second, cv::Rect( offset, 0, cols[i].second, h ) );
            cv::Mat( input.second, cv::Rect( cols[i].first, 0, cols[i].second, h ) ).copyTo( tile );
            offset += cols[i].second;
        }
        return output;
    }
};

template < typename H >
struct crop_rows_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type input, const std::vector< stripe_t > & rows )
    {
        unsigned int w = input.second.cols;
        unsigned int h = std::accumulate( rows.begin(), rows.end(), 0, sum_up );
        value_type output( input.first, cv::Mat( h, w, input.second.type() ) );
        unsigned int offset = 0;
        for( std::size_t i = 0; i < rows.size(); ++i)
        {
            cv::Mat tile( output.second, cv::Rect( 0, offset, w, rows[i].second ) );
            cv::Mat( input.second, cv::Rect( 0, rows[i].first, w, rows[i].second ) ).copyTo( tile );
            offset += rows[i].second;
        }
        return output;
    }
};

static const int bands_method_default = CV_REDUCE_AVG;

template < typename H >
struct bands_to_cols_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type input, bool bands_to_cols, const std::vector< stripe_t > & bands, int cv_reduce_method, int cv_reduce_dtype = -1 )
    {
        unsigned int w = input.second.cols;
        unsigned int h = input.second.rows;
        static std::set< int > good_input_depths = boost::assign::list_of( CV_8U )( CV_16U )( CV_16S )( CV_32F )( CV_64F );
        if ( good_input_depths.find( input.second.depth() ) == good_input_depths.end() )
        {
            COMMA_THROW( comma::exception, "depth of the " << type_as_string( input.second.type() ) << " image type is not supported by cv::reduce; consider 'convert-to' before processing" );
        }
        unsigned int output_type = cv_reduce_dtype >= 0
                                 ? CV_MAKETYPE( CV_MAT_DEPTH( cv_reduce_dtype ), input.second.channels() )
                                 : input.second.type() ;
        value_type output( input.first, cv::Mat( bands_to_cols ? h : bands.size()
                                                        , bands_to_cols ? bands.size() : w
                                                        , output_type ) );
        for( std::size_t i = 0; i < bands.size(); ++i )
        {
            cv::Mat intile( input.second, bands_to_cols
                                        ? cv::Rect( bands[i].first, 0, bands[i].second, h )
                                        : cv::Rect( 0, bands[i].first, w, bands[i].second ) );
            cv::Mat outtile( output.second, bands_to_cols
                                          ? cv::Rect( i, 0, 1, h )
                                          : cv::Rect( 0, i, w, 1 ) );
            int cv_reduce_dim = bands_to_cols ? 1 : 0 ;
            cv::reduce( intile, outtile, cv_reduce_dim, cv_reduce_method, output_type );
        }
        return output;
    }
};

template < typename H >
struct cols_to_channels_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type input, bool cols_to_channels, const std::vector< unsigned int > & values, double padding_value, unsigned int repeat )
    {
        if ( input.second.channels() != 1 ) { COMMA_THROW( comma::exception, "input image for cols-to-channels operation must have a single channel" ); }
        std::string element = cols_to_channels ? "column" : "row";

        unsigned int w = input.second.cols;
        unsigned int h = input.second.rows;

        unsigned int min_element = *std::min_element( values.begin(), values.end() );
        if ( min_element >= ( cols_to_channels ? w : h ) ) { COMMA_THROW( comma::exception, "the first output " << element << " is outside of the input image" ); }
        unsigned int output_w = cols_to_channels
                              ? ( repeat ? ( w - min_element ) / repeat : 1 )
                              : w ;
        unsigned int output_h = cols_to_channels
                              ? h
                              : ( repeat ? ( h - min_element ) / repeat : 1 ) ;
        unsigned int output_c = values.size() == 2 ? 3 : values.size();
        unsigned int output_t = CV_MAKETYPE( input.second.depth(), output_c );
        value_type output( input.first, cv::Mat( output_h, output_w, output_t ) );

        std::vector< int > from_to;
        from_to.reserve( 2 * output_c );
        for ( size_t i = 0; i < output_c; ++i ) { from_to.push_back( i ); from_to.push_back( i ); }

        std::vector< cv::Mat > src( output_c );
        std::vector< cv::Mat > dst( 1 );
        cv::Mat padding = ( cols_to_channels
                          ? cv::Mat::ones( h, 1, input.second.type() )
                          : cv::Mat::ones( 1, w, input.second.type() ) ) * padding_value; // instantiation is lazy
        std::vector< unsigned int > indices( values );
        for( size_t i = indices.size(); i < output_c; ++i ) { src[i] = padding; }
        for ( unsigned int opos = 0; opos < ( cols_to_channels ? output_w : output_h ) ; ++opos )
        {
            for( size_t i = 0; i < indices.size(); ++i )
            {
                unsigned int j = indices[i];
                if ( j >= ( cols_to_channels ? w : h ) ) {
                    src[i] = padding;
                } else {
                    src[i] = cols_to_channels
                           ? cv::Mat( input.second, cv::Rect( j, 0, 1, h ) )
                           : cv::Mat( input.second, cv::Rect( 0, j, w, 1 ) ) ;
                }
                indices[i] += repeat;
            }
            dst[0] = cols_to_channels
                   ? cv::Mat( output.second, cv::Rect( opos, 0, 1, h ) )
                   : cv::Mat( output.second, cv::Rect( 0, opos, w, 1 ) ) ;
            cv::mixChannels( src, dst, &from_to[0], output_c );
        }
        return output;
    }
};

template < typename H >
struct channels_to_cols_impl_ {
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type input, bool channels_to_cols )
    {
        unsigned int w = input.second.cols;
        unsigned int h = input.second.rows;

        unsigned int input_c = input.second.channels();
        unsigned int output_t = CV_MAKETYPE( input.second.depth(), 1 );
        value_type output( input.first, cv::Mat( channels_to_cols ? h : input_c * h
                                                        , channels_to_cols ? input_c * w : w
                                                        , output_t ) );

        std::vector< int > from_to;
        from_to.reserve( 2 * input_c );
        for ( size_t i = 0; i < input_c; ++i ) { from_to.push_back( i ); from_to.push_back( i ); }

        std::vector< cv::Mat > src( 1 );
        std::vector< cv::Mat > dst( input_c );
        for ( unsigned int ipos = 0; ipos < ( channels_to_cols ? w : h ) ; ++ipos )
        {
            for( size_t i = 0; i < input_c; ++i )
            {
                dst[i] = cv::Mat( output.second, channels_to_cols
                                               ? cv::Rect( ipos * input_c + i, 0, 1, h )
                                               : cv::Rect( 0, ipos * input_c + i, w, 1 ) );
            }
            src[0] = cv::Mat( input.second, channels_to_cols
                                          ? cv::Rect( ipos, 0, 1, h )
                                          : cv::Rect( 0, ipos, w, 1 ) );
            cv::mixChannels( src, dst, &from_to[0], input_c );
        }

        return output;
    }
};

template < typename H >
class accumulate_impl_
{
    public:
        typedef typename impl::filters< H >::value_type value_type;
        enum how_values { sliding = 0, fixed = 1 };
        accumulate_impl_( unsigned int how_many, how_values how, bool reverse ) : how_many_ ( how_many ), how_( how ), count_( 0 ), reverse_( reverse ), index_( 0 ), initialised_ ( false ) {}
        value_type operator()( value_type input )
        {
            if( !initialised_ )
            {
                initialised_ = true;
                cols_ = input.second.cols;
                h_ = input.second.rows;
                rows_ = h_ * how_many_;
                type_ = input.second.type();
                accumulated_image_[0] = cv::Mat::zeros( rows_, cols_, type_ );
                if( how_ == fixed ) { accumulated_image_[1] = cv::Mat::zeros( rows_, cols_, type_ ); }
                rect_for_new_data_ = reverse_ ? cv::Rect( 0, rows_ - h_, cols_, h_ ) : cv::Rect( 0, 0, cols_, h_ );
                rect_for_old_data_ = reverse_ ? cv::Rect( 0, 0, cols_, rows_ - h_ ) : cv::Rect( 0, h_, cols_, rows_ - h_ );
                rect_to_keep_ = reverse_ ? cv::Rect( 0, h_, cols_, rows_ - h_ ) : cv::Rect( 0, 0, cols_, rows_ - h_ );
            }
            if( input.second.cols != cols_ ) { COMMA_THROW( comma::exception, "accumulate: expected input image with " << cols_ << " columns, got " << input.second.cols << " columns"); }
            if( input.second.rows != h_ ) { COMMA_THROW( comma::exception, "accumulate: expected input image with " << h_ << " rows, got " << input.second.rows << " rows"); }
            if( input.second.type() != type_ ) { COMMA_THROW( comma::exception, "accumulate: expected input image of type " << type_ << ", got type " << input.second.type() << " rows"); }
            value_type output( input.first, cv::Mat( accumulated_image_[0].size(), accumulated_image_[0].type() ) );
            switch( how_ )
            {
                case sliding:
                {
                    cv::Mat new_data( output.second, rect_for_new_data_ );
                    input.second.copyTo( new_data );
                    cv::Mat old_data( output.second, rect_for_old_data_ );
                    cv::Mat( accumulated_image_[0], rect_to_keep_ ).copyTo( old_data );
                    output.second.copyTo( accumulated_image_[0] );
                    break;
                }
                case fixed:
                {
                    cv::Mat r( accumulated_image_[index_], cv::Rect( 0, h_ * ( reverse_ ? how_many_ - count_ - 1 : count_ ), cols_, h_ ) );
                    input.second.copyTo( r );
                    accumulated_image_[ 1 - index_ ].copyTo( output.second );
                    ++count_;
                    if( count_ == how_many_ ) { count_ = 0; index_ = 1 - index_; }
                    break;
                }
            }
            return output;
        }
    private:
        unsigned int how_many_;
        how_values how_;
        unsigned int count_;
        bool reverse_;
        unsigned int index_;
        bool initialised_;
        int cols_, h_, rows_, type_;
        cv::Rect rect_for_new_data_, rect_for_old_data_, rect_to_keep_;
        std::array< cv::Mat, 2 > accumulated_image_;
};

template < typename H >
static typename impl::filters< H >::value_type convert_to_impl_( typename impl::filters< H >::value_type m, int type, boost::optional< double > scale, boost::optional< double > offset )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    if ( !scale ) {
        static const normalization::map_t & map = normalization::create_map();
        normalization::map_t::const_iterator s = map.find( std::make_pair( m.second.depth(), CV_MAT_DEPTH( type ) ) );
        if ( s == map.end() ) { COMMA_THROW( comma::exception, "cannot find normalization scaling from type " << type_as_string( m.second.type() ) << " to type " << type_as_string( type ) ); }
        scale = s->second.first;
        offset = s->second.second;
    }
    m.second.convertTo( n.second, type, *scale, *offset );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type flip_impl_( typename impl::filters< H >::value_type m, int how )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::flip( m.second, n.second, how );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type resize_impl_( typename impl::filters< H >::value_type m, unsigned int width, unsigned int height, double w, double h, int interpolation )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::resize( m.second, n.second, cv::Size( width ? width : m.second.cols * w, height ? height : m.second.rows * h ), 0, 0, interpolation );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type brightness_impl_( typename impl::filters< H >::value_type m, double scale, double offset )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    n.second = (m.second * scale) + offset;
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type colour_map_impl_( typename impl::filters< H >::value_type m, int type )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::applyColorMap( m.second, n.second, type );
    return n;
}

template < typename H >
struct tee_impl_
{
    typedef typename impl::filters< H >::value_type value_type;
    boost::function< value_type( value_type ) > tee;
    tee_impl_( const boost::function< value_type( value_type ) >& tee ): tee( tee ) {}
    value_type operator()( value_type m )
    { 
        value_type n;
        n.first = m.first;
        m.second.copyTo( n.second );
        tee( m ).second;
        return n;
    }
};

template < typename H >
struct mask_impl_
{
    typedef typename impl::filters< H >::value_type value_type;
    boost::function< value_type( value_type ) > mask;
    mask_impl_( const boost::function< value_type( value_type ) >& mask ): mask( mask ) {}
    value_type operator()( value_type m )
    {
        value_type n;
        n.first = m.first;
        const cv::Mat & f = mask( m ).second;
        if ( f.depth() != CV_8U ) { COMMA_THROW( comma::exception, "the mask type is " << type_as_string( f.type() ) << ", must have CV_8U depth; use convert-to explicitly" ); }
        m.second.copyTo( n.second, f );
        return n;
    }
};

template < typename H >
struct bitwise_impl_
{
    typedef typename impl::filters< H >::value_type value_type;
    boost::function< value_type( value_type ) > operation;
    bitwise_impl_( const boost::function< value_type( value_type ) >& operation ): operation( operation ) {}
    value_type operator()( value_type m ) { return value_type( m.first, operation( m ).second ); }
};

struct blur_t
{
    enum types { box, gaussian, median, bilateral, adaptive_bilateral };
    types blur_type;
    cv::Size kernel_size;
    cv::Point2d std;
    int neighbourhood_size;
    double sigma_colour;
    double sigma_space;

    static types from_string( const std::string& s )
    {
        if( s == "box" ) { return box; }
        if( s == "gaussian" ) { return gaussian; }
        if( s == "median") { return median; }
        if( s == "bilateral") { return bilateral; }
        if( s == "adaptive-bilateral") { return adaptive_bilateral; }
        COMMA_THROW( comma::exception, "expected blur type; got: \"" << s << "\"" );
    }
    blur_t() : neighbourhood_size(0), sigma_colour(0), sigma_space(0) {}
};

template < typename H >
static typename impl::filters< H >::value_type blur_impl_( typename impl::filters< H >::value_type m, blur_t params )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    switch( params.blur_type )
    {
        case blur_t::box:
            cv::blur(m.second, n.second, params.kernel_size);
            break;
        case blur_t::gaussian:
            cv::GaussianBlur(m.second, n.second, params.kernel_size, params.std.x, params.std.y);
            break;
        case blur_t::median:
            cv::medianBlur(m.second, n.second, params.neighbourhood_size);
            break;
        case blur_t::bilateral:
            cv::bilateralFilter(m.second, n.second, params.neighbourhood_size, params.sigma_colour, params.sigma_space);
            break;
        case blur_t::adaptive_bilateral:
            //cv::adaptiveBilateralFilter(m.second, n.second, params.kernel_size, params.sigma_colour, params.sigma_space);
            break;
    }
    return n;
}

struct threshold_t
{
    enum types { binary = CV_THRESH_BINARY
               , binary_inv = CV_THRESH_BINARY_INV
               , trunc = CV_THRESH_TRUNC
               //, trunc_inv = CV_THRESH_TRUNC | CV_THRESH_BINARY_INV // == CV_THRESH_TOZERO
               , tozero = CV_THRESH_TOZERO
               , tozero_inv = CV_THRESH_TOZERO_INV };

    static types from_string( const std::string& s )
    {
        if( s.empty() || s == "binary" ) { return binary; }
        if( s == "binary_inv" ) { return binary_inv; }
        if( s == "trunc" ) { return trunc; }
        if( s == "tozero" ) { return tozero; }
        if( s == "tozero_inv" ) { return tozero_inv; }
        COMMA_THROW( comma::exception, "expected threshold type, got: \"" << s << "\"" );
    }
};

template < typename H >
static typename impl::filters< H >::value_type threshold_impl_( typename impl::filters< H >::value_type m, double threshold, double max_value, threshold_t::types type, bool otsu )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::threshold( m.second, n.second, threshold, max_value, otsu ? type | CV_THRESH_OTSU : type );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type transpose_impl_( typename impl::filters< H >::value_type m )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::transpose( m.second, n.second );
    return n;
}

int single_channel_type( int t )
{
    switch( t )
    {
        case CV_8UC1:
        case CV_8UC2:
        case CV_8UC3:
        case CV_8UC4:
            return CV_8UC1;
        case CV_8SC1:
        case CV_8SC2:
        case CV_8SC3:
        case CV_8SC4:
            return CV_8SC1;
        case CV_16UC1:
        case CV_16UC2:
        case CV_16UC3:
        case CV_16UC4:
            return CV_16UC1;
        case CV_16SC1:
        case CV_16SC2:
        case CV_16SC3:
        case CV_16SC4:
            return CV_16SC1;
        case CV_32SC1:
        case CV_32SC2:
        case CV_32SC3:
        case CV_32SC4:
            return CV_32SC1;
        case CV_32FC1:
        case CV_32FC2:
        case CV_32FC3:
        case CV_32FC4:
            return CV_32FC1;
        case CV_64FC1:
        case CV_64FC2:
        case CV_64FC3:
        case CV_64FC4:
            return CV_64FC1;
    }
    return CV_8UC1;
}

template < typename H >
static typename impl::filters< H >::value_type split_impl_( typename impl::filters< H >::value_type m )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    n.second = cv::Mat( m.second.rows * m.second.channels(), m.second.cols, single_channel_type( m.second.type() ) ); // todo: check number of channels!
    std::vector< cv::Mat > channels;
    channels.reserve( m.second.channels() );
    for( unsigned int i = 0; i < static_cast< unsigned int >( m.second.channels() ); ++i )
    {
        channels.push_back( cv::Mat( n.second, cv::Rect( 0, i * m.second.rows, m.second.cols, m.second.rows ) ) );
    }
    cv::split( m.second, channels );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type kmeans_impl_( typename impl::filters< H >::value_type m, int k )
{
    cv::Mat pixels = m.second.reshape(1, m.second.rows*m.second.cols);
    cv::Mat classes;
    cv::Mat centers;
    int attempts = 5;
    cv::kmeans(pixels, k, classes, cv::TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 0.01 ), attempts, cv::KMEANS_PP_CENTERS, centers);
    cv::Mat out = cv::Mat::zeros(m.second.rows, m.second.cols, m.second.type());
    out = out.reshape(1, m.second.rows*m.second.cols);
    for (int p = 0; p < out.rows ; p++) { centers.row(classes.at<int>(p)).copyTo(out.row(p)); }
    m.second = out.reshape(m.second.channels(), m.second.rows);
    return m;
}

template < typename H >
class log_impl_ // quick and dirty; poor-man smart pointer, since boost::mutex is non-copyable
{
    public:
        typedef typename impl::filters< H >::value_type value_type;
        typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;

        log_impl_( const std::string& filename, const get_timestamp_functor& get_timestamp ) : logger_( new logger( filename, get_timestamp ) ) {}

        log_impl_( const std::string& directory, boost::posix_time::time_duration period, bool index, const get_timestamp_functor& get_timestamp ) : logger_( new logger( directory, period, index, get_timestamp ) ) {}

        log_impl_( const std::string& directory, unsigned int size, bool index, const get_timestamp_functor& get_timestamp ) : logger_( new logger( directory, size, index, get_timestamp ) ) {}

        value_type operator()( value_type m ) { return logger_->operator()( m ); }

        class logger
        {
            public:

                logger( const std::string& filename, const get_timestamp_functor& get_timestamp )
                    : ofstream_( new std::ofstream( &filename[0] ) )
                    , serialization_( "t,rows,cols,type", comma::csv::format( "t,3ui" ) ), size_( 0 ), count_( 0 ), get_timestamp_(get_timestamp)
                {
                    if( !ofstream_->is_open() ) { COMMA_THROW( comma::exception, "failed to open \"" << filename << "\"" ); }
                }

                logger( const std::string& directory, boost::posix_time::time_duration period, bool index, const get_timestamp_functor& get_timestamp )
                    : directory_( directory ), serialization_( "t,rows,cols,type", comma::csv::format( "t,3ui" ) )
                    , period_( period ), size_( 0 ), count_( 0 ), index_(index, directory), get_timestamp_(get_timestamp) { }

                logger( const std::string& directory, unsigned int size, bool index, const get_timestamp_functor& get_timestamp )
                    : directory_( directory ), serialization_( "t,rows,cols,type", comma::csv::format( "t,3ui" ) )
                    , size_( size ), count_( 0 ), index_(index, directory), get_timestamp_(get_timestamp) { }

                ~logger() { if( ofstream_ ) { ofstream_->close(); } }

                value_type operator()( value_type m )
                {
                    if( m.second.empty() ) { return m; } // quick and dirty, end of stream
                    boost::mutex::scoped_lock lock( mutex_ ); // somehow, serial_in_order still may have more than one instance of filter run at a time
                    update_on_size_();
                    update_on_time_( m );
                    if( !ofstream_ )
                    {
                        std::string filename = directory_ + '/' + snark::cv_mat::impl::make_filename( get_timestamp_(m.first), "bin");
                        ofstream_.reset( new std::ofstream( &filename[0] ) );
                        if( !ofstream_->is_open() ) { COMMA_THROW( comma::exception, "failed to open \"" << filename << "\"" ); }
                    }
                    serialization_.write(*ofstream_, m);
//                     writer< H >::write( serialization_, *ofstream_, m );
                    index_.write( get_timestamp_(m.first), m, serialization_.size( m ) );
//                     index_.write( m, writer< H >::size( serialization_, m ) );
                    return m;
                }

                struct indexer
                {
                    boost::posix_time::ptime t;
                    comma::uint16 file;
                    comma::uint64 offset;
                    boost::scoped_ptr< std::ofstream > filestream;
                    boost::scoped_ptr< comma::csv::binary_output_stream< indexer > > csv_stream;

                    indexer( ) : file( 0 ), offset( 0 ) {}

                    indexer( bool enabled, const std::string& directory ) : file ( 0 ), offset( 0 )
                    {
                        if( !enabled ) { return; }
                        std::string index_file = directory + "/index.bin";
                        filestream.reset( new std::ofstream( &index_file[0] ) );
                        if( !filestream->is_open() ) { COMMA_THROW( comma::exception, "failed to open \"" << index_file << "\"" ); }
                        csv_stream.reset( new comma::csv::binary_output_stream< indexer >( *filestream ) );
                    }

                    ~indexer() { if ( filestream ) { filestream->close(); } }

                    void increment_file() { ++file; offset = 0; }

                    void write( const boost::posix_time::ptime& time, const value_type& m, std::size_t size )
                    {
                        t = time;
                        if( csv_stream )
                        {
                            csv_stream->write( *this );
                            csv_stream->flush();
                        }
                        offset += size;
                    }
                };

            private:
                boost::mutex mutex_;
                std::string directory_;
                boost::scoped_ptr< std::ofstream > ofstream_;
                snark::cv_mat::serialization serialization_;
                boost::optional< boost::posix_time::time_duration > period_;
                boost::posix_time::ptime start_;
                unsigned int size_;
                unsigned int count_;
                indexer index_;
                const get_timestamp_functor get_timestamp_;

                void update_on_size_()
                {
                    if( size_ == 0 ) { return; }
                    if( count_ < size_ ) { ++count_; return; }
                    count_ = 1;
                    ofstream_->close();
                    ofstream_.reset();
                    index_.increment_file();
                }

                void update_on_time_( value_type m )
                {
                    if( !period_ ) { return; }
                    boost::posix_time::ptime t = get_timestamp_( m.first );
                    if( start_.is_not_a_date_time() ) { start_ = t; return; }
                    if( ( t - start_ ) < *period_ ) { return; }
                    start_ = t;
                    ofstream_->close();
                    ofstream_.reset();
                    index_.increment_file();
                }
        };

    private:
        boost::shared_ptr< logger > logger_; // todo: watch performance
};

template < typename H >
static typename impl::filters< H >::value_type merge_impl_( typename impl::filters< H >::value_type m, unsigned int nchannels )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    if( m.second.rows % nchannels != 0 ) { COMMA_THROW( comma::exception, "merge: expected " << nchannels << " horizontal strips of equal height, got " << m.second.rows << " rows, which is not a multiple of " << nchannels ); }
    std::vector< cv::Mat > channels( nchannels );
    for( std::size_t i = 0; i < nchannels; ++i ) { channels[i] = cv::Mat( m.second, cv::Rect( 0, i * m.second.rows / nchannels, m.second.cols, m.second.rows / nchannels ) ); }
    cv::merge( channels, n.second );
    return n;
}

template < typename H >
class view_impl_
{
public:
    typename impl::filters< H >::get_timestamp_functor get_timestamp;
    std::string name;
    unsigned int delay;
    std::string suffix;

    view_impl_< H >( const typename impl::filters< H >::get_timestamp_functor& get_timestamp, const std::string& name, unsigned int delay,const std::string& suffix )
        : get_timestamp( get_timestamp )
        , name( make_name_( name ) )
        , delay( delay )
        ,suffix(suffix)
    {
    }
    
    typename impl::filters< H >::value_type operator()( typename impl::filters< H >::value_type m )
    {
        cv::imshow( &name[0], m.second );
        char c = cv::waitKey( delay );
        if( c == 27 ) { return typename impl::filters< H >::value_type(); } // HACK to notify application to exit
        if( c == ' ' ) { cv::imwrite( snark::cv_mat::impl::make_filename( get_timestamp( m.first ), suffix ), m.second ); }
        if( c>='0' && c<='9') { cv::imwrite( snark::cv_mat::impl::make_filename( get_timestamp( m.first ), suffix, unsigned(c-'0') ), m.second ); }
        return m;
    }
    
private:
    static std::string make_name_( const std::string& name ) // quick and dirty
    {
        static unsigned int count = 0;
        const std::string& n = name.empty() ? boost::lexical_cast< std::string >( count ) : name;
        ++count;
        return n;
    }
};

namespace drawing {

struct shape
{
    cv::Scalar color;
    int thickness;
    int line_type;
    int shift;
    shape() : thickness( 1 ), line_type( 8 ), shift( 0 ) {}
    shape( const cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0 ) : color( color ), thickness( thickness ), line_type( line_type ), shift( shift ) {}
};

struct circle : public shape
{
    cv::Point center;
    int radius;
    circle() {}
    circle( const cv::Point& center, int radius, const cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0 ) : shape( color, thickness, line_type, shift ), center( center ), radius( radius ) {}
    void draw( cv::Mat m ) const { cv::circle( m, center, radius, color, thickness, line_type, shift ); }
};

struct rectangle : public shape
{
    cv::Point upper_left;
    cv::Point lower_right;
    rectangle() {};
    rectangle( const cv::Point& upper_left, const cv::Point& lower_right, const cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0 ) : shape( color, thickness, line_type, shift ), upper_left( upper_left ), lower_right( lower_right ) {}
    void draw( cv::Mat m ) const { cv::rectangle( m, upper_left, lower_right, color, thickness, line_type, shift ); }
};

struct cross : public shape
{
    cv::Point centre;
    cross(): centre( 0, 0 ) {};
    cross( const cv::Point& centre, const cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0 ) : shape( color, thickness, line_type, shift ), centre( centre ) {}
    void draw( cv::Mat m ) const
    {
        cv::line( m, cv::Point( centre.x, 0 ), cv::Point( centre.x, m.size().height ), color, thickness, line_type, shift );
        cv::line( m, cv::Point( 0, centre.y ), cv::Point( m.size().width, centre.y ), color, thickness, line_type, shift );
    }
};

} // namespace drawing {

template < typename H >
static typename impl::filters< H >::value_type circle_impl_( typename impl::filters< H >::value_type m, const drawing::circle& circle ) { circle.draw( m.second ); return m; }

template < typename H >
static typename impl::filters< H >::value_type rectangle_impl_( typename impl::filters< H >::value_type m, const drawing::rectangle& rectangle ) { rectangle.draw( m.second ); return m; }

template < typename H >
static typename impl::filters< H >::value_type cross_impl_( typename impl::filters< H >::value_type m, const drawing::cross& cross ) { cross.draw( m.second ); return m; }

template < typename H >
struct encode_impl_ {
    typedef typename impl::filters< H >::value_type value_type;
    typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;
    const get_timestamp_functor get_timestamp_;

    encode_impl_< H >( const get_timestamp_functor& gt ) : get_timestamp_(gt) {}
    value_type operator()( const value_type& m, const std::string& type, const boost::optional<int>& quality )
    {
        if( is_empty< H >( m, get_timestamp_ ) ) { return m; }
        snark::cv_mat::impl::check_image_type( m.second, type );
        std::vector< unsigned char > buffer;
        std::vector<int> params;
        if (quality) { params = snark::cv_mat::impl::imwrite_params(type, *quality); }
        std::string format = "." + type;
        cv::imencode( format, m.second, buffer, params );
        typename impl::filters< H >::value_type p;
        p.first = m.first;
        p.second = cv::Mat( buffer.size(), 1, CV_8UC1 );
        ::memcpy( p.second.data, &buffer[0] , buffer.size() );
        return p;
    }

};

template < typename T > static comma::csv::options make_csv_options_( bool binary ) // quick and dirty
{
    comma::csv::options csv;
    if( binary ) { csv.format( comma::csv::format::value< snark::timestamped< T > >() ); }
    return csv;
}

template < typename T > static T cv_read_( const std::string& filename = "", const std::string& path = "" )
{
    if( filename.empty() ) { return T(); }
    T t;
    if( !boost::filesystem::is_regular_file( filename ) ) { COMMA_THROW( comma::exception, "file not found: \"" << filename << "\"" ); }
    const std::vector< std::string > v = comma::split( filename, '.' );
    cv::FileStorage f( filename, cv::FileStorage::READ | ( v.size() > 1 || v.back() == "xml" ? cv::FileStorage::FORMAT_XML : cv::FileStorage::FORMAT_YAML ) );
    cv::FileNode n = f[path];
    t.read( n );
    return t;
}

#if CV_MAJOR_VERSION <= 2
template < typename H >
struct simple_blob_impl_ {
    typedef typename impl::filters< H >::value_type value_type;
    typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;
    const get_timestamp_functor get_timestamp_;

    simple_blob_impl_< H >( const get_timestamp_functor& get_timestamp ) : get_timestamp_(get_timestamp) {}
    value_type operator()( const value_type& m, const cv::SimpleBlobDetector::Params& params, bool is_binary )
    {
        static cv::SimpleBlobDetector detector( params ); // quick and dirty
        std::vector< cv::KeyPoint > key_points;
        detector.detect( m.second, key_points );
        static comma::csv::output_stream< snark::timestamped< cv::KeyPoint > > os( std::cout, make_csv_options_< snark::timestamped< cv::KeyPoint > >( is_binary ) );
        for( unsigned int i = 0; i < key_points.size(); ++i ) { os.write( snark::timestamped< cv::KeyPoint >( get_timestamp_( m.first ), key_points[i] ) ); }
        return m;
    }
};
#endif // #if CV_MAJOR_VERSION <= 2

template < typename H >
struct grab_impl_ {
    typedef typename impl::filters< H >::value_type value_type;
    typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;
    const get_timestamp_functor get_timestamp_;

    grab_impl_< H >( const get_timestamp_functor& get_timestamp ) : get_timestamp_(get_timestamp) {}
    value_type operator()( value_type m, const std::string& type, const boost::optional<int>& quality )
    {
        std::vector<int> params;
        if (quality) { params = snark::cv_mat::impl::imwrite_params(type, *quality); }
        cv::imwrite( snark::cv_mat::impl::make_filename( get_timestamp_(m.first), type ), m.second, params );
        return typename impl::filters< H >::value_type(); // HACK to notify application to exit
    }
};

template < typename H >
struct timestamp_impl_ {
    typedef typename impl::filters< H >::value_type value_type;
    typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;
    const get_timestamp_functor get_timestamp_;

    timestamp_impl_< H >( const get_timestamp_functor& gt ) : get_timestamp_(gt) {}

    value_type operator()( value_type m )
    {
        cv::rectangle( m.second, cv::Point( 5, 5 ), cv::Point( 228, 25 ), cv::Scalar( 0xffff, 0xffff, 0xffff ), CV_FILLED, CV_AA );
        cv::putText( m.second, boost::posix_time::to_iso_string( get_timestamp_(m.first) ), cv::Point( 10, 20 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 0 ), 1, CV_AA );
        return m;
    }
};

template < typename H >
class save_impl_
{
    public:
        typedef typename impl::filters< H >::value_type value_type;

        save_impl_( const std::string& filename, bool no_header, const boost::optional< int >& quality, bool do_index ) : filename_( filename ), index_( 0 ), quality_( quality ), do_index_( do_index )
        {
            snark::cv_mat::serialization::options options;
            options.no_header = no_header;
            serialization_ = snark::cv_mat::serialization( options );
            const auto& v = comma::split( filename, '.' );
            if( v.size() == 1 ) { COMMA_THROW( comma::exception, "save: expected filename with image type extension, got: \"" << filename << "\"" ); }
            basename_ = comma::join( v, v.size() - 1, '.' );
            type_ = v.back();
        }

        value_type operator()( value_type m )
        {
            if( m.second.empty() ) { return m; }
            std::vector< int > params;
            if( quality_ ) { params = snark::cv_mat::impl::imwrite_params( type_, *quality_ ); }
            std::string filename = do_index_ ? basename_ + '.' + boost::lexical_cast< std::string  >( index_ ) + '.' + type_ : filename_;
            if( type_ == "bin" )
            {
                std::ofstream ofs( filename );
                if( !ofs.is_open() ) { COMMA_THROW( comma::exception, "" ); }
                serialization_.write( ofs, m );
            }
            else
            {
                snark::cv_mat::impl::check_image_type( m.second, type_ );
                cv::imwrite( filename, m.second, params );
            }
            ++index_;
            return m;
        }
        
    private:
        snark::cv_mat::serialization serialization_;
        std::string filename_;
        std::string basename_;
        std::string type_;
        unsigned int index_;
        boost::optional< int > quality_;
        bool do_index_;
};

template < typename H >
struct count_impl_
{
    count_impl_() : count( 0 ) {}

    unsigned int count;
    typedef typename impl::filters< H >::value_type value_type;

    value_type operator()( value_type m )
    {
        cv::rectangle( m.second, cv::Point( 5, 5 ), cv::Point( 80, 25 ), cv::Scalar( 0xffff, 0xffff, 0xffff ), CV_FILLED, CV_AA );
        cv::putText( m.second, boost::lexical_cast< std::string >( count++ ), cv::Point( 10, 20 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 0 ), 1, CV_AA );
        return m;
    }
};

template < typename H >
static typename impl::filters< H >::value_type invert_impl_( const typename impl::filters< H >::value_type& m )
{
    if( m.second.type() != CV_8UC1 && m.second.type() != CV_8UC2 && m.second.type() != CV_8UC3 && m.second.type() != CV_8UC4 ) { COMMA_THROW( comma::exception, "expected image type ub, 2ub, 3ub, 4ub; got: " << type_as_string( m.second.type() ) ); }
    for( unsigned char* c = const_cast< unsigned char* >( m.second.datastart ); c < m.second.dataend; *c = 255 - *c, ++c );
    return m;
}

template < typename H >
static typename impl::filters< H >::value_type invert_brightness_impl_( typename impl::filters< H >::value_type m )
{
    if( m.second.type() != CV_8UC3 ) { COMMA_THROW( comma::exception, "expected image type 3ub; got: " << type_as_string( m.second.type() ) ); }
    cv::Mat n;
    cv::cvtColor( m.second, n, CV_RGB2HSV );
    for( unsigned char* c = const_cast< unsigned char* >( m.second.datastart ) + 2; c < n.dataend; *c = 255 - *c, c += 3 );
    cv::cvtColor( n, m.second, CV_HSV2RGB );
    return m;
}

template < typename H >
class clahe_impl_
{
    public:
        clahe_impl_( float clip_limit, const cv::Size& tile_size ) : clahe_( cv::createCLAHE( clip_limit, tile_size ) ) {}

        typedef typename impl::filters< H >::value_type value_type;

        value_type operator()( value_type m )
        {
            value_type n;
            n.first = m.first;
            if( m.second.channels() == 1 )
            {
                clahe_->apply( m.second, n.second );
            }
            else
            {
                std::vector< cv::Mat > channels( m.second.channels() );
                std::vector< cv::Mat > output_channels( m.second.channels() );
                cv::split( m.second, channels );
                for( int i = 0; i < m.second.channels(); ++i ) { clahe_->apply( channels[i], output_channels[i] ); }
                cv::merge( output_channels, n.second );
            }
            return n;
        }
    private:
        cv::Ptr< cv::CLAHE > clahe_;
};

template < typename H >
static typename impl::filters< H >::value_type equalize_histogram_impl_(typename impl::filters< H >::value_type m)
{
    if( single_channel_type( m.second.type() ) != CV_8UC1 ) { COMMA_THROW( comma::exception, "expected currently supported types: ub, 2ub, 3ub, 4ub; got: " << type_as_string( m.second.type() ) ); } //cv::equalizeHist only supports 8-bit single channel
    int chs=m.second.channels();
    std::vector<cv::Mat> planes;
    for( int i=0; i<chs; i++ ) { planes.push_back(cv::Mat(1,1,single_channel_type(m.second.type()))); }
    cv::split(m.second,planes);
    for(int i=0;i<chs;i++) { cv::equalizeHist( planes[i],planes[i] ); }
    cv::merge(planes,m.second);
    return m;
}

template < typename H >
static typename impl::filters< H >::value_type normalize_cv_impl_( typename impl::filters< H >::value_type m, int norm_type = cv::NORM_INF )
{
    //std::cerr<<"my type: "<<m.second.type()<<" ,CV_8UC3: "<<int(CV_8UC3)<<" ,CV_32FC3: "<<int(CV_32FC3)<<" ,CV_8UC1: "<<CV_8UC1<<std::endl;
    cv::normalize( m.second, m.second, 1, 0, norm_type, CV_32FC( m.second.channels() ) );
    return m;
}

static cv::Scalar max_scalar(const cv::Scalar& s, const cv::Scalar& t)
{
    cv::Scalar out;
    for(int i=0;i<4;i++)
        out[i]=std::max(s[i],t[i]);
    return out;
}

struct filter_table_t
{
    struct filter_i
    {
        virtual cv::Mat normalize_max(const cv::Mat& v)=0;
        virtual cv::Mat normalize_sum(const cv::Mat& v)=0;
        virtual ~filter_i(){}
    };
    std::vector<filter_i*> filters_;
    filter_i& filter(int depth){return *filters_[depth];}
    static const int number_of_partitions=8;
    //typed filter
    template<int Depth>
    struct filter_t:public filter_i
    {
        typedef typename depth_traits< Depth >::value_t value_t;
        cv::Mat normalize_max(const cv::Mat& mat)
        {
            bool use_double=(single_channel_type(mat.type()) == CV_64FC1);
            int partition_size=mat.rows/number_of_partitions;
            //max
            cv::Scalar s(0,0,0,0);
            s=tbb::parallel_reduce(tbb::blocked_range<int>(0,mat.rows, partition_size), s, boost::bind(max_,_1,mat,_2), max_scalar);
            //invert
            s=cv::Scalar(1/(s[0]?s[0]:1),1/(s[1]?s[1]:1),1/(s[2]?s[2]:1),1/(s[3]?s[3]:1));
            //scale
            cv::Mat result(mat.rows,mat.cols,use_double?CV_64FC(mat.channels()):CV_32FC(mat.channels()));
            if(use_double)
                tbb::parallel_for(tbb::blocked_range<int>(0,mat.rows, partition_size), boost::bind(scale_<CV_64F>, _1, mat, s, boost::ref(result)));
            else
                tbb::parallel_for(tbb::blocked_range<int>(0,mat.rows, partition_size), boost::bind(scale_<CV_32F>, _1, mat, s, boost::ref(result)));
            return result;
        }
        cv::Mat normalize_sum(const cv::Mat& mat)
        {
            bool use_double=(single_channel_type(mat.type()) == CV_64FC1);
            cv::Mat result(mat.rows,mat.cols, use_double?CV_64FC(mat.channels()):CV_32FC(mat.channels()));
            int partition_size=mat.rows/number_of_partitions;
            if(use_double)
                tbb::parallel_for(tbb::blocked_range<int>(0,mat.rows, partition_size), boost::bind(normalize_sum_<CV_64F>, _1, mat, boost::ref(result)));
            else
                tbb::parallel_for(tbb::blocked_range<int>(0,mat.rows, partition_size), boost::bind(normalize_sum_<CV_32F>, _1, mat, boost::ref(result)));
            return result;
        }
        static cv::Scalar max_(const tbb::blocked_range<int>& r, const cv::Mat& v, cv::Scalar s)
        {
            int chs=v.channels();
            for(int i=r.begin();i<r.end();i++)
            {
                const value_t* row=v.ptr<value_t>(i);
                for(int j=0;j<v.cols;j++)
                    for(int k=0;k<chs;k++)
                        s[k]=std::max(s[k],double(*row++));
            }
            return s;
        }
        // Output_depth: CV_32FC or CV_64FC for float or double precision
        template<int Output_depth>
        static void scale_(const tbb::blocked_range<int>& r, const cv::Mat& v, const cv::Scalar& s, cv::Mat& out)
        {
            typedef typename depth_traits<Output_depth>::value_t output_t;
            int chs=v.channels();
            if(out.cols!=v.cols || out.rows!=v.rows || out.channels()!=chs)
            {
                COMMA_THROW(comma::exception, "scale: in and out Matrix mismatch");
            }
            if(single_channel_type(out.type())!=CV_MAKETYPE(Output_depth,1))
                COMMA_THROW(comma::exception, "scale: invalid output type: "<<type_as_string(out.type()) << " expected :"<<type_as_string(CV_MAKETYPE(Output_depth,1)));
            for(int i=r.begin();i<r.end();i++)
            {
                const value_t* in=v.ptr<value_t>(i);
                output_t* outp=out.ptr<output_t>(i);
                for(int j=0;j<v.cols;j++)
                    for(int k=0;k<chs;k++)
                        *outp++=output_t(*in++)*s[k];
            }
        }
        // Output_depth: CV_32FC or CV_64FC for float or double precision
        template<int Output_depth>
        static void normalize_sum_(const tbb::blocked_range<int>& r, const cv::Mat& v, cv::Mat& out)
        {
            typedef typename depth_traits<Output_depth>::value_t output_t;
            int chs=v.channels();
            if(out.cols!=v.cols || out.rows!=v.rows || out.channels()!=chs)
            {
                COMMA_THROW(comma::exception, "normalize_sum: in and out Matrix mismatch");
            }
            if(single_channel_type(out.type())!=CV_MAKETYPE(Output_depth,1))
                COMMA_THROW(comma::exception, "normalize_sum: invalid output type: "<<type_as_string(out.type()) << " expected :"<<type_as_string(CV_MAKETYPE(Output_depth,1)));
            for(int i=r.begin();i<r.end();i++)
            {
                const value_t* in=v.ptr<value_t>(i);
                output_t* outp=out.ptr<output_t>(i);
                for(int j=0;j<v.cols;j++)
                {
                    output_t sum=0;
                    for(int k=0;k<chs;k++)
                        sum+=output_t(in[k]);
                    for(int k=0;k<chs;k++)
                        *outp++=output_t(*in++)/(sum?sum:1);
                }
            }
        }
    };
    filter_table_t() : filters_(7)
    {
        filters_[CV_8U]=new filter_t<CV_8U>();
        filters_[CV_8S]=new filter_t<CV_8S>();
        filters_[CV_16U]=new filter_t<CV_16U>();
        filters_[CV_16S]=new filter_t<CV_16S>();
        filters_[CV_32S]=new filter_t<CV_32S>();
        filters_[CV_32F]=new filter_t<CV_32F>();
        filters_[CV_64F]=new filter_t<CV_64F>();
    }
    ~filter_table_t() { for(int i=0;i<=CV_64F;i++) { delete filters_[i]; } }
};
static filter_table_t filter_table;

template < typename H >
static typename impl::filters< H >::value_type normalize_max_impl_( typename impl::filters< H >::value_type m )
{
    filter_table_t::filter_i& filter=filter_table.filter(m.second.depth());
    return typename impl::filters< H >::value_type(m.first, filter.normalize_max(m.second));
}

template < typename H >
static typename impl::filters< H >::value_type normalize_sum_impl_( typename impl::filters< H >::value_type m )
{
    filter_table_t::filter_i& filter=filter_table.filter(m.second.depth());
    return typename impl::filters< H >::value_type(m.first, filter.normalize_sum(m.second));
}

static float colour_scale_factor( int const depth )
{
    static auto const factors = std::unordered_map< int, float > { { CV_8S, 0.5 }, { CV_16U, 256.0 }, { CV_16S, 128.0 }, { CV_32S, 8388608.0 }, { CV_32F, 1.0 / 255.0 } };
    auto found = factors.find( depth );
    return ( factors.cend() != found ? found->second : 1.0 );
}

template < typename H >
static typename impl::filters< H >::value_type text_impl_( typename impl::filters< H >::value_type m, const std::string& s, const cv::Point& origin, const cv::Scalar& colour )
{
    cv::putText( m.second, s, origin, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar( colour * colour_scale_factor( m.second.depth() ) ), 1, CV_AA );
    return m;
}

template < typename H >
class undistort_impl_
{
    public:
        typedef typename impl::filters< H >::value_type value_type;
        undistort_impl_( const std::string& filename ) : distortion_coefficients_( 0, 0, 0, 0, 0 )
        {
            try
            {
                const std::vector< std::string >& v = comma::split( filename, ':' );
                snark::camera::pinhole::config_t config;
                switch( v.size() )
                {
                    case 1: comma::read( config, filename ); break;
                    case 2: comma::read( config, v[0], v[1] ); break;
                    default: COMMA_THROW( comma::exception, "undistort: expected <filename>[:<path>]; got: \"" << filename << "\"" );
                }
                if( config.distortion )
                {
                    if( config.distortion->map_filename.empty() )
                    {
                        distortion_coefficients_ = config.distortion->as< cv::Vec< double, 5 > >();
                        camera_matrix_ = config.camera_matrix();
                    }
                    else
                    {
                        filename_ = config.distortion->map_filename;
                    }
                }
            }
            catch( ... )
            {
                filename_ = filename;
            }
        }

        value_type operator()( value_type m )
        {
            value_type n( m.first, cv::Mat( m.second.size(), m.second.type(), cv::Scalar::all(0) ) );
            if( filename_.empty() )
            {
                if( camera_matrix_.empty() ) { n.second = m.second.clone(); }
                else { cv::undistort( m.second, n.second, camera_matrix_, distortion_coefficients_ ); }
            }
            else
            {
                init_map_( m.second.rows, m.second.cols );
                cv::remap( m.second, n.second, x_, y_, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT );
            }
            return n;
        }

    private:
        cv::Mat camera_matrix_;
        cv::Vec< double, 5 > distortion_coefficients_;
        std::string filename_;
        std::vector< char > xbuf_;
        std::vector< char > ybuf_;
        cv::Mat x_;
        cv::Mat y_;
        void init_map_( unsigned int rows, unsigned int cols )
        {
            if( !x_.empty() ) { return; }
            std::ifstream stream( &filename_[0] );
            if( !stream ) { COMMA_THROW( comma::exception, "failed to open undistort map in \"" << filename_ << "\"" ); }
            std::size_t size = rows * cols * 4;
            xbuf_.resize( size );
            stream.read( &xbuf_[0], size );
            if( stream.gcount() < 0 || std::size_t( stream.gcount() ) != size ) { COMMA_THROW( comma::exception, "failed to read \"" << filename_ << "\"" ); }
            ybuf_.resize( size );
            stream.read( &ybuf_[0], size );
            if( stream.gcount() < 0 || std::size_t( stream.gcount() ) != size ) { COMMA_THROW( comma::exception, "failed to read \"" << filename_ << "\"" ); }
            stream.peek(); // quick and dirty
            if( !stream.eof() ) { COMMA_THROW( comma::exception, "expected " << ( size * 2 ) << " bytes in \"" << filename_ << "\", got more" ); }
            x_ = cv::Mat( rows, cols, CV_32FC1, &xbuf_[0] );
            y_ = cv::Mat( rows, cols, CV_32FC1, &ybuf_[0] );
        }
};

template < typename H >
class max_impl_ // experimental, to debug
{
    public:
        typedef typename impl::filters< H >::value_type value_type;
        max_impl_( unsigned int size, bool is_max ) : size_( size ), is_max_( is_max ) {}

        value_type operator()( value_type m )
        {
            if( deque_.size() == size_ ) { deque_.pop_front(); }
            deque_.push_back( value_type() );
            m.second.copyTo( deque_.back().second );
            value_type s( m.first, cv::Mat( m.second.rows, m.second.cols, m.second.type() ) );
            // For min, memset has to set max value
            // TODO: support other image types other than ub
            ::memset( const_cast< unsigned char* >( m.second.datastart ), 0, m.second.rows * m.second.cols * m.second.channels() );
            static unsigned int count = 0;
            for( unsigned int i = 0; i < deque_.size(); ++i )
            {
                unsigned char* p = const_cast< unsigned char* >( deque_[i].second.datastart );
                for( unsigned char* q = const_cast< unsigned char* >( s.second.datastart ); q < s.second.dataend; *q = is_max_ ? std::max( *p, *q ) : std::min( *p, *q ), ++p, ++q );
            }
            ++count;
            return s;
        }

    private:
        unsigned int size_;
        bool is_max_;
        std::deque< value_type > deque_; // use vector?
};

#if CV_MAJOR_VERSION <= 2
template < typename H >
class map_impl_
{
    typedef typename impl::filters< H >::value_type value_type;
    typedef map_input_t::key_type key_type;
    typedef map_input_t::value_type output_value_type;
    public:
        map_impl_( const std::string& map_filter_options, bool permissive ) : permissive_ ( permissive )
        {
            comma::csv::options csv_options = comma::name_value::parser( "filename", '&' , '=' ).get< comma::csv::options >( map_filter_options );
            if( csv_options.fields.empty() ) { csv_options.fields = "value"; }
            if( !csv_options.has_field( "value" ) ) { COMMA_THROW( comma::exception, "map filter: fields option is given but \"value\" field is not found" ); }
            bool no_key_field = !csv_options.has_field( "key" );
            std::ifstream ifs( &csv_options.filename[0] );
            if( !ifs.is_open() ) { COMMA_THROW( comma::exception, "map filter: failed to open \"" << csv_options.filename << "\"" ); }
            comma::csv::input_stream< map_input_t > map_stream( ifs , csv_options );
            for( key_type counter = 0; map_stream.ready() || ( ifs.good() && !ifs.eof() ) ; ++counter )
            {
                const map_input_t* map_input = map_stream.read();
                if( !map_input ) { break; }
                map_[ no_key_field ? counter : map_input->key ] = map_input->value;
            }
        }

        value_type operator()( value_type m ) // todo: support multiple channels
        {
            value_type n( m.first, cv::Mat( m.second.size(), cv::DataType< output_value_type >::type ) );
            try
            {
                switch( m.second.type() ) // quick and dirty; opencv really got their design wrong: type is known in runtime whereas handling types is a compile-time thing
                {
                    case cv::DataType< unsigned char >::type : apply_map_< unsigned char >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< unsigned char, 2 > >::type : apply_map_< cv::Vec< unsigned char, 2 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< unsigned char, 3 > >::type : apply_map_< cv::Vec< unsigned char, 3 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< unsigned char, 4 > >::type : apply_map_< cv::Vec< unsigned char, 4 > >( m.second, n.second ); break;
                    case cv::DataType< comma::uint16 >::type : apply_map_< comma::uint16 >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::uint16, 2 > >::type : apply_map_< cv::Vec< comma::uint16, 2 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::uint16, 3 > >::type : apply_map_< cv::Vec< comma::uint16, 3 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::uint16, 4 > >::type : apply_map_< cv::Vec< comma::uint16, 4 > >( m.second, n.second ); break;
                    case cv::DataType< char >::type : apply_map_< char >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< char, 2 > >::type : apply_map_< cv::Vec< char, 2 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< char, 3 > >::type : apply_map_< cv::Vec< char, 3 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< char, 4 > >::type : apply_map_< cv::Vec< char, 4 > >( m.second, n.second ); break;
                    case cv::DataType< comma::int16 >::type : apply_map_< comma::int16 >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int16, 2 > >::type : apply_map_< cv::Vec< comma::int16, 2 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int16, 3 > >::type : apply_map_< cv::Vec< comma::int16, 3 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int16, 4 > >::type : apply_map_< cv::Vec< comma::int16, 4 > >( m.second, n.second ); break;
                    case cv::DataType< comma::int32 >::type : apply_map_< comma::int32 >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int32, 2 > >::type : apply_map_< cv::Vec< comma::int32, 2 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int32, 3 > >::type : apply_map_< cv::Vec< comma::int32, 3 > >( m.second, n.second ); break;
                    case cv::DataType< cv::Vec< comma::int32, 4 > >::type : apply_map_< cv::Vec< comma::int32, 4 > >( m.second, n.second ); break;
                    default: std::cerr << "map filter: expected integer cv type, got " << m.second.type() << std::endl; return value_type();
                }
            }
            catch ( std::out_of_range ) { return value_type(); }
            return n;
        }

    private:
        typedef std::unordered_map< key_type, output_value_type > map_t_;
        map_t_ map_;
        bool permissive_;
        
        template < typename T, int Size > static T get_channel_( const cv::Vec< T, Size >& pixel, int channel ) { return pixel[channel]; }
        template < typename T > static T get_channel_( const T& pixel, int channel ) { return pixel; }
        template < typename T, int Size > static void set_channel_( cv::Vec< T, Size >& pixel, int channel, T value ) { return pixel[channel] = value; }
        template < typename T > static void set_channel_( T& pixel, int channel, T value ) { pixel = value; }

        template < typename input_value_type >
        void apply_map_( const cv::Mat& input, cv::Mat& output ) // todo: certainly reimplement with tbb::parallel_for
        {
            for( int i = 0; i < input.rows; ++i )
            {
                for( int j = 0; j < input.cols; ++j )
                {
                    const auto& keys = input.at< input_value_type >(i,j);
                    for( int channel = 0; channel < input.channels(); ++channel )
                    {
                        auto key = get_channel_( keys, channel );
                        map_t_::const_iterator it = map_.find( key );
                        if( it == map_.end() )
                        {
                            if( permissive_ ) { std::cerr << "map filter: expected a pixel value from the map, got: pixel at " << i << "," << j << " with value " << key << std::endl; throw std::out_of_range(""); }
                            set_channel_( output.at< output_value_type >(i,j), channel, output_value_type( key ) ); // todo? implement value clipping to 0 or 1? refactor not-found behaviour!
                        }
                        else
                        { 
                            set_channel_( output.at< output_value_type >(i,j), channel, it->second );
                        }
                    }
                }
            }
        }
};
#endif // #if CV_MAJOR_VERSION <= 2

template < typename H >
static typename impl::filters< H >::value_type magnitude_impl_( typename impl::filters< H >::value_type m )
{
    if( m.second.channels() != 2 ) { std::cerr << "cv filters: magnitude: expected 2 channels, got " << m.second.channels() << std::endl; return typename impl::filters< H >::value_type(); }
    boost::array< cv::Mat, 2 > planes;
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::split( m.second, &planes[0] );
    cv::magnitude( planes[0], planes[1], n.second );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type convert( typename impl::filters< H >::value_type m, bool scale, bool complex, bool magnitude, bool log_scale, bool normalize )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::dft( m.second, n.second, ( scale ? cv::DFT_SCALE : cv::DFT_INVERSE ) | ( complex ? cv::DFT_COMPLEX_OUTPUT : cv::DFT_REAL_OUTPUT ) );
    if( !magnitude ) { return n; }
    boost::array< cv::Mat, 2 > planes = {{ cv::Mat::zeros( m.second.size(), m.second.type() ), cv::Mat::zeros( m.second.size(), m.second.type() ) }};
    cv::split( n.second, &planes[0] );
    cv::magnitude( planes[0], planes[1], n.second ); // make separate filters: magnitude, log, scale, normalize?
    if( log_scale )
    {
        n.second += cv::Scalar::all( 1 );
        cv::log( n.second, n.second ); // todo: optional
    }
    if( normalize ) { cv::normalize( n.second, n.second, 0, 1, CV_MINMAX ); }
    return n;
}

template < typename H, typename T, int Type >
static typename impl::filters< H >::value_type convert( typename impl::filters< H >::value_type m, bool magnitude, bool log_scale, bool normalize )
{
    cv::Mat padded;
    int padded_rows = cv::getOptimalDFTSize( m.second.rows );
    int padded_cols = cv::getOptimalDFTSize( m.second.cols );
    cv::copyMakeBorder( m.second, padded, 0, padded_rows - m.second.rows, 0, padded_cols - m.second.cols, cv::BORDER_CONSTANT, cv::Scalar::all( 0 ) );
    boost::array< cv::Mat, 2 > planes = {{ cv::Mat_< T >( padded ), cv::Mat::zeros( padded.size(), Type ) }};
    typename impl::filters< H >::value_type p;
    p.first = m.first;
    cv::merge( &planes[0], 2, p.second );
    cv::dft( p.second, p.second );
    if( !magnitude ) { return p; }
    cv::split( p.second, &planes[0] );
    cv::magnitude( planes[0], planes[1], planes[0] );
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    n.second = planes[0];
    if( log_scale )
    {
        n.second += cv::Scalar::all( 1 );
        cv::log( n.second, n.second ); // todo: optional
    }
    n.second = n.second( cv::Rect( 0, 0, n.second.cols & -2, n.second.rows & -2 ) );

    int cx = n.second.cols / 2 ;
    int cy = n.second.rows / 2 ;

    cv::Mat q0(n.second, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(n.second, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(n.second, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(n.second, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;
    q0.copyTo( tmp ); // swap top-left with bottom-right
    q3.copyTo( q0 );
    tmp.copyTo( q3 );
    q1.copyTo( tmp ); // swap top-right with bottom-left
    q2.copyTo( q1 );
    tmp.copyTo( q2 );

    if( normalize ) { cv::normalize( n.second, n.second, 0, 1, CV_MINMAX ); }
    return n;
}

template < typename H >
typename impl::filters< H >::value_type fft_impl_( typename impl::filters< H >::value_type m, bool direct, bool complex, bool magnitude, bool log_scale, bool normalize )
{
    switch( m.second.type() )
    {
        case CV_32FC1:
            //return convert< float, CV_32FC1 >( m, magnitude, log_scale, normalize );
        case CV_32FC2:
            return convert< H >( m, direct, complex, magnitude, log_scale, normalize );
        case CV_32FC3:
        case CV_32FC4:
            std::cerr << "fft: multichannel image support: todo, got: " << type_as_string( m.second.type() ) << std::endl;
            return typename impl::filters< H >::value_type();
        case CV_64FC1:
            //return convert< double, CV_64FC1 >( m, magnitude, log_scale, normalize, normalize );
        case CV_64FC2:
            return convert< H >( m, direct, complex, magnitude, log_scale, normalize );
        case CV_64FC3:
        case CV_64FC4:
            std::cerr << "fft: multichannel image support: todo, got: " << type_as_string( m.second.type() ) << std::endl;
            return typename impl::filters< H >::value_type();
        default:
            std::cerr << "fft: expected a floating-point image type, got: " << type_as_string( m.second.type() ) << std::endl;
            return typename impl::filters< H >::value_type();
    }
}

template< int DepthIn, int DepthOut >
static void ratio( const tbb::blocked_range< std::size_t >& r, const cv::Mat& m, const std::vector< ratios::coefficients > & coefficients, cv::Mat& result )
{
    typedef typename depth_traits< DepthIn >::value_t value_in_t;
    typedef typename depth_traits< DepthOut >::value_t value_out_t;
    const unsigned int ichannels = m.channels();
    const unsigned int ochannels = result.channels();
    if ( ochannels != coefficients.size() ) { COMMA_THROW( comma::exception, "the number of output channels " << ochannels << " differs from the number of ratios " << coefficients.size() ); }
    const unsigned int icols = m.cols * ichannels;
    static const value_out_t highest = std::numeric_limits< value_out_t >::max();
    static const value_out_t lowest = std::numeric_limits< value_out_t >::is_integer ? std::numeric_limits< value_out_t >::min() : -highest;
    for( unsigned int i = r.begin(); i < r.end(); ++i )
    {
        const value_in_t* in = m.ptr< value_in_t >(i);
        const value_in_t* inc = in;
        value_out_t* out = result.ptr< value_out_t >(i);
        for( unsigned int j = 0; j < icols; j += ichannels )
        {
            for ( const auto & coeffs : coefficients )
            {
                in = inc;
                double n = coeffs[0].first;
                double d = coeffs[0].second;
                for( unsigned int k = 0; k < ichannels; ++k ) {
                    n += *in * coeffs[k + 1].first;
                    d += *in++ * coeffs[k + 1].second;
                }
                double value = ( d == 0 ? ( n == 0 ? 0 : highest ) : n / d );
                *out++ = value > highest ? highest : value < lowest ? lowest : value;
            }
            inc = in;
        }
    }
}

template< typename H, int DepthIn, int DepthOut >
static typename impl::filters< H >::value_type per_element_ratio( const typename impl::filters< H >::value_type m, const std::vector< ratios::coefficients > & coefficients )
{
    int otype = CV_MAKETYPE( DepthOut, coefficients.size() );
    cv::Mat result( m.second.size(), otype );
    tbb::parallel_for( tbb::blocked_range< std::size_t >( 0, m.second.rows ), boost::bind( &ratio< DepthIn, DepthOut >, _1, m.second, coefficients, boost::ref( result ) ) );
    return typename impl::filters< H >::value_type( m.first, result );
}

template< typename H, int DepthIn >
static typename impl::filters< H >::value_type per_element_ratio_selector( const typename impl::filters< H >::value_type m, const std::vector< ratios::coefficients > & coefficients, int otype, const std::string & opname )
{
    switch( otype )
    {
        case CV_8U : return per_element_ratio< H, DepthIn, CV_8U  >( m, coefficients );
        case CV_8S : return per_element_ratio< H, DepthIn, CV_8S  >( m, coefficients );
        case CV_16U: return per_element_ratio< H, DepthIn, CV_16U >( m, coefficients );
        case CV_16S: return per_element_ratio< H, DepthIn, CV_16S >( m, coefficients );
        case CV_32S: return per_element_ratio< H, DepthIn, CV_32S >( m, coefficients );
        case CV_32F: return per_element_ratio< H, DepthIn, CV_32F >( m, coefficients );
        case CV_64F: return per_element_ratio< H, DepthIn, CV_64F >( m, coefficients );
    }
    COMMA_THROW( comma::exception, opname << ": unrecognised output image type " << otype );
}

template < typename H >
static typename impl::filters< H >::value_type ratio_impl_( const typename impl::filters< H >::value_type m, const std::vector< ratios::coefficients > & coefficients, const std::string & opname )
{
    // the coefficients are always constant,r,g,b,a (some of the values can be zero); it is ok to have fewer channels than coefficients as long as all the unused coefficients are zero
    for ( size_t n = static_cast< size_t >( m.second.channels() ) + 1 ; n < coefficients.size(); ++n ) {
        for ( size_t l = 0; l < coefficients[n].size(); ++l ) {
            if ( coefficients[n][l].first != 0.0 || coefficients[n][l].second != 0.0 ) {
                const std::string & what = coefficients[n][l].first != 0.0 ? "numerator" : "denominator";
                COMMA_THROW( comma::exception, opname << ": have " << m.second.channels() << " channel(s) only, requested non-zero " << what << " coefficient for channel " << n - 1 );
            }
        }
    }
    int otype = single_channel_type( m.second.type() );
    if ( opname != "shuffle" && otype != CV_64FC1 ) { otype = CV_32FC1; }
    switch( m.second.depth() )
    {
        case CV_8U : return per_element_ratio_selector< H, CV_8U  >( m, coefficients, otype, opname );
        case CV_8S : return per_element_ratio_selector< H, CV_8S  >( m, coefficients, otype, opname );
        case CV_16U: return per_element_ratio_selector< H, CV_16U >( m, coefficients, otype, opname );
        case CV_16S: return per_element_ratio_selector< H, CV_16S >( m, coefficients, otype, opname );
        case CV_32S: return per_element_ratio_selector< H, CV_32S >( m, coefficients, otype, opname );
        case CV_32F: return per_element_ratio_selector< H, CV_32F >( m, coefficients, otype, opname );
        case CV_64F: return per_element_ratio_selector< H, CV_64F >( m, coefficients, otype, opname );
    }
    COMMA_THROW( comma::exception, opname << ": unrecognised input image type " << m.second.type() );
}

static double max_value(int depth)
{
    switch(depth)
    {
        case CV_8U : return depth_traits< CV_8U  >::max_value();
        case CV_8S : return depth_traits< CV_8S  >::max_value();
        case CV_16U: return depth_traits< CV_16U >::max_value();
        case CV_16S: return depth_traits< CV_16S >::max_value();
        case CV_32S: return depth_traits< CV_32S >::max_value();
        case CV_32F: return depth_traits< CV_32F >::max_value();
        case CV_64F: return depth_traits< CV_64F >::max_value();
        default: { COMMA_THROW(comma::exception, "invalid depth: "<<depth ); }
    }
}

static cv::Mat convert_and_scale(const cv::Mat& m, int depth)
{
    cv::Mat result;
    double scale=max_value(depth)/max_value(m.depth());
    m.convertTo(result, CV_MAKETYPE(depth, m.channels()), scale);
    return result;
}

template < typename H >
struct overlay_impl_
{
    int x;
    int y;
    cv::Mat overlay;
    int alpha;
    typedef typename impl::filters< H >::value_type value_type;

    overlay_impl_(const std::string& image_file, int a, int b) : x(a), y(b), alpha(0)
    {
//         comma::verbose<<"overlay_impl_: image_file "<<image_file<<std::endl;
//         comma::verbose<<"overlay_impl_: x,y "<<x<<","<<y<<std::endl;
        overlay=cv::imread(image_file,-1);
        if(overlay.data==NULL) { COMMA_THROW( comma::exception, "failed to load image file: "<<image_file); }
    }
    value_type operator()( value_type m )
    {
        cv::Mat& mat=m.second;
//         comma::verbose<<"mat rows,cols,type;channels,depth "<<mat.rows<<","<<mat.cols<<","<<type_as_string(mat.type())<<";"<<mat.channels()<<","<<mat.depth()<<std::endl;
//         comma::verbose<<"overlay rows,cols,type;channels,depth "<<overlay.rows<<","<<overlay.cols<<","<<type_as_string(overlay.type())<<";"<<overlay.channels()<<","<<overlay.depth()<<std::endl;
        if(mat.channels()!=overlay.channels()-1) { COMMA_THROW(comma::exception, "mat's channels ("<<mat.channels()<<") should be one less than overlay's channel: "<<overlay.channels()); }
        if(mat.depth() != overlay.depth())
        {
            comma::verbose<<"converting overlay from depth "<<overlay.depth()<<" to "<<mat.depth()<<std::endl;
            overlay=convert_and_scale(overlay, mat.depth());
        }
        cv::Mat result(mat.rows, mat.cols, CV_MAKETYPE(overlay.depth(), overlay.channels()-1));
        switch(mat.depth())
        {
            case CV_8U: process<CV_8U>(mat, overlay, x, y, result); break;
            case CV_8S: process<CV_8S>(mat, overlay, x, y, result); break;
            case CV_16U: process<CV_16U>(mat, overlay, x, y, result); break;
            case CV_16S: process<CV_16S>(mat, overlay, x, y, result); break;
            case CV_32S: process<CV_32S>(mat, overlay, x, y, result); break;
            case CV_32F: process<CV_32F>(mat, overlay, x, y, result); break;
            case CV_64F: process<CV_64F>(mat, overlay, x, y, result); break;
            default: { COMMA_THROW( comma::exception, "invalid depth: " << mat.depth()); }
        }
        return value_type(m.first, result);
    }
    template<int Depth> static void process(const cv::Mat& mat, const cv::Mat& overlay, int x, int y, cv::Mat& result)
    {
        double max_a=depth_traits< Depth >::max_value();
//         comma::verbose<<"overrlay process<Depth="<<Depth<<">: x,y "<<x<<","<<y<<" max_a "<<max_a<<std::endl;
        if(mat.depth() != Depth) { COMMA_THROW( comma::exception, "mat depth ("<<mat.depth() <<")mismatch, expected: "<< Depth); }
        if(overlay.depth() != Depth) { COMMA_THROW( comma::exception, "overlay depth ("<<overlay.depth() <<")mismatch, expected: "<< Depth); }
        typedef typename depth_traits< Depth >::value_t value_t;
        int alpha;
        int ch=overlay.channels();
        switch(ch)
        {
            case 2: alpha=1; break;
            case 4: alpha=3; break;
            default: { COMMA_THROW( comma::exception, "overlay needs to have alpha channel; expected number of channels 2 or 4, got "<<ch ); }
        }
//         comma::verbose<<"alpha: "<<alpha<<" ;result rows,cols,channels "<< result.rows<<","<<result.cols<<","<<result.channels()<<std::endl;
        //assert result.rows==mat.rows && result.cols==mat.cols
        int channels=result.channels();
        int y2=y+overlay.rows;
        int x2=x+overlay.cols;
        for(int j=0;j<result.rows;j++)
        {
            const value_t* mat_ptr=mat.ptr<value_t>(j);
            const value_t* overlay_ptr=NULL;
            if(j>=y && j<y2) { overlay_ptr=overlay.ptr<value_t>(j-y); }
            value_t* data=result.ptr<value_t>(j);
            int index=0;
            for(int i=0;i<result.cols;i++)
            {
                if(overlay_ptr!=NULL && i>=x && i<x2)
                {
                    double a=static_cast<double>(overlay_ptr[alpha]) / max_a;
                    for(int c=0;c<channels;c++)
                    {
                        double m=static_cast<double>(*mat_ptr++);
                        double o=static_cast<double>(*overlay_ptr++);
                        data[index++]= o * a + m * (1-a);
                    }
                    //skip alpha
                    overlay_ptr++;
                }
                else
                {
                    //just copy
                    for(int c=0;c<channels;c++)
                        data[index++]=*mat_ptr++;
                }
            }
        }
    }
};

template < unsigned int Depth > struct gamma_traits {};

template <> struct gamma_traits< CV_8U >
{
    typedef unsigned char type;
    enum { is_signed = false };
    static const type min = 0;
    static const type max = 255;
};

template <> struct gamma_traits< CV_8S >
{
    typedef char type;
    enum { is_signed = true };
    static const type min = -128;
    static const type max = 127;
};

template < unsigned int Depth > static cv::Mat lut_matrix_gamma_( double gamma )
{
    double num_states = gamma_traits< Depth >::max - gamma_traits< Depth >::min; //std::numeric_limits< gamma_traits< Depth >::type >::max();
    cv::Mat lut_matrix( 1, num_states + 1, Depth );
    uchar * ptr = lut_matrix.ptr();
    double scale = std::abs( gamma_traits< Depth >::max );
    for( unsigned int i = 0, j = gamma_traits< Depth >::min; i <= num_states; ++i, ++j ) { ptr[i] = std::pow( j / scale, 1.0 / gamma ) * scale; }
    return lut_matrix;
}

template < typename H, unsigned int Depth >
static typename impl::filters< H >::value_type gamma_( const typename impl::filters< H >::value_type m, const double gamma )
{
    static cv::Mat lut_matrix = lut_matrix_gamma_< Depth >( gamma );
    cv::LUT( m.second, lut_matrix, m.second );
    return m;
}

// todo! refactor as class
// todo! support multichannel images
template < typename H >
static typename impl::filters< H >::value_type gamma_impl_(const typename impl::filters< H >::value_type m, const double gamma )
{
    switch( m.second.depth() )
    {
        case CV_8U: { return gamma_< H, CV_8U >( m, gamma ); break; }
        default: COMMA_THROW( comma::exception, "gamma: expected single-channel CV_8U, got: " << m.second.depth() );;
    }
}

template < typename H >
static typename impl::filters< H >::value_type pow_impl_(const typename impl::filters< H >::value_type m, const double value )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::pow( m.second, value, n.second );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type inrange_impl_( const typename impl::filters< H >::value_type m, const cv::Scalar& lower, const cv::Scalar& upper )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    n.second = cv::Mat( m.second.rows, m.second.cols, single_channel_type( m.second.type() ) );
    cv::inRange( m.second, lower, upper, n.second );
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type remove_mean_impl_(const typename impl::filters< H >::value_type m, const cv::Size kernel_size, const double ratio )
{
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    cv::GaussianBlur(m.second, n.second, kernel_size, 0, 0);
    n.second = m.second - ratio * n.second;
    return n;
}

template < typename H >
static typename impl::filters< H >::value_type clone_channels_impl_( typename impl::filters< H >::value_type m, unsigned int nchannels )
{
    if( m.second.channels() > 1 ) { COMMA_THROW( comma::exception, "clone-channels: expected one channel, got " << m.second.channels() ); }
    typename impl::filters< H >::value_type n;
    n.first = m.first;
    std::vector< cv::Mat > channels( nchannels );
    for( std::size_t i = 0; i < nchannels; ++i ) { m.second.copyTo( channels[i] ); }
    cv::merge( channels, n.second );
    return n;
}

template < typename O, typename H >
struct make_filter {
    typedef typename impl::filters< H >::value_type value_type_t;
    typedef operation< O, H > filter_type;
    typedef typename filter_type::input_type input_type;
    typedef typename filter_type::output_type output_type;
    typedef boost::function< input_type( input_type ) > functor_type;
    typedef typename impl::filters< H >::get_timestamp_functor get_timestamp_functor;
    
static std::pair< functor_type, bool > make_filter_functor( const std::vector< std::string >& e, const get_timestamp_functor& get_timestamp, unsigned int default_delay )
{
    if( e[0] == "accumulate" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "accumulate: please specify at least <size>" ); }
        const auto& w = comma::split( e[1], ',' );
        unsigned int how_many = boost::lexical_cast< unsigned int >( w[0] );
        if( how_many == 0 ) { COMMA_THROW( comma::exception, "expected positive number of images to accumulate in accumulate filter, got " << how_many ); }
        typename accumulate_impl_< H >::how_values how = accumulate_impl_< H >::sliding;
        bool reverse = false;
        if( w.size() > 1 )
        {
            if( w[1] == "sliding" || w[1].empty() ) { how = accumulate_impl_< H >::sliding; }
            else if( w[1] == "fixed" ) { how = accumulate_impl_< H >::fixed; }
            else { COMMA_THROW( comma::exception, "accumulate: expected <how>, got\"" << w[1] << "\"" ); }
        }
        if( w.size() > 2 )
        {
            if( w[2] == "reverse" || w[1].empty() ) { reverse = true; }
            else { COMMA_THROW( comma::exception, "accumulate: expected reverse, got\"" << w[1] << "\"" ); }
        }
        return std::make_pair( accumulate_impl_< H >( how_many, how, reverse ), false );
    }
    if( e[0] == "accumulated" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "accumulated: please specify operation" ); }
        const auto& s = comma::split(e[1], ',');
        try
        {
            if( s.front() == "average" ) { return std::make_pair(  boost::bind< value_type_t >( accumulated::average< H >(), _1 ), false ); }
            if( s.front() == "moving-average" )
            { 
                if( s.size() < 2 ){ COMMA_THROW(comma::exception, "accumulated: error please provide window size for " << s.front() ); }
                return std::make_pair(  boost::bind< value_type_t >( accumulated::moving_average< H >( boost::lexical_cast< comma::uint32 >(s[1]) ), _1 ), false ); 
            }
            if( s.front() == "ema" )
            { 
                if( s.size() < 2 ){ COMMA_THROW(comma::exception, "accumulated: error please provide alpha value for " << s.front() ); }
                return std::make_pair( boost::bind< value_type_t >( accumulated::ema< H >( boost::lexical_cast< float >(s[1]), s.size() < 3 ? 1 : boost::lexical_cast< comma::uint32 >(s[2]) ), _1 ), false ); 
            }
            if( s.front() == "min" ) { return std::make_pair( boost::bind< value_type_t >( accumulated::min< H >(), _1 ), false ); }
            if( s.front() == "max" ) { return std::make_pair( boost::bind< value_type_t >( accumulated::max< H >(), _1 ), false ); }
            COMMA_THROW(comma::exception, "accumulated: unrecognised operation: " << s.front());
        }
        catch( boost::bad_lexical_cast& bc )
        {
            COMMA_THROW(comma::exception, "accumulated=" << s.front() << ": failed to cast filter parameter(s): " << bc.what());
        }
    }
    if( e[0] == "balance-white" ) { return std::make_pair( impl::balance_white< H >(), false ); }
    if( e[0] == "canny" )
    {
        if( e.size() == 1 ) { COMMA_THROW( comma::exception, "canny: please specify <threshold1>,<threshold2>[,<kernel_size>]" ); }
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        if( s.size() < 2 ) { COMMA_THROW( comma::exception, "canny: expected canny=<threshold1>,<threshold2>[,<kernel_size>]; got: \"" << comma::join( e, '=' ) << "\"" ); }
        double threshold1 = boost::lexical_cast< double >( s[0] );
        double threshold2 = boost::lexical_cast< double >( s[1] );
        int kernel_size = s.size() > 2 ? boost::lexical_cast< int >( s[2] ) : 3;
        return std::make_pair( boost::bind< value_type_t >( canny_impl_< H >(), _1, threshold1, threshold2, kernel_size ), true );
    }
    if( e[0] == "clahe" )
    {
        std::vector< std::string > s = comma::split( e[1], ',' );
        if( s.size() != 3 ) { COMMA_THROW( comma::exception, "clahe: please specify <clip_limit>,<tile_size_x>,<tile_size_y>" ); }
        try
        {
            float clip_limit = boost::lexical_cast< float >( s[0] );
            cv::Size tile_size( boost::lexical_cast< unsigned int >( s[1] ), boost::lexical_cast< unsigned int >( s[ s.size() == 2 ? 1 : 2 ] ) );
            return std::make_pair( clahe_impl_< H >( clip_limit, tile_size ), false );
        }
        catch( boost::bad_lexical_cast& bc )
        {
            COMMA_THROW( comma::exception, "clahe=" << e[1] << ": failed to cast parameter: " << bc.what());
        }
    }
    if( e[0] == "convert-color" || e[0] == "convert_color" )
    {
        if( e.size() == 1 ) { COMMA_THROW( comma::exception, "convert-color: please specify conversion" ); }
        return std::make_pair(boost::bind< value_type_t >( cvt_color_impl_< H >(), _1, impl::cvt_color_type_from_string( e[1] ) ), true );
    }
    if( e[0] == "count" ) { return std::make_pair(count_impl_< H >(), false ); }
    if( e[0] == "crop" )
    {
        unsigned int x = 0;
        unsigned int y = 0;
        unsigned int w, h;
        std::vector< std::string > s = comma::split( e[1], ',' );
        switch( s.size() )
        {
            case 2:
                w = boost::lexical_cast< unsigned int >( s[0] );
                h = boost::lexical_cast< unsigned int >( s[1] );
                break;
            case 4:
                x = boost::lexical_cast< unsigned int >( s[0] );
                y = boost::lexical_cast< unsigned int >( s[1] );
                w = boost::lexical_cast< unsigned int >( s[2] );
                h = boost::lexical_cast< unsigned int >( s[3] );
                break;
            default:
                COMMA_THROW( comma::exception, "expected crop=[x,y,]width,height, got \"" << comma::join( e, '=' ) << "\"" );
        }
        return std::make_pair(boost::bind< value_type_t >( crop_impl_< H >, _1, x, y, w, h ), true );
    }
    if( e[0] == "crop-cols" || e[0] == "crop-rows" )
    {
        const std::string & op_name = e[0];
        const std::string & what = ( e[0] == "crop-cols" ? "column" : "row" );
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, op_name << ": specify at least one " << what << " to extract, e.g. " << op_name << "=1,10" ); }
        std::vector< std::string > inputs = comma::split( e[1], ',' );
        if ( inputs.size() % 2 ) { COMMA_THROW( comma::exception, op_name << ": must provide an even number of integers in <" << what << ">,<size> pairs" ); }
        std::vector< stripe_t > stripes;
        for ( size_t s = 0; s < inputs.size(); ++s, ++s )
        {
            unsigned int x = boost::lexical_cast< unsigned int >( inputs[s] );
            unsigned int w = boost::lexical_cast< unsigned int >( inputs[s+1] );
            stripes.push_back( std::make_pair( x, w ) );
        }
        if ( e[0] == "crop-cols" ) { return std::make_pair(boost::bind< value_type_t >( crop_cols_impl_< H >(), _1, stripes ), true ); }
        else { return std::make_pair(boost::bind< value_type_t >( crop_rows_impl_< H >(), _1, stripes ), true ); }
    }
    if( e[0] == "bands-to-cols" || e[0] == "bands-to-rows" )
    {
        // rhs looks like "12,23,50,30,100,1,method:average,output-depth:d"
        // the ','-separated entries shall be either:
        // - integers always coming in pairs, or
        // - colon-separated words with a known keyword on the left and one of the known enumeration names on the right
        const bool bands_to_cols = e[0] == "bands-to-cols";
        const std::string & op_name = bands_to_cols ? "bands-to-cols" : "bands-to-rows";
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, op_name << ": specify at least one band to extract, e.g. " << op_name << "=1,10" ); }
        std::vector< std::string > stripes = comma::split( e[1], ',' );
        std::vector< stripe_t > bands;
        // iterate over pair of integers until all taken; then iterate over "name:value" pairs
        size_t s = 0;
        while ( s < stripes.size() )
        {
            if ( stripes[s].empty() ) { COMMA_THROW( comma::exception, op_name << ": empty comma-separated field in '" << e[1] << "'" ); }
            unsigned int y;
            try {
                y = boost::lexical_cast< int >( stripes[s] );
            } catch ( boost::bad_lexical_cast & ) {
                break; // possibly a keyword, not an error
            }
            if ( ! (++s < stripes.size() ) ) { COMMA_THROW( comma::exception, op_name << ": expected <int, int> pairs, got a single int in '" << e[1] << "'" ); }
            unsigned int h;
            try {
                h = boost::lexical_cast< int >( stripes[s] );
            } catch ( boost::bad_lexical_cast & ) {
                COMMA_THROW( comma::exception, op_name << ": expected <position>,<size> integer pairs, got " << e[1] );
            }
            bands.push_back( std::make_pair( y, h ) );
            ++s;
        }
        // the rest of the string shall be comma-separated name:value pairs
        int cv_reduce_method = bands_method_default;
        int cv_reduce_dtype = -1;
        while ( s < stripes.size() )
        {
            if ( stripes[s].empty() ) { COMMA_THROW( comma::exception, op_name << ": empty comma-separated field in '" << e[1] << "'" ); }
            std::vector< std::string > setting = comma::split( stripes[s], ':' );
            if ( setting.size() != 2 ) { COMMA_THROW( comma::exception, op_name << ": expected keyword:value; got " << setting.size() << " parameter '" << stripes[s] << "'" ); }
            if ( setting[0] == "method" )
            {
                static std::map< std::string, int > methods = boost::assign::map_list_of ( "average", CV_REDUCE_AVG ) ( "sum", CV_REDUCE_SUM ) ( "min", CV_REDUCE_MIN ) ( "max", CV_REDUCE_MAX );
                std::map< std::string, int >::const_iterator found = methods.find( setting[1] );
                if ( found == methods.end() ) { COMMA_THROW( comma::exception, op_name << ": the method is not one of [average,sum,min,max]" ); }
                cv_reduce_method = found->second;
            }
            else if ( setting[0] == "output-depth" )
            {
                // the permitted list is very restrictive and explicit
                static std::map< std::string, int > depths = boost::assign::map_list_of ( "CV_32S", CV_32S ) ( "i", CV_32S ) ( "CV_32F", CV_32F ) ( "f", CV_32F ) ( "CV_64F", CV_64F ) ( "d", CV_64F );
                std::map< std::string, int >::const_iterator found = depths.find( setting[1] );
                if ( found == depths.end() ) { COMMA_THROW( comma::exception, op_name << ": the output-depth '" << setting[1] << "' is not one of [i,f,d] or [CV_32S,CV_32F,CV_64F]" ); }
                cv_reduce_dtype = found->second;
            }
            else
            {
                COMMA_THROW( comma::exception, op_name << ": the keyword '" << setting[0] << "' is not one of [method,output-depth]" );
            }
            ++s;
        }
        if ( bands.empty() ) { COMMA_THROW( comma::exception, op_name << ": specify at least one band" ); }
        return std::make_pair( boost::bind< value_type_t >( bands_to_cols_impl_< H >(), _1, bands_to_cols, bands, cv_reduce_method, cv_reduce_dtype ), true );
    }
    if( e[0] == "crop-tile" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "crop-tile: specify at least tile count along x and y" ); }
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        bool vertical = true;
        boost::optional< unsigned int > count_x = boost::make_optional< unsigned int >( false, 0 );
        boost::optional< unsigned int > count_y;
        boost::optional< unsigned int > x;
        boost::optional< unsigned int > y;
        std::vector< tile_t > tiles;
        for( unsigned int i = 0; i < s.size(); ++i )
        {
            if( s[i] == "horizontal" ) { vertical = false; continue; }
            if( !count_x ) { count_x = boost::lexical_cast< unsigned int >( s[i] ); continue; }
            if( !count_y ) { count_y = boost::lexical_cast< unsigned int >( s[i] ); continue; }
            if( !x ) { x = boost::lexical_cast< unsigned int >( s[i] ); continue; }
            y = boost::lexical_cast< unsigned int >( s[i] );
            tiles.push_back( tile_t( *x, *y ) );
            x.reset();
            y.reset();
        }
        if( !count_x ) { COMMA_THROW( comma::exception, "crop-tile: expected tile count along x; got: \"" << e[1] << "\"" ); }
        if( !count_y ) { COMMA_THROW( comma::exception, "crop-tile: expected tile count along x and y; got: \"" << e[1] << "\"" ); }
        if( x && !y ) { COMMA_THROW( comma::exception, "crop-tile: expected tile count along x and y followed by tile positions; got: \"" << e[1] << "\"" ); }
        for( auto tile: tiles )
        {
            if( tile.first >= *count_x ) { COMMA_THROW( comma::exception, "crop-tile: expected tile column index less than " << *count_x << "; got: " << tile.first ); }
            if( tile.second >= *count_y ) { COMMA_THROW( comma::exception, "crop-tile: expected tile row index less than " << *count_y << "; got: " << tile.second ); }
        }
        return std::make_pair( boost::bind< value_type_t >( crop_tile_impl_< H >(), _1, *count_x, *count_y, tiles, vertical ), true );
    }
    if( e[0] == "tile" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "tile: specify tile width and height" ); }
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        bool vertical = true;
        boost::optional< unsigned int > width = boost::make_optional< unsigned int >( false, 0 );
        boost::optional< unsigned int > height = boost::make_optional< unsigned int >( false, 0 );
        for( unsigned int i = 0; i < s.size(); ++i )
        {
            if( s[i] == "horizontal" ) { vertical = false; continue; }
            if( !width ) { width = boost::lexical_cast< unsigned int >( s[i] ); continue; }
            height = boost::lexical_cast< unsigned int >( s[i] );
        }
        if( !width ) { COMMA_THROW( comma::exception, "untile: expected tile count along x; got: \"" << e[1] << "\"" ); }
        if( !height ) { COMMA_THROW( comma::exception, "untile: expected tile count along x and y; got: \"" << e[1] << "\"" ); }
        return std::make_pair( boost::bind< value_type_t >( tile_impl_< H >(), _1, *width, *height, vertical ), true );
    }
    if( e[0] == "untile" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "untile: specify tile count along x and y" ); }
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        bool vertical = true;
        boost::optional< unsigned int > count_x = boost::make_optional< unsigned int >( false, 0 );
        boost::optional< unsigned int > count_y = boost::make_optional< unsigned int >( false, 0 );
        for( unsigned int i = 0; i < s.size(); ++i )
        {
            if( s[i] == "horizontal" ) { vertical = false; continue; }
            if( !count_x ) { count_x = boost::lexical_cast< unsigned int >( s[i] ); continue; }
            count_y = boost::lexical_cast< unsigned int >( s[i] );
        }
        if( !count_x ) { COMMA_THROW( comma::exception, "untile: expected tile count along x; got: \"" << e[1] << "\"" ); }
        if( !count_y ) { COMMA_THROW( comma::exception, "untile: expected tile count along x and y; got: \"" << e[1] << "\"" ); }
        return std::make_pair( boost::bind< value_type_t >( untile_impl_< H >(), _1, *count_x, *count_y, vertical ), true );
    }
    if( e[0] == "cols-to-channels" || e[0] == "rows-to-channels" )
    {
        // rhs looks like "cols-to-channels=1,4,5[,pad:value,repeat:step]"
        // the ','-separated entries shall be either:
        // - integers, or
        // - colon-separated words with a known keyword on the left and one of the known enumeration names on the right
        const bool cols_to_channels = e[0] == "cols-to-channels";
        const std::string & op_name = cols_to_channels ? "cols-to-channels" : "rows-to-channels";
        const std::string & op_what = cols_to_channels ? "column" : "row";
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, op_name << ": specify at least one column or column list to extract, e.g. " << op_name << "=1,10" ); }
        std::vector< std::string > inputs = comma::split( e[1], ',' );
        std::vector< unsigned int > values;
        double padding = 0.0;
        unsigned int repeat = 0;
        size_t s = 0;
        // first, iterate over column number, then, over options
        while ( s < inputs.size() )
        {
            try {
                values.push_back( boost::lexical_cast< unsigned int >( inputs[s] ) );
            } catch ( boost::bad_lexical_cast & ) {
                break; // maybe an option
            }
            ++s;
        }
        while ( s < inputs.size() )
        {
            std::vector< std::string > setting = comma::split( inputs[s], ':' );
            if ( setting.size() != 2 ) { COMMA_THROW( comma::exception, op_name << ": expected keyword:value; got " << setting.size() << " parameter '" << inputs[s] << "'" ); }
            if ( setting[0] == "pad" )
            {
                padding = boost::lexical_cast< double >( setting[1] );
            }
            else if ( setting[0] == "repeat" )
            {
                repeat = boost::lexical_cast< unsigned int >( setting[1] );
            }
            else
            {
                COMMA_THROW( comma::exception, op_name << ": the keyword '" << setting[0] << "' is not one of [pad,repeat]" );
            }
            ++s;
        }
        if ( values.empty() ) { COMMA_THROW( comma::exception, op_name << ": specify at least one " << op_what << " to store as channel" ); }
        if ( values.size() > 4 ) { COMMA_THROW( comma::exception, op_name << ": can have at most 4 output channels" ); }
        return std::make_pair(  boost::bind< value_type_t >( cols_to_channels_impl_< H >(), _1, cols_to_channels, values, padding, repeat ), true );
    }
    if( e[0] == "channels-to-cols" || e[0] == "channels-to-rows" )
    {
        const bool channels_to_cols = e[0] == "channels-to-cols";
        return std::make_pair( boost::bind< value_type_t >( channels_to_cols_impl_ < H >(), _1, channels_to_cols ), true );
    }
    if( e[0] == "cross" ) // todo: quick and dirty, implement using traits
    {
        boost::array< int, 9 > p = {{ 0, 0, 0, 0, 0, 1, 8, 0 }};
        const std::vector< std::string > v = comma::split( e[1], ',' );
        for( unsigned int i = 0; i < v.size(); ++i ) { if( !v[i].empty() ) { p[i] = boost::lexical_cast< int >( v[i] ); } }
        return std::make_pair( boost::bind< value_type_t >( cross_impl_< H >, _1, drawing::cross( cv::Point( p[0], p[1] ), cv::Scalar( p[4], p[3], p[2] ), p[5], p[6], p[7] ) ), true );
    }
    if( e[0] == "circle" ) // todo: quick and dirty, implement using traits
    {
        boost::array< int, 9 > p = {{ 0, 0, 0, 0, 0, 0, 1, 8, 0 }};
        const std::vector< std::string > v = comma::split( e[1], ',' );
        for( unsigned int i = 0; i < v.size(); ++i ) { if( !v[i].empty() ) { p[i] = boost::lexical_cast< int >( v[i] ); } }
        return std::make_pair( boost::bind< value_type_t >( circle_impl_< H >, _1, drawing::circle( cv::Point( p[0], p[1] ), p[2], cv::Scalar( p[5], p[4], p[3] ), p[6], p[7], p[8] ) ), true );
    }
    if( e[0] == "rectangle" || e[0] == "box" ) // todo: quick and dirty, implement using traits
    {
        boost::array< int, 10 > p = {{ 0, 0, 0, 0, 0, 0, 0, 1, 8, 0 }};
        const std::vector< std::string > v = comma::split( e[1], ',' );
        for( unsigned int i = 0; i < v.size(); ++i ) { if( !v[i].empty() ) { p[i] = boost::lexical_cast< int >( v[i] ); } }
        return std::make_pair( boost::bind< value_type_t >( rectangle_impl_< H >, _1, drawing::rectangle( cv::Point( p[0], p[1] ), cv::Point( p[2], p[3] ), cv::Scalar( p[6], p[5], p[4] ), p[7], p[8], p[9] ) ), true );
    }
    if( e[0] == "file" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "file: expected file type like jpg, ppm, etc" ); }
        std::vector< std::string > s = comma::split( e[1], ',' );
        boost::optional< int > quality = boost::make_optional< int >( false, 0 );
        bool do_index = false;
        bool no_header = false;
        std::vector< std::string > filenames;
        for( unsigned int i = 1; i < s.size(); ++i )
        {
            if( s[i] == "index" )
            {
                do_index = true;
            }
            else if( s[i] == "no-header" )
            { 
                no_header = true;
            }
            else if( s[i].substr( 0, 10 ) == "filenames:" ) // quick and dirty
            { 
                std::ifstream ifs( s[i].substr( 10 ) );
                if( !ifs.is_open() ) { COMMA_THROW( comma::exception, "file: failed to open '" << s[i].substr( 11 ) << "'" ); }
                while( ifs.good() && !ifs.eof() )
                {
                    std::string g;
                    std::getline( ifs, g );
                    if( comma::strip( g, " \t" ).empty() ) { continue; }
                    filenames.push_back( g );
                }
                ifs.close();
            }
            else
            {
                quality = boost::lexical_cast< int >( s[i] );
            }
        }
        return std::make_pair( boost::bind< value_type_t >( impl::file< H >( get_timestamp, s[0], no_header, quality, do_index, filenames ), _1 ), false );
    }
    if( e[0] == "save" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "save: please specify filename" ); }
        std::vector< std::string > s = comma::split( e[1], ',' );
        boost::optional< int > quality;
        bool do_index = false;
        bool no_header = false;
        for( unsigned int i = 1; i < s.size(); ++i )
        {
            if( s[i] == "index" ) { do_index = true; }
            else if( s[i] == "no-header" ) { no_header = true; }
            else { quality = boost::lexical_cast< int >( s[i] ); }
        }
        return std::make_pair( boost::bind< value_type_t >( save_impl_< H >( s[0], no_header, quality, do_index ), _1 ), false );
    }
    if( e[0] == "gamma" ) { return std::make_pair( boost::bind< value_type_t >( gamma_impl_< H >, _1, boost::lexical_cast< double >( e[1] ) ), true ); }
    if( e[0] == "pow" || e[0] == "power" ) { return std::make_pair( boost::bind< value_type_t >( pow_impl_< H >, _1, boost::lexical_cast< double >( e[1] ) ), true ); }
    if( e[0] == "remove-mean")
    {
        std::vector< std::string > s = comma::split( e[1], ',' );
        if( s.size() != 2 ) { COMMA_THROW( comma::exception, "remove-mean expected 2 parameters" ); }
        unsigned int neighbourhood_size = boost::lexical_cast< unsigned int >( s[0] );
        cv::Size kernel_size(neighbourhood_size, neighbourhood_size);
        double ratio = boost::lexical_cast< double >( s[1] );
        return std::make_pair( boost::bind< value_type_t >( remove_mean_impl_< H >, _1, kernel_size, ratio ), true );
    }
    if( e[0] == "fft" )
    {
        bool direct = true;
        bool complex = true;
        bool magnitude = false;
        bool log_scale = false;
        bool normalize = false;
        if( e.size() > 1 )
        {
            const std::vector< std::string >& w = comma::split( e[1], ',' );
            for( unsigned int i = 0; i < w.size(); ++i )
            {
                if( w[i] == "direct" ) { direct = true; }
                else if( w[i] == "inverse" ) { direct = false; }
                else if( w[i] == "complex" ) { complex = true; }
                else if( w[i] == "real" ) { complex = false; }
                else if( w[i] == "magnitude" ) { magnitude = true; }
                else if( w[i] == "normalize" ) { normalize = true; }
                else if( w[i] == "log" || w[i] == "log-scale" ) { log_scale = true; }
            }
        }
        return std::make_pair( boost::bind< value_type_t >( fft_impl_< H >, _1, direct, complex, magnitude, log_scale, normalize ), true );
    }
    if( e[0] == "flip" ) { return std::make_pair( boost::bind< value_type_t >( flip_impl_< H >, _1, 0 ), true ); }
    if( e[0] == "flop" ) { return std::make_pair( boost::bind< value_type_t >( flip_impl_< H >, _1, 1 ), true ); }
    if( e[0] == "magnitude" ) { return std::make_pair( boost::bind< value_type_t >( magnitude_impl_< H >, _1 ), true ); }
    if( e[0] == "text" )
    {
        if( e.size() <= 1 ) { COMMA_THROW( comma::exception, "text: expected text value" ); }
        std::vector< std::string > w = comma::split( e[1], ',' );
        cv::Point p( 10, 10 );
        if( w.size() >= 3 ) { p = cv::Point( boost::lexical_cast< unsigned int >( w[1] ), boost::lexical_cast< unsigned int >( w[2] ) ); }
        cv::Scalar s( 0, 0xffff, 0xffff );
        if( w.size() >= 4 )
        {
            if( w[3] == "red" ) { s = cv::Scalar( 0, 0, 0xffff ); }
            else if( w[3] == "green" ) { s = cv::Scalar( 0, 0xffff, 0 ); }
            else if( w[3] == "blue" ) { s = cv::Scalar( 0xffff, 0, 0 ); }
            else if( w[3] == "white" ) { s = cv::Scalar( 0xffff, 0xffff, 0xffff ); }
            else if( w[3] == "black" ) { s = cv::Scalar( 0, 0, 0 ); }
            else if( w[3] == "yellow" ) { s = cv::Scalar( 0, 0xffff, 0xffff ); }
            else { COMMA_THROW( comma::exception, "expected colour of text, e.g. 'red', in \"" << comma::join( e, '=' ) << "\", got '" << w[3] << "'" ); }
        }
        return std::make_pair( boost::bind< value_type_t >( text_impl_< H >, _1, w[0], p, s ), true );
    }
    if( e[0] == "convert-to" || e[0] == "convert_to" )
    {
        if( e.size() <= 1 ) { COMMA_THROW( comma::exception, "convert-to: expected options, got none" ); }
        const std::vector< std::string >& w = comma::split( e[1], ',' );
        boost::unordered_map< std::string, int >::const_iterator it = types_.find( w[0] );
        if( it == types_.end() ) { COMMA_THROW( comma::exception, "convert-to: expected target type, got \"" << w[0] << "\"" ); }
        boost::optional< double > scale( 1.0 );
        boost::optional< double > offset( 0.0 );
        if ( w.size() > 1 ) {
            if ( w[1] == "normalize" ) {
                if ( w.size() != 2 ) { COMMA_THROW( comma::exception, "normalized scale takes no extra parameters" ); }
                scale = boost::none;
                offset = boost::none;
            } else {
                scale = boost::lexical_cast< double >( w[1] );
            }
        }
        if ( w.size() > 2 ) { offset = boost::lexical_cast< double >( w[2] ); }
        return std::make_pair( boost::bind< value_type_t >( convert_to_impl_< H >, _1, it->second, scale, offset ), true );
    }
    if( e[0] == "resize" )
    {
        unsigned int width = 0;
        unsigned int height = 0;
        double w = 0;
        double h = 0;
        const std::vector< std::string >& r = comma::split( e[1], ',' );
        int interpolation = cv::INTER_LINEAR;
        unsigned int size = r.size();
        if( r.size() > 1 && r.back()[0] >= 'a' && r.back()[0] <= 'z' )
        {
            if( r.back() == "nearest" ) { interpolation = cv::INTER_NEAREST; }
            else if( r.back() == "linear" ) { interpolation = cv::INTER_LINEAR; }
            else if( r.back() == "area" ) { interpolation = cv::INTER_AREA; }
            else if( r.back() == "cubic" ) { interpolation = cv::INTER_CUBIC; }
            else if( r.back() == "lanczos4" ) { interpolation = cv::INTER_LANCZOS4; }
            else { COMMA_THROW( comma::exception, "resize: expected interpolation type, got: \"" << e[1] << "\"" ); }
            --size;
        }
        else if( r.size() == 3 ) { interpolation = boost::lexical_cast< int >( r[3] ); }
        switch( size )
        {
            case 1:
                w = h = boost::lexical_cast< double >( r[0] );
                break;
            case 2:
            case 3:
                try { width = boost::lexical_cast< unsigned int >( r[0] ); }
                catch ( ... ) { w = boost::lexical_cast< double >( r[0] ); }
                try { height = boost::lexical_cast< unsigned int >( r[1] ); }
                catch ( ... ) { h = boost::lexical_cast< double >( r[1] ); }
                break;
            default:
                COMMA_THROW( comma::exception, "expected resize=<width>,<height>, got: \"" << e[1] << "\"" );
        }
        return std::make_pair( boost::bind< value_type_t >( resize_impl_< H >, _1, width, height, w, h, interpolation ), true );
    }
    else if( e[0] == "timestamp" ) { return std::make_pair(timestamp_impl_< H >( get_timestamp ), true); }
    else if( e[0] == "transpose" ) { return std::make_pair(transpose_impl_< H >, true); }
    else if( e[0] == "split" ) { return std::make_pair(split_impl_< H >, true); }
    else if( e[0] == "merge" )
    {
        unsigned int default_number_of_channels = 3;
        unsigned int nchannels = e.size() == 1 ? default_number_of_channels : boost::lexical_cast< unsigned int >( e[1] );
        if ( nchannels == 0 ) { COMMA_THROW( comma::exception, "expected positive number of channels in merge filter, got " << nchannels ); }
        return std::make_pair( boost::bind< value_type_t >( merge_impl_< H >, _1, nchannels ), true );
    }
    else if( e[0] == "clone-channels" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "clone-channels: please specify number of channels" ); }
        return std::make_pair( boost::bind< value_type_t >( clone_channels_impl_< H >, _1, boost::lexical_cast< unsigned int >( e[1] ) ), true );
    }
    if( e[0] == "undistort" ) { return std::make_pair( undistort_impl_< H >( e[1] ), true ); }
    if( e[0] == "invert" )
    {
        if( e.size() == 1 ) { return std::make_pair( invert_impl_< H >, true ); }
        else if( e[1] == "brightness" ) { return std::make_pair( invert_brightness_impl_< H >, true ); } // quick and dirty, a secret option
    }
    if(e[0]=="normalize")
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "please specify parameter: expected normalize=<how>" ); }
        const std::vector< std::string > w = comma::split( e[1], ',' );
        int norm_type = cv::NORM_INF;
        if( w.size() == 2 )
        {
            if( w[1] == "inf" || w[1] == "INF" || w[1] == "NORM_INF" ) { norm_type = cv::NORM_INF; }
            else if( w[1] == "l1" || w[1] == "L1" || w[1] == "NORM_L1" ) { norm_type = cv::NORM_L1; }
            else if( w[1] == "l2" || w[1] == "L2" || w[1] == "NORM_L2" ) { norm_type = cv::NORM_L2; }
            else { COMMA_THROW( comma::exception, "expected norm type, got: \"" + w[1] + "\"" ); }
            if( w[0] != "all" && norm_type != cv::NORM_INF ) { COMMA_THROW( comma::exception, "norm type \"" + w[1] + "\" implemented only for 'all'; todo" ); } // quick and dirty
        }
        if( w[0] == "max" ) { return std::make_pair( normalize_max_impl_< H >, true ); }
        else if( w[0] == "sum" ) { return std::make_pair( normalize_sum_impl_< H >, true ); }
        else if( w[0] == "all" ) { return std::make_pair( boost::bind( normalize_cv_impl_< H >, _1, norm_type ), true ); }
        COMMA_THROW( comma::exception, "expected all, max, or sum option for normalize, got: \"" << w[0] << "\"" );
    }
    if( e[0]=="equalize-histogram" ) { return std::make_pair( equalize_histogram_impl_< H >, true ); }
    if( e[0] == "brightness" || e[0] == "scale" )
    {
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        double scale = boost::lexical_cast< double >( s[0] );
        double offset = s.size() == 1 ? 0.0 : boost::lexical_cast< double >( s[1] );
        return std::make_pair( boost::bind< value_type_t >( brightness_impl_< H >, _1, scale, offset ), true );
    }
    if( e[0] == "color-map" )
    {
        if( e.size() != 2 ) { COMMA_THROW( comma::exception, "expected colour-map=<type>; got: \"" << e[1] << "\"" ); }
        int type;
        if( e[1] == "autumn" ) { type = cv::COLORMAP_AUTUMN; }
        else if( e[1] == "bone" ) { type = cv::COLORMAP_BONE; }
        else if( e[1] == "jet" ) { type = cv::COLORMAP_JET; }
        else if( e[1] == "winter" ) { type = cv::COLORMAP_WINTER; }
        else if( e[1] == "rainbow" ) { type = cv::COLORMAP_RAINBOW; }
        else if( e[1] == "ocean" ) { type = cv::COLORMAP_OCEAN; }
        else if( e[1] == "summer" ) { type = cv::COLORMAP_SUMMER; }
        else if( e[1] == "spring" ) { type = cv::COLORMAP_SPRING; }
        else if( e[1] == "cool" ) { type = cv::COLORMAP_COOL; }
        else if( e[1] == "hsv" ) { type = cv::COLORMAP_HSV; }
        else if( e[1] == "pink" ) { type = cv::COLORMAP_PINK; }
        else if( e[1] == "hot" ) { type = cv::COLORMAP_HOT; }
        else { COMMA_THROW( comma::exception, "expected colour-map type; got: \"" << e[1] << "\"" ); }
        return std::make_pair( boost::bind< value_type_t >( colour_map_impl_< H >, _1, type ), true );
    }
    if( e[0] == "blur" )
    {
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        if( s.size() < 2 ) { COMMA_THROW( comma::exception, "expected blur=<blur_type>,<blur_type_parameters>" ); }

        blur_t params;
        params.blur_type = blur_t::from_string(s[0]);
        if (s[0] == "box")
        {
            params.kernel_size.height = params.kernel_size.width = boost::lexical_cast< int >( s[1] );
        }
        else if (s[0] == "gaussian")
        {
            if (s.size() != 3) { COMMA_THROW( comma::exception, "expected blur=gaussian,kernel_size,std" ); }
            params.kernel_size.width = params.kernel_size.height = boost::lexical_cast< int >( s[1] );
            params.std.x = params.std.y = boost::lexical_cast< double >( s[2] );
        }
        else if (s[0] == "median")
        {
            if (s.size() != 2) { COMMA_THROW( comma::exception, "blur=median,kernel_size" ); }
            params.neighbourhood_size = boost::lexical_cast< int >( s[1] );
        }
        else if (s[0] == "bilateral")
        {
            if (s.size() != 4) // nsize,std_space,std_colour
            { COMMA_THROW( comma::exception, "blur=bilateral expected 3 parameters" ); }
            params.neighbourhood_size = boost::lexical_cast< int >( s[1] );
            params.sigma_space = boost::lexical_cast< double >( s[2] );
            params.sigma_colour = boost::lexical_cast< double >( s[3] );
        }
        else if (s[0] == "adaptive-bilateral")
        {
            if (s.size() != 4) { COMMA_THROW( comma::exception, "expected blur=adaptive-bilateral,kernel_size,std_space,std_colour_max" ); }
            params.kernel_size.width = params.kernel_size.height = boost::lexical_cast< int >( s[1] );
            params.sigma_space = boost::lexical_cast< double >( s[2] );
            params.sigma_colour = boost::lexical_cast< double >( s[3] ); // max sigma color
        }
        else { COMMA_THROW( comma::exception, "invalid blur type" ); }
        return std::make_pair( boost::bind< value_type_t >( blur_impl_< H >, _1, params ), true );
    }
    if( e[0] == "load" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "please specify filename load=<filename>" ); }
        // For the case where the file is just a file descriptor, no extenstion, use e[2]
        return std::make_pair( impl::load< H >( e[1] ), true );
    }
    if( e[0] == "map" ) // todo! refactor usage, especially csv option separators and equal sign; make optionally map for each channel separately
    {
#if CV_MAJOR_VERSION <= 2
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "expected file name with the map, e.g. map=f.csv" ); }
        std::stringstream s; s << e[1]; for( std::size_t i = 2; i < e.size(); ++i ) { s << "=" << e[i]; }
        std::string map_filter_options = s.str();
        std::vector< std::string > items = comma::split( map_filter_options, '&' );
        bool permissive = std::find( items.begin()+1, items.end(), "permissive" ) != items.end();
        return std::make_pair( map_impl_ < H >( map_filter_options, permissive ), true );
#else // #if CV_MAJOR_VERSION <= 2
        COMMA_THROW( comma::exception, "map: opencv 3 support: todo" );
#endif // #if CV_MAJOR_VERSION <= 2
    }
    if( e[0] == "inrange" )
    {
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        if( s.size() < 2 || s.size() % 2 != 0 ) { COMMA_THROW( comma::exception, "inrange: expected <upper>,<lower> got: \"" << comma::join( e, '=' ) << "\"" ); }
        cv::Scalar lower = impl::scalar_from_strings( &s[0], s.size() / 2 );
        cv::Scalar upper = impl::scalar_from_strings( &s[ s.size() / 2 ], s.size() / 2 );
        return std::make_pair( boost::bind< value_type_t >( inrange_impl_< H >, _1, lower, upper ), true );
    }
    if( e[0] == "threshold" )
    {
        const std::vector< std::string >& s = comma::split( e[1], ',' );
        if( s[0].empty() ) { COMMA_THROW( comma::exception, "threshold: expected <threshold|otsu>[,<maxval>[,<type>]] got: \"" << comma::join( e, '=' ) << "\"" ); }
        bool otsu = s[0] == "otsu";
        double threshold = otsu ? 0 : boost::lexical_cast< double >( s[0] );
        double maxval = s.size() < 2 ? 255 : boost::lexical_cast< double >( s[1] );
        threshold_t::types type = threshold_t::from_string( s.size() < 3 ? "" : s[2] );
        return std::make_pair( boost::bind< value_type_t >( threshold_impl_< H >, _1, threshold, maxval, type, otsu ), true );
    }
    if ( e[0] == "kmeans" )
    {
        if( e.size() < 2 ) { COMMA_THROW( comma::exception, "expected kmeans=<k>" ); }
        int k = boost::lexical_cast<int>(e[1]);
        return std::make_pair( boost::bind< value_type_t >(kmeans_impl_< H >, _1, k), true );
    }
    if( e[0] == "linear-combination" || e[0] == "ratio" || e[0] == "shuffle" )
    {
        typedef std::string::const_iterator iterator_type;
        ratios::rules< iterator_type > rules;
        ratios::parser< iterator_type, ratios::ratio > parser( rules.ratio_ );
        std::vector< ratios::coefficients > coefficients;
        const std::vector< std::string > & s = comma::split( e[1], ',' );
        if ( s.empty() ) { COMMA_THROW( comma::exception, e[0] << ": empty right-hand side" ); }
        if ( s.size() > 4 ) { COMMA_THROW( comma::exception, e[0] << ": cannot support more then 4 output channels" ); }
        if ( e[0] == "shuffle" ) {
            std::set< std::string > permitted = { "r", "g", "b", "a" };
            for ( const auto & t : s ) {
                if ( t.empty() ) continue;
                if ( permitted.find( t ) != permitted.end() ) continue;
                COMMA_THROW( comma::exception, "shuffle operation allows only symbolic channel names (rgba) or empty fields, not '" << t << "'" );
            }
        }
        for ( const auto & t : s ) {
            if ( t.empty() ) { // pass-through this channel
                coefficients.push_back( ratios::coefficients( ratios::channel::NUM_CHANNELS, std::make_pair( 0.0, 0.0 ) ) );
                coefficients.back()[ coefficients.size() ].first = 1.0;
                coefficients.back()[ 0 ].second = 1.0;
                continue;
            }
            ratios::ratio r;
            iterator_type begin = t.begin();
            iterator_type end = t.end();
            bool status = phrase_parse( begin, end, parser, boost::spirit::ascii::space, r );
            if ( !status || ( begin != end ) ) { COMMA_THROW( comma::exception, e[0] << ": expected a " << e[0] << " expression, got: \"" << t << "\"" ); }
            if ( e[0] == "linear-combination" && !r.denominator.unity() ) { COMMA_THROW( comma::exception, e[0] << ": expected a linear combination expression, got a ratio" ); }
            if( r.numerator.terms.size() != r.denominator.terms.size() ) { COMMA_THROW( comma::exception, e[0] << ": the number of numerator " << r.numerator.terms.size() << " and denominator " << r.denominator.terms.size() << " coefficients differs" ); }
            coefficients.push_back( ratios::coefficients() );
            coefficients.back().reserve( ratios::channel::NUM_CHANNELS );
            for( size_t j = 0; j < r.numerator.terms.size(); ++j ) { coefficients.back().push_back( std::make_pair( r.numerator.terms[j].value, r.denominator.terms[j].value ) ); }
        }
        return std::make_pair( boost::bind< value_type_t >( ratio_impl_< H >, _1, coefficients, e[0] ), true );
    }
    if( snark::cv_mat::morphology::operations().find( e[0] ) != snark::cv_mat::morphology::operations().end() )
    {
        snark::cv_mat::morphology::parameters parameters( e );
        return std::make_pair( boost::bind< value_type_t >( morphology::morphology< H >, _1, snark::cv_mat::morphology::operations().at( e[0] ), parameters.kernel_, parameters.iterations_ ), true );
    }
    if( e[0] == "skeleton" || e[0] == "thinning" )
    {
        return std::make_pair( boost::bind< value_type_t >( morphology::skeleton< H >(snark::cv_mat::morphology::parameters( e )), _1 ), true );
    }
    if( e[0] == "overlay" )
    {
        if( e.size() != 2 ) { COMMA_THROW( comma::exception, "expected file name (and optional x,y) with the overlay, e.g. overlay=a.svg" ); }
        std::vector< std::string > s = comma::split( e[1], ',' );
        if(s.size()!=1 && s.size()!=3) { COMMA_THROW( comma::exception, "expected one or three parameters (file[,x,y]); found " << s.size() ); }
        int x=0, y=0;
        if(s.size()>1)
        {
            x=boost::lexical_cast<int>(s[1]);
            y=boost::lexical_cast<int>(s[2]);
        }
        return std::make_pair( overlay_impl_ < H >( s[0], x, y ), true );
    }
    if( e[0] == "view" )
    {
        unsigned int default_delay = 1; // todo!
        unsigned int delay = default_delay;
        std::string n;
        std::string suffix="ppm";
        if( e.size() > 1 )
        {
            const std::vector< std::string >& w = comma::split( e[1], ',' );
            if( w.size() > 0 && !w[0].empty() ) { delay = boost::lexical_cast< unsigned int >( w[0] ); }
            if( w.size() > 1 ) { n = w[1]; }
            if(w.size()>2) { suffix=w[2]; }
        }
        return std::make_pair( boost::bind< value_type_t >( view_impl_< H >( get_timestamp, n, delay, suffix ), _1 ), false );
    }
    boost::function< value_type_t( value_type_t ) > functor = imaging::vegetation::impl::filters< H >::make_functor( e );
    if( functor ) { return std::make_pair( functor, true ); }
    COMMA_THROW( comma::exception, "expected filter, got: \"" << comma::join( e, '=' ) << "\"" );
}

    struct maker
    {
        maker( const get_timestamp_functor & get_timestamp, char separator = ';', char equal_sign = '=' ) 
            : get_timestamp_( get_timestamp ), separator_( separator ), equal_sign_( equal_sign ) {}
        std::pair< functor_type, bool > operator()( const std::string & s ) const
        {
            const std::vector< std::string > & w = comma::split( s, separator_ );
            std::pair< functor_type, bool > g = make_filter< O, H >::make_filter_functor( comma::split( w[0], equal_sign_ ), get_timestamp_, 1 );
            auto functor = g.first;
            bool parallel = g.second;
            for( unsigned int k = 1; k < w.size(); ++k ) 
            { 
                auto b = make_filter< O, H >::make_filter_functor( comma::split( w[k], equal_sign_ ), get_timestamp_, 1 );
                if( b.second == false ) { parallel = false; } // If any filter must be serial, then turn parallel off
                functor = boost::bind( b.first , boost::bind( functor, _1 ) ); 
            }
            return std::make_pair( functor, parallel );
        }
        
        private:
            const get_timestamp_functor & get_timestamp_;
            char separator_, equal_sign_;
    };

    struct composer
    {
        composer( const maker & m ) : m_( m ) {}
        const maker & m_;

        typedef typename boost::static_visitor< std::pair< functor_type, bool > >::result_type result_type;

        result_type term( const std::string & s ) const { return m_( s ); };
        result_type op_and( const result_type & opl, const result_type & opr ) const { 
            return std::make_pair([ opl, opr ]( const input_type & i ) -> input_type { const input_type & l = opl.first( i ); const input_type & r = opr.first( i ); composer::assert_integer( l ); composer::assert_integer( r ); return std::make_pair( i.first, l.second & r.second ); }, opl.second && opr.second ); }
        result_type op_or( const result_type & opl, const result_type & opr ) const { 
            return std::make_pair([ opl, opr ]( const input_type & i ) -> input_type { const input_type & l = opl.first( i ); const input_type & r = opr.first( i ); composer::assert_integer( l ); composer::assert_integer( r ); return std::make_pair( i.first, l.second | r.second ); }, opl.second && opr.second ); 
        }
        result_type op_xor( const result_type & opl, const result_type & opr ) const { 
            return std::make_pair([ opl, opr ]( const input_type & i ) -> input_type { const input_type & l = opl.first( i ); const input_type & r = opr.first( i ); composer::assert_integer( l ); composer::assert_integer( r ); return std::make_pair( i.first, l.second ^ r.second ); }, opl.second && opr.second ); 
        }
        result_type op_not( const result_type & op ) const { return std::make_pair( [ op ]( const input_type & i ) -> input_type { const input_type & o = op.first( i ); composer::assert_integer( o ); return std::make_pair( i.first, ~o.second ); }, op.second ); }

        static void assert_integer( const input_type & i )
        {
            if ( i.second.depth() == CV_32F || i.second.depth() == CV_64F ) { COMMA_THROW( comma::exception, "bitwise operations shall be done on integer inputs" ); }
        }
    };

};

template < typename H > struct time_traits
{
    static boost::posix_time::ptime pass_time( const H& h ) { COMMA_THROW( comma::exception, "cannot make timestamp out of header without binary serializer" ); }
};

template <> struct time_traits< boost::posix_time::ptime >
{
    static boost::posix_time::ptime pass_time( const boost::posix_time::ptime& t ) { return t; }
};

template < typename H >
std::vector< typename impl::filters< H >::filter_type > impl::filters< H >::make( const std::string& how, unsigned int default_delay )
{
    return impl::filters< H >::make( how, boost::bind( &time_traits< H >::pass_time, _1 ), default_delay );
}

template < typename H >
std::vector< typename impl::filters< H >::filter_type > impl::filters< H >::make( const std::string& how, const get_timestamp_functor& get_timestamp, unsigned int default_delay )
{
    typedef typename impl::filters< H >::value_type value_type_t;
    typedef typename impl::filters< H >::filter_type filter_type;
    typedef typename filter_type::input_type input_type;
    typedef typename make_filter< cv::Mat, H >::maker maker_t;
    typedef typename make_filter< cv::Mat, H >::composer composer_t;

    std::vector< std::string > v = comma::split( how, ';' );
    std::vector< filter_type > f;
    if( how == "" ) { return f; }
    std::string name;
    bool modified = false;
    for( std::size_t i = 0; i < v.size(); name += ( i > 0 ? ";" : "" ) + v[i], ++i )
    {
        std::vector< std::string > e = comma::split( v[i], '=' );
        if( e[0] == "mask" )
        {
             if( e.size() == 1 ) { COMMA_THROW( comma::exception, "mask: please specify mask filters" ); }
             if( e.size() > 2 ) { COMMA_THROW( comma::exception, "mask: expected 1 parameter; got: " << comma::join( e, '=' ) ); }
             snark::cv_mat::bitwise::expr result = snark::cv_mat::bitwise::parse( e[1] );
             maker_t m( get_timestamp, '|', ':' );
             composer_t c( m );
             auto g = boost::apply_visitor( snark::cv_mat::bitwise::visitor< input_type, input_type, composer_t >( c ), result );
             f.push_back( filter_type( boost::bind< value_type_t >( mask_impl_< H >( g.first ), _1 ), g.second ) );
        }
        else if( e[0] == "tee" )
        {
             if( e.size() == 1 ) { COMMA_THROW( comma::exception, "tee: please specify tee filters" ); }
             if( e.size() > 2 ) { COMMA_THROW( comma::exception, "tee: expected 1 parameter; got: " << comma::join( e, '=' ) ); }
             snark::cv_mat::bitwise::expr result = snark::cv_mat::bitwise::parse( e[1] );
             maker_t m( get_timestamp, '|', ':' );
             composer_t c( m );
             auto g = boost::apply_visitor( snark::cv_mat::bitwise::visitor< input_type, input_type, composer_t >( c ), result );
             f.push_back( filter_type( boost::bind< value_type_t >( tee_impl_< H >( g.first ), _1 ), g.second ) );
        }
        else if( e[0] == "multiply" || e[0] == "divide" || e[0] == "add" || e[0] == "subtract" || e[0] == "absdiff" )
        {
             if( e.size() == 1 ) { COMMA_THROW( comma::exception, e[0] << ": please specify " << e[0] << " filters" ); }
             if( e.size() > 2 ) { COMMA_THROW( comma::exception, e[0] << ": expected 1 parameter; got: " << comma::join( e, '=' ) ); }
             snark::cv_mat::bitwise::expr result = snark::cv_mat::bitwise::parse( e[1] );
             maker_t m( get_timestamp, '|', ':' );
             composer_t c( m );
             auto operand_filters = boost::apply_visitor( snark::cv_mat::bitwise::visitor< input_type, input_type, composer_t >( c ), result );
             auto op = arithmetic< H >::str_to_operation(e[0]);
             f.push_back( filter_type( boost::bind< value_type_t >( arithmetic< H >( op ), _1, operand_filters.first ), operand_filters.second ) );
        }
        else if( e[0] == "bitwise" )
        {
             if( e.size() == 1 ) { COMMA_THROW( comma::exception, e[0] << ": please specify " << e[0] << " filters" ); }
             if( e.size() > 2 ) { COMMA_THROW( comma::exception, e[0] << ": expected 1 parameter; got: " << comma::join( e, '=' ) ); }
             snark::cv_mat::bitwise::expr result = snark::cv_mat::bitwise::parse( e[1] );
             maker_t m( get_timestamp, '|', ':' );
             composer_t c( m );
             auto operand_filters = boost::apply_visitor( snark::cv_mat::bitwise::visitor< input_type, input_type, composer_t >( c ), result );
             f.push_back( filter_type( boost::bind< value_type_t >( bitwise_impl_< H >( operand_filters.first ), _1 ), operand_filters.second ) );
        }
        else if( e[0] == "bayer" ) // kept for backwards-compatibility, use convert-color=BayerBG,BGR etc..
        {
            if( modified ) { COMMA_THROW( comma::exception, "cannot covert from bayer after transforms: " << name ); }
            unsigned int which = boost::lexical_cast< unsigned int >( e[1] ) + 45u; // HACK, bayer as unsigned int, but I don't find enum { BG2RGB, GB2BGR ... } more usefull
            f.push_back( filter_type( boost::bind< value_type_t >( cvt_color_impl_< H >(), _1, which ) ) );
        }
        else if( e[0] == "pack" )
        {
            if( e.size() < 2 ) { COMMA_THROW( comma::exception, "pack: missing arguments" ); }
            std::vector< std::string > args = comma::split( e[1], ',' );
            if( args[0] == "12" )
            {
                if( args.size() != 4 ) { COMMA_THROW( comma::exception, "12-bit pack expects 3 formats, got: " << args.size() - 1 ); }
                f.push_back( filter_type( pixel_format_impl_< H, 2, CV_16UC1, 3, CV_8UC1 >( e[0], { args[1], args[2], args[3] } ) ) );
            }
            else { COMMA_THROW( comma::exception, "pack size not implemented: " << args[0] ); }
        }
        else if( e[0] == "unpack" )
        {
            if( e.size() < 2 ) { COMMA_THROW( comma::exception, "unpack: missing arguments"); }
            std::vector< std::string > args = comma::split( e[1], ',' );
            if( args[0] == "12" )
            {
                if( args.size() != 3 ) { COMMA_THROW( comma::exception, "12-bit unpack expects 2 formats, got: " << args.size() - 1 ); }
                f.push_back( filter_type( pixel_format_impl_< H, 3, CV_8UC1, 2, CV_16UC1 >( e[0], { args[1], args[2] } ) ) );
            }
            else { COMMA_THROW( comma::exception, "unpack size not implemented: " << args[0] ); }
        }
        else if( e[0] == "unpack12" )
        {
            if( modified ) { COMMA_THROW( comma::exception, "cannot covert from 12 bit packed after transforms: " << name ); }
            if(e.size()!=1) { COMMA_THROW( comma::exception, "unexpected arguement: "<<e[1]); }
            f.push_back( filter_type( boost::bind< value_type_t >( unpack12_impl_< H >, _1 ) ) );
        }
        else if( e[0] == "log" ) // todo: rotate log by size: expose to user
        {
            boost::optional < double > period = boost::make_optional< double >( false, 0 );
            boost::optional < comma::uint32 > size = boost::make_optional< comma::uint32 >( false, 0 );
            bool index = false;
            if( e.size() <= 1 ) { COMMA_THROW( comma::exception, "please specify log=<filename> or log=<directory>[,<options>]" ); }
            const std::vector< std::string >& w = comma::split( e[1], ',' );

            std::string file = w[0];
            for ( std::size_t option = 1; option < w.size(); ++option )
            {
                const std::vector< std::string >& u = comma::split( w[option], ':' );
                if( u[0] == "period" )
                {
                    if (size) { COMMA_THROW( comma::exception, "log: expected \"period\" or \"size\"; got both" ); }
                    if( u.size() <= 1 ) { COMMA_THROW( comma::exception, "log: please specify period:<seconds>" ); }
                    period = boost::lexical_cast< double >( u[1] );
                }
                else if( u[0] == "size" )
                {
                    if (period) { COMMA_THROW( comma::exception, "log: expected \"period\" or \"size\"; got both" ); }
                    if( u.size() <= 1 ) { COMMA_THROW( comma::exception, "log: please specify size:<number of frames>" ); }
                    size = boost::lexical_cast< comma::uint32 >( u[1] );
                }
                else if( u[0] == "index" )
                {
                    index = true;
                }
                else
                {
                    COMMA_THROW( comma::exception, "log: expected \"index\", \"period\" or \"size\"; got: \"" << u[0] << "\"" );
                }
            }
            if (period)
            {
                unsigned int seconds = static_cast< unsigned int >( *period );
                f.push_back( filter_type( log_impl_< H >( file, boost::posix_time::seconds( seconds ) + boost::posix_time::microseconds( static_cast< unsigned int >( ( *period - seconds ) * 1000000 ) ), index, get_timestamp ), false ) );
            }
            else if ( size )
            {
                f.push_back( filter_type( log_impl_< H >( file, *size, index, get_timestamp), false ) );
            }
            else
            {
                if( index ) { COMMA_THROW( comma::exception, "log: index should be specified with directory and period or size, not with filename" );  }
                f.push_back( filter_type( log_impl_< H >( file, get_timestamp ), false ) );
            }
        }
        else if( e[0] == "max" ) // todo: remove this filter; not thread-safe, should be run with --threads=1
        {
            f.push_back( filter_type( max_impl_< H >( boost::lexical_cast< unsigned int >( e[1] ), true ), false ) );
        }
        else if( e[0] == "min" ) // todo: remove this filter; not thread-safe, should be run with --threads=1
        {
            f.push_back( filter_type( max_impl_< H >( boost::lexical_cast< unsigned int >( e[1] ), false ), false ) );
        }
        else if( e[0] == "encode" )
        {
            if( i < v.size() - 1 )
            {
                std::string next_filter = comma::split( v[i+1], '=' )[0];
                if( next_filter != "head" ) COMMA_THROW( comma::exception, "cannot have a filter after encode unless next filter is head" );
            }
            if( e.size() < 2 ) { COMMA_THROW( comma::exception, "expected encoding type like jpg, ppm, etc" ); }
            std::vector< std::string > s = comma::split( e[1], ',' );
            boost::optional < int > quality;
            if (s.size()> 1) { quality = boost::lexical_cast<int>(s[1]); }
            f.push_back( filter_type( boost::bind< value_type_t >( encode_impl_< H >( get_timestamp ), _1, s[0], quality ), false ) );
        }
        else if( e[0] == "grab" )
        {
            if( e.size() < 2 ) { COMMA_THROW( comma::exception, "expected encoding type like jpg, ppm, etc" ); }
            std::vector< std::string > s = comma::split( e[1], ',' );
            boost::optional < int > quality;
            if (s.size() == 1) { quality = boost::lexical_cast<int>(s[1]); }
            f.push_back( filter_type( boost::bind< value_type_t >( grab_impl_< H >(get_timestamp), _1, s[0], quality ) ) );
        }
//         else if( e[0] == "file" )
//         {
//             if( e.size() < 2 ) { COMMA_THROW( comma::exception, "expected file type like jpg, ppm, etc" ); }
//             std::vector< std::string > s = comma::split( e[1], ',' );
//             boost::optional< int > quality;
//             bool do_index = false;
//             bool no_header = false;
//             for( unsigned int i = 1; i < s.size(); ++i )
//             {
//                 if( s[i] == "index" ) { do_index = true; }
//                 else if( s[i] == "no-header" ) { no_header = true; }
//                 else { quality = boost::lexical_cast< int >( s[i] ); }
//             }
//             f.push_back( filter_type( boost::bind< value_type_t >( impl::file< H >( get_timestamp, no_header ), _1, s[0], quality, do_index ), false ) );
//         }
        else if( e[0] == "simple-blob" )
        {
#if CV_MAJOR_VERSION <= 2
            if( i < v.size() - 1 ) { COMMA_THROW( comma::exception, "expected 'simple-blob' as the last filter, got \"" << how << "\"" ); }
            std::vector< std::string > s;
            if( e.size() > 1 ) { s = comma::split( e[1], ',' ); }
            bool is_binary = false;
            std::string config;
            std::string path;
            for( unsigned int i = 0; i < s.size(); ++i )
            {
                if( s[i] == "output-binary" ) { is_binary = true; }
                else if( s[i] == "output-fields" ) { std::cout << comma::join( comma::csv::names< snark::timestamped< cv::KeyPoint > >( false ), ',' ) << std::endl; exit( 0 ); }
                else if( s[i] == "output-format" ) { std::cout << comma::csv::format::value< snark::timestamped< cv::KeyPoint > >() << std::endl; exit( 0 ); }
                else if( s[i] == "output-default-params" || s[i] == "default-params" )
                {
                    cv::FileStorage fs( "dummy", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_XML );
                    cv::SimpleBlobDetector::Params().write( fs );
                    std::cout << fs.releaseAndGetString() << std::endl;
                    exit( 0 ); // hyper quick and dirty
                }
                else
                {
                    const std::vector< std::string >& t = comma::split( s[i], ':' );
                    config = t[0];
                    if( t.size() > 1 ) { path = s[1]; }
                }
            }
            f.push_back( filter_type( boost::bind< value_type_t >( simple_blob_impl_< H >(get_timestamp), _1, cv_read_< cv::SimpleBlobDetector::Params >( config, path ), is_binary ), false ) );
            f.push_back( filter_type( NULL ) ); // quick and dirty
#else // #if CV_MAJOR_VERSION <= 2
            COMMA_THROW( comma::exception, "simple-blob: opencv 3 support: todo" );
#endif // #if CV_MAJOR_VERSION <= 2
        }
        else if( e[0] == "null" )
        {
            if( i < v.size() - 1 ) { COMMA_THROW( comma::exception, "expected 'null' as the last filter, got \"" << how << "\"" ); }
            if( i == 0 ) { COMMA_THROW( comma::exception, "'null' as the only filter is not supported; use cv-cat > /dev/null, if you need" ); }
            f.push_back( filter_type( NULL ) );
        }
        else if ( e[0] == "head" )
        {
            if( i < v.size() - 1 )
            {
                std::string next_filter = comma::split( v[i+1], '=' )[0];
                if( next_filter != "null" && next_filter != "encode" && next_filter != "file") { COMMA_THROW( comma::exception, "cannot have a filter after head unless next filter is null or encode or file" ); }
            }
            unsigned int n = e.size() < 2 ? 1 : boost::lexical_cast< unsigned int >( e[1] );
            f.push_back( filter_type( boost::bind< value_type_t >( head_impl_< H >, _1, n ), false ) );
        }
        else if( e[0] == "rotate90" )
        {
            int n = e.size() > 1 ? boost::lexical_cast< int >( e[1] ) % 4 : 1;
            if( n < 0 ) { n += 4; }
            std::vector< std::string > filters;
            switch( n )
            {
                case 1: filters = { "transpose", "flop" };  break;
                case 2: filters = { "flip",      "flop" };  break;
                case 3: filters = { "transpose", "flip" };  break;
                default: break;
            }
            // They are all parallel=true
            for( std::string s : filters ) { f.push_back( filter_type( make_filter< cv::Mat, H >::make_filter_functor( { s }, get_timestamp, default_delay ) ) ); }
        }
        else
        {
            f.push_back( filter_type( make_filter< cv::Mat, H >::make_filter_functor( e, get_timestamp, default_delay ) ) );
        }
        modified = e[0] != "view" && e[0] != "tee" && e[0] != "split" && e[0] !="unpack12"; // do we even need it?
    }
    return f;
}

template < typename H >
typename impl::filters< H >::value_type impl::filters< H >::apply( std::vector< filter_type >& filters, typename impl::filters< H >::value_type m )
{
    for( std::size_t i = 0; i < filters.size(); m = filters[ i++ ].filter_function( m ) );
    return m;
}

template < typename H >
static std::string usage_impl_()
{
    std::ostringstream oss;
    oss << std::endl;
    oss << "    OpenCV version: " << CV_MAJOR_VERSION << std::endl;
    oss << std::endl;
    oss << "    cv::Mat image filters usage (';'-separated):" << std::endl;
    oss << "        accumulate=<n>[,<how>]: accumulate the last n images and concatenate them vertically (useful for slit-scan and spectral cameras like pika2)" << std::endl;
    oss << "            example: cat slit-scan.bin | cv-cat \"accumulate=400;view;null\"" << std::endl;
    oss << "            <how>" << std::endl;
    oss << "                sliding: sliding window" << std::endl;
    oss << "                fixed: input image location in the output image is defined by its number modulo <n>" << std::endl;
    oss << "                fixed-reverse: same as fixed, but in the opposite order" << std::endl;
    oss << "                default: sliding" << std::endl;
    oss << "                example: run the command line below; press white space key to see image accumulating; try it with fixed or fixed-inverse" << std::endl;
    oss << "                    > ( yes 255 | head -n $(( 64 * 64 * 20 )) | csv-to-bin ub ) | cv-cat --input 'no-header;rows=64;cols=64;type=ub' 'count;accumulate=20,sliding;view=0;null'" << std::endl;
    oss << "        accumulated=<operation>: apply a pixel-wise operation to the input images" << std::endl;
    oss << "            <operation>" << std::endl;
    oss << "                 average: pixelwise average using all images from the beginning of the stream" << std::endl;
    oss << "                 ema,alpha[,<spin_up>]: pixelwise exponential moving average" << std::endl;
    oss << "                     <alpha>: range: between 0 and 1.0, larger value will retain more historical data." << std::endl;
    oss << "                     <spin_up>: default = 1;" << std::endl;
    oss << "                     formula: ema += (new_pixel_value - ema) * <alpha>" << std::endl;
    oss << "                 min: image minimum" << std::endl;
    oss << "                 max: image maximum" << std::endl;
    oss << "                 moving-average,<window>: pixelwise moving average" << std::endl;
    oss << "                     <window>: number of images in sliding window" << std::endl;
    oss << "        bayer=<mode>: convert from bayer, <mode>=1-4 (see also convert-color)" << std::endl;
    oss << "        blur=<type>,<parameters>: apply a blur to the image (positive and odd kernel sizes)" << std::endl;
    oss << "            blur=box,<kernel_size> " << std::endl;
    oss << "            blur=median,<kernel_size>" << std::endl;
    oss << "            blur=gaussian,<kernel_size>,<std_size>. Set either std or kernel size to 0 to calculate based on other parameter." << std::endl;
    oss << "            blur=bilateral,<neighbourhood_size>,<sigma_space>,<sigma_colour>; preserves edges" << std::endl;
    oss << "            blur=adaptive-bilateral: <kernel_size>,<sigma_space>,<sigma_colour_max>; preserve edges, automatically calculate sigma_colour (and cap to sigma_colour_max)" << std::endl;
    oss << "        brightness,scale=<scale>[,<offset>]: output=(scale*input)+offset; default offset=0" << std::endl;
    oss << "        canny=<threshold1>,<threshold2>[,<kernel_size>]: finds edges using the Canny86 algorithm (see cv::Canny)" << std::endl;
    oss << "                                                         generates a mask with bright lines representing the edges on a black background, requires single-channel 8-bit input image" << std::endl;
    oss << "                threshold1, threshold2: the smaller value is used for edge linking, the larger value is used to find initial segments of strong edges" << std::endl;
    oss << "        clahe=<clip_limit>,<tile_size_x>[,<tile_size_y>]: CLAHE, contrast limited adaptive histogram equalization (see opencv documentation for more), e.g. try clahe=2.0,8,8" << std::endl;
    oss << "                kernel_size: size of the extended Sobel kernel; it must be 1, 3, 5 or 7" << std::endl;
    oss << "        color-map=<type>: take image, apply colour map; see cv::applyColorMap for detail" << std::endl;
    oss << "            <type>: autumn, bone, jet, winter, rainbow, ocean, summer, spring, cool, hsv, pink, hot" << std::endl;
    oss << "        convert-to,convert_to=<type>[,<scale>[,<offset>]]: convert to given type; should be the same number of channels; see opencv convertTo for details; values will not overflow" << std::endl;
    oss << "        convert-color,convert_color=<from>,<to>: convert from colour space to new colour space (BGR, RGB, Lab, XYZ, Bayer**, GRAY); eg: BGR,GRAY or CV_BGR2GRAY" << std::endl;
    oss << "        count: write frame number on images" << std::endl;
    oss << "        crop=[<x>,<y>],<width>,<height>: crop the portion of the image starting at x,y with size width x height" << std::endl;
    oss << "        crop-tile=<ncols>,<nrows>[,<i>,<j>,...[,horizontal]]: divide the image into a grid of tiles (ncols-by-nrows), and output an image made of the croped tiles defined by i,j (count from zero)" << std::endl;
    oss << "                                                              if width or height of input image is not divisible by the corresponding count, the image will be clipped" << std::endl;
    oss << "            horizontal: if present, tiles will be stacked horizontally (by default, vertical stacking is used)" << std::endl;
    oss << "            examples" << std::endl;
    oss << "                \"crop-tile=2,5,1,0,2,3\": crop 2 tiles of the image 2x5 tiles: tile at 1,0 and at 2,3" << std::endl;
    oss << "                \"crop-tile=2,5\": similar to tile operation; crop all tiles from the image of 2x5 tiles; output image will be composed of 2*5=10 tiles" << std::endl;
    oss << "                \"crop-tile=2,5,1,0,2,3,horizontal\": crop 2 tiles out of image split into 2x5 tiles: tile at 1,0 and at 2,3; arrange tiles horizontally in the output image" << std::endl;
    oss << "            deprecated: old syntax <i>,<j>,<ncols>,<nrows> is used for one tile if i < ncols and j < ncols" << std::endl;
    oss << "        encode=<format>[,<quality>]: encode images to the specified format. <format>: jpg|ppm|png|tiff..., make sure to use --no-header" << std::endl;
    oss << "                                     <quality>: for jpg files, compression quality from 0 (smallest) to 100 (best)" << std::endl;
    oss << "        equalize-histogram: equalize each channel by its histogram" << std::endl;
    oss << "        fft[=<options>]: do fft on a floating point image" << std::endl;
    oss << "            options: inverse: do inverse fft" << std::endl;
    oss << "                     real: output real part only" << std::endl;
    oss << "                     magnitude: output magnitude only" << std::endl;
    oss << "            examples: cv-cat --file image.jpg \"split;crop-tile=2,5,0,0,1,3;convert-to=f,0.0039;fft;fft=inverse,magnitude;view;null\"" << std::endl;
    oss << "                      cv-cat --file image.jpg \"split;crop-tile=2,5,0,0,1,3;convert-to=f,0.0039;fft=magnitude;convert-to=f,40000;view;null\"" << std::endl;
    oss << "        flip: flip vertically" << std::endl;
    oss << "        flop: flip horizontally" << std::endl;
    oss << "        grab=<format>[,<quality>]: write an image to file with timestamp as name in the specified format. <format>: jpg|ppm|png|tiff..., if no timestamp, system time is used" << std::endl;
    oss << "                                   <quality>: for jpg files, compression quality from 0 (smallest) to 100 (best)" << std::endl;
    oss << "        head=<n>: output <n> frames and exit" << std::endl;
    oss << "        inrange=<lower>,<upper>: a band filter on r,g,b or greyscale image; for rgb: <lower>::=<r>,<g>,<b>; <upper>::=<r>,<g>,<b>; see cv::inRange() for detail" << std::endl;
    oss << "        invert: invert image (to negative)" << std::endl;
    oss << "        kmeans=<k>[,<params>]: perform k-means clustering on image and replace each pixel with the mean of its cluster" << std::endl;
    oss << "        magnitude: calculate magnitude for a 2-channel image; see cv::magnitude() for details" << std::endl;
    oss << "        map=<map file>[&<csv options>][&permissive]: map integer values to floating point values read from the map file" << std::endl;
    oss << "             <csv options>: usual csv options for map file, but &-separated (running out of separator characters)" << std::endl;
    oss << "                  fields: key,value; default: value" << std::endl;
    oss << "                  default: read a single column of floating point values (with the row counter starting from zero used as key)" << std::endl;
    oss << "             <permissive>: if present, integer values in the input are simply copied to the output unless they are in the map" << std::endl;
    oss << "                  default: filter fails with an error message if it encounters an integer value which is not in the map" << std::endl;
    oss << "             example: \"map=map.bin&fields=,key,value&binary=2ui,d\"" << std::endl;
    oss << "        normalize=<how>[,<norm_type>]: normalize image and scale to 0 to 1 float (or double if input is CV_64F)" << std::endl;
    oss << "            <how>" << std::endl;
    oss << "                normalize=max: normalize each pixel channel by its max value" << std::endl;
    oss << "                normalize=sum: normalize each pixel channel by the sum of all channels" << std::endl;
    oss << "                normalize=all: normalize each pixel by max of all channels (cv::normalize())" << std::endl;
    oss << "            <norm_type>" << std::endl;
    oss << "                inf: cv::NORM_INF" << std::endl;
    oss << "                l1: cv::NORM_L1" << std::endl;
    oss << "                l2: cv::NORM_L2" << std::endl;
    oss << "                default: inf (for backward compatibility)" << std::endl;
    oss << "        null: same as linux /dev/null (since windows does not have it)" << std::endl;
    oss << "        overlay=<image_file>[,x,y]: overlay image_file on top of current stream at optional x,y location; overlay image should have alpha channel" << std::endl;
    oss << "        pack=<bits>,<format>: pack pixel data in specified bits, example: pack=12,cd,hb,fg" << std::endl;
    oss << "            <bits>: number of bits, currently only 12-bit packing from 16-bit is supported (pack 2 pixels into 3 bytes)" << std::endl;
    oss << "            <format>: output pixel formats in quadbits" << std::endl;
    oss << "                where 'a' is high quadbit of byte 0, 'b' is low quadbit of byte 0, 'c' is high quadbit of byte 1, etc... and '0' means quadbit zero" << std::endl;
    oss << "        pow,power=<value>; each image channel power, currently plain wrapper of opencv pow(), thus may be slow; todo? parallelize and/or implement mapping with interpolation" << std::endl;
    oss << "        resize=<factor>[,<interpolation>]; resize=<width>,<height>[,<interpolation>]" << std::endl;
    oss << "            <interpolation>: nearest, linear, area, cubic, lanczos4; default: linear" << std::endl;
    oss << "                             in format <width>,<height>,<interpolation> corresponding numeric values can be used: " << cv::INTER_NEAREST << ", " << cv::INTER_LINEAR << ", " << cv::INTER_AREA << ", " << cv::INTER_CUBIC << ", " << cv::INTER_LANCZOS4 << std::endl;
    oss << "            examples" << std::endl;
    oss << "                resize=0.5,1024 : 50% of width; heigth 1024 pixels" << std::endl;
    oss << "                resize=512,1024,cubic : resize to 512x1024 pixels, cubic interpolation" << std::endl;
    oss << "                resize=0.2,0.4 : resize to 20% of width and 40% of height" << std::endl;
    oss << "                resize=0.5 : resize proportionally to 50%" << std::endl;
    oss << "                resize=0.5,1024 : 50% of width; heigth 1024 pixels" << std::endl;
    oss << "            note: if no decimal dot '.', size is in pixels; if decimal dot present, size as a fraction" << std::endl;
    oss << "                  i.e. 5 means 5 pixels; 5.0 means 5 times" << std::endl;
    oss << "        remove-mean=<kernel_size>,<ratio>: simple high-pass filter removing <ratio> times the mean component on <kernel_size> scale" << std::endl;
    oss << "        rotate90[=n]: rotate image 90 degrees clockwise n times (default: 1); sign denotes direction (convenience wrapper around { tranpose, flip, flop })" << std::endl;
    oss << "        text=<text>[,x,y][,colour]: print text; default x,y: 10,10; default colour: yellow" << std::endl;
    oss << "        threshold=<threshold|otsu>[,<maxval>[,<type>]]: threshold image; same semantics as cv::threshold()" << std::endl;
    oss << "            <threshold|otsu>: threshold value; if 'otsu' then the optimum threshold value using the Otsu's algorithm is used (only for 8-bit images)" << std::endl;
    oss << "            <maxval>: maximum value to use with the binary and binary_inv thresholding types (default:255)" << std::endl;
    oss << "            <type>: binary, binary_inv, trunc, tozero, tozero_inv (default:binary)" << std::endl;
    oss << "        timestamp: write timestamp on images" << std::endl;
    oss << "        transpose: transpose the image (swap rows and columns)" << std::endl;
    oss << "        undistort=<how>: undistort" << std::endl;
    oss << "            <how>" << std::endl;
    oss << "                <undistort_map_filename>, e.g. map.bin" << std::endl;
    oss << "                <pinhole_config_filename>[:<path>], e.g. config.json:camera/pinhole" << std::endl;
    oss << "        unpack=<bits>,<format>: unpack compressed pixel formats, example: unpack=12,cba0,fed0" << std::endl;
    oss << "            <bits>: number of bits, currently only 12 is supported" << std::endl;
    oss << "            <format>: output pixel formats in quadbits" << std::endl;
    oss << "                where 'a' is high quadbit of byte 0, 'b' is low quadbit of byte 0, 'c' is high quadbit of byte 1, etc... and '0' means quadbit zero" << std::endl;
    oss << "        unpack12: convert from 12-bit packed (2 pixels in 3 bytes) to 16UC1; use before other filters; equivalent to unpack=12,abc0,efd0" << std::endl;
    oss << "        untile=<ncols>,<nrows>[,horizontal]: inverse to tile; input image used as a stip of <ncols>*<nrows> tiles, output image will be be composed by <nrows> of tiles with <ncols> tiles per row" << std::endl;
    oss << "                                             if width or height of input image is not divisible by the corresponding count, the input image will be clipped" << std::endl;
    oss << "                                             horizontal: if present, input tiles are considered stacked horizontally (by default, vertical stacking is used)" << std::endl;
    oss << "                                             example" << std::endl;
    oss << "                                                 > ( yes 255 | head -n $(( 64 * 64 * 20 )) | csv-to-bin ub ) | cv-cat --input 'no-header;rows=64;cols=64;type=ub' 'count' | cv-cat --input 'no-header;rows=1280;cols=64;type=ub' 'encode=png' > original.png" << std::endl;
    oss << "                                                 > eog original.png" << std::endl;
    oss << "                                                 > ( yes 255 | head -n $(( 64 * 64 * 20 )) | csv-to-bin ub ) | cv-cat --input 'no-header;rows=64;cols=64;type=ub' 'count' | cv-cat --input 'no-header;rows=1280;cols=64;type=ub' 'untile=5,4;encode=png' > untiled.png" << std::endl;
    oss << "                                                 > eog untiled.png" << std::endl;
    oss << "        view[=<wait-interval>[,<name>[,<suffix]]]: view image;" << std::endl; 
    oss << "                                press <space> to save image (timestamp or system time as filename); " << std::endl;
    oss << "                                press <esc>: to close" << std::endl;
    oss << "                                press numerical '0' to '9' to add the id (0-9) to the file name: <timestamp>.<id>.<suffix>" << std::endl;
    oss << "                                press any other key to show the next frame" << std::endl;
    oss << "                                <wait-interval>: a hack for now; milliseconds to wait for image display and key press (0 waits indefinitely); default 1" << std::endl;
    oss << "                                <name>: view window name; default: the number of view occurence in the filter string" << std::endl;
    oss << "                                <suffix>: image suffix type e.g. png, default ppm" << std::endl;
    oss << "            attention! it seems that lately using cv::imshow() in multithreaded context has been broken in opencv or in underlying x window stuff" << std::endl;
    oss << "                       therefore, unfortunately:" << std::endl;
    oss << "                           instead of: cv-cat 'view;do-something;view'" << std::endl;
    oss << "                                  use: cv-cat 'view;do-something' | cv-cat 'view'" << std::endl;
    oss << std::endl;
    oss << "    file read/write operations" << std::endl;
    oss << "        file=<format>[,<quality>][,index][,no-header][,filenames:<filenames>]: write images to files with timestamp as name in the specified format; if input images have no timestamp, system time is used" << std::endl;
    oss << "            <format>" << std::endl;
    oss << "                - anything that opencv imwrite can take, e.g. jpg, ppm, png, tiff etc" << std::endl;
    oss << "                - 'bin' to write image as binary in cv-cat format" << std::endl;
    oss << "            <quality>: for jpg files, compression quality from 0 (smallest) to 100 (best)" << std::endl;
    oss << "            index: if present, for each timestamp, files will be named as: <timestamp>.<index>.<extension>, e.g: 20170101T000000.123456.0.png, 20170101T000000.123456.1.png, etc" << std::endl;
    oss << "            no-header: makes sense only for 'bin' format; if present, write image without header" << std::endl;
//    oss << "            <filenames>: file containing filenames ..." << std::endl;
    oss << "        load=<filename>: load image from file instead of taking an image on stdin; the main meaningful use would be in association with 'forked' image processing" << std::endl;
    oss << "            supported file types by filename extension:" << std::endl;
    oss << "                - .bin or <no filename extension>: file is in cv-cat binary format: <t>,<rows>,<cols>,<type>,<image data>" << std::endl;
    oss << "                - otherwise whatever cv::imread supports" << std::endl;
    oss << "        log=<options>: write images to files" << std::endl;
    oss << "            log=<filename>: write images to a single file" << std::endl;
    oss << "            log=<dirname>,size:<number of frames>: write images to files in a given directory, each file (except possibly the last one) containing <number of frames> frames" << std::endl;
    oss << "            log=<dirname>,period:<seconds>: write images to files in a given directory, each file containing frames for a given period of time" << std::endl;
    oss << "                                            e.g. for log=tmp,period:1.5 each file will contain 1.5 seconds worth of images" << std::endl;
    oss << "            log=<options>,index: write index file, describing file number and offset of each frame" << std::endl;
    oss << "        save=<filename>[,<quality>][,index][,no-header]: write images to files with timestamp as name in the specified format. <filename>: <base>.<format>: bin|jpg|ppm|png|tiff...; if no timestamp, system time is used" << std::endl;
    oss << "            <format>: anything that opencv imwrite can take or 'bin' to write image as binary in cv-cat format" << std::endl;
    oss << "            <quality>: for jpg files, compression quality from 0 (smallest) to 100 (best)" << std::endl;
    oss << "            index: if present, for each image, files will be named as: <base>.<index>.<extension>, e.g: my-file.0.png, my-file.1.png, etc" << std::endl;
    oss << std::endl;
    oss << "    operations on channels" << std::endl;
    oss << "        clone-channels=<n>: take 1-channel image, output n-channel image, with each channel a copy of the input" << std::endl;
    oss << "        merge=<n>: split an image into n horizontal bands of equal height and merge them into an n-channel image (the number of rows must be a multiple of n)" << std::endl;
    oss << "        split: split n-channel image into a nx1 grey-scale image" << std::endl;
    oss << "        shuffle=<list>; re-shuffle input channels, e.g., shuffle=r,b,g - swap channels 1 and 2; channels are described by symbolic names 'r', 'g', 'b', and 'a'," << std::endl;
    oss << "            where 'r' is always channel[0], 'b' is channel[1], etc.; if a field is left empty, the corresponding channel is copied verbatim from the input" << std::endl;
    oss << "            more examples: shuffle=,b - drop channel b, leave 2 channels; shuffle=, - drop channel b; shuffle=r,g,r,g - duplicate r and g, drop b" << std::endl;
    oss << std::endl;
    oss << "    operations on \"forked\" image stream:" << std::endl;
    oss << "        semantics: take input image, apply some filters to it, then apply the operation between the input and resulting image" << std::endl;
    oss << "        'double-forked' operations are not supported, e.g., you cannot use mask inside an arithmetic operation or vise versa" << std::endl;
    oss << "        usage: <operation>=<filters>" << std::endl;
    oss << "            <filters>: any sequence of cv-cat filters that outputs a single-channel image of the same dimensions as the images on stdin" << std::endl;
    oss << "                the separator between the filters is '|'" << std::endl;
    oss << "                the equal sign for a filter is ':'" << std::endl;
    oss << "            e.g: cat images.bin | cv-mat subtract=\"average|threshold:0.5\"" << std::endl;
    oss << std::endl;
    oss << "        arithmetic operations" << std::endl;
    oss << "            add=<filters>: forked image is pixelwise added to the input image, see cv::add()" << std::endl;
    oss << "            divide=<filters>: forked image is pixelwise divided to the input image, see cv::divide()" << std::endl;
    oss << "            multiply=<filters>: forked image is pixelwise multiplied to the input image, see cv::multiply()" << std::endl;
    oss << "            subtract=<filters>: forked image is pixelwise subtract to the input image, see cv::subtract()" << std::endl;
    oss << "            absdiff=<filters>: forked image is pixelwise absolute difference between images, see cv::absdiff()" << std::endl;
    oss << "                examples:" << std::endl;
    oss << "                    multiply operation with accumulated and threshold sub filters" << std::endl;
    oss << "                        cat images.bin | cv-cat \"multiply=accumulated:average,5|threshold:0.5,1.0\" >results.bin" << std::endl;
    oss << "                    scaling input images by a mask file, input type:  3ub, mask type: 3f" << std::endl;
    oss << "                        cat images.bin | cv-cat \"multiply=load:mask.bin\" >results.bin" << std::endl;
    oss << "            forked arithmetic operations can use bitwise combinations of filters" << std::endl;
    oss << std::endl;
    oss << "        bitwise operations on filters:" << std::endl;
    oss << "            bitwise=<filter1> and <filter2>: apply bitwise 'and' of <filter1> and <filter2> outputs" << std::endl;
    oss << "            bitwise=<filter1> or <filter2>: apply bitwise 'or'" << std::endl;
    oss << "            bitwise=<filter1> xor <filter2>: apply bitwise 'xor'" << std::endl;
    oss << "            bitwise=not <filter1>: apply bitwise 'not' (complement, tilde in C++)" << std::endl;
    oss << "            bitwise=(( not <filter1> ) and <filter2>) or <filter3>: apply the given logical expression of 3 masks" << std::endl;
    oss << "            note: standard precedence rules apply; use brackets to explicitly denote precedence" << std::endl;
    oss << std::endl;
    oss << "        mask=<filters>: apply mask to image (see cv::copyTo for details)" << std::endl;
    oss << "            examples" << std::endl;
    oss << "                apply a constant mask from a file" << std::endl;
    oss << "                    cat images.bin | cv-cat 'mask=load:mask.bin' > masked.bin" << std::endl;
    oss << "                    cat images.bin | cv-cat 'mask=load:mask.png' > masked.bin" << std::endl;
    oss << "                extract pixels brighter than 100" << std::endl;
    oss << "                    cat images.bin | cv-cat 'mask=convert-color:BGR,GRAY|threshold=otsu,100' > masked.bin" << std::endl;
    oss << "            masks can use bitwise operations, e.g." << std::endl;
    oss << "                cv-cat \"mask=ratio:(r + b - g)/( 1 + r + b )|convert-to:ub|threshold:4 xor ratio:2./(1.5e1 - g + r)|convert-to:ub|threshold:5\"" << std::endl;
    oss << std::endl;
    oss << "        tee=<filters>: run a forked pipeline; pass the image in the main pipeline unchanged" << std::endl;
    oss << "            examples" << std::endl;
    oss << "                view modified image" << std::endl;
    oss << "                    cat images.bin | cv-cat 'tee=resize:0.25|timestamp|view;...'" << std::endl;
    oss << std::endl;
    oss << "        it is the user responsibility to convert data to integer format before applying bitwise combinations" << std::endl;
    oss << std::endl;
    oss << "    operations on subsets of columns, rows, or channels" << std::endl;
    oss << "        bands-to-cols=x,w[,x,w][,method:<method-name>,output-depth:<depth>]; take a number of columns (bands) from the input, process together by method," << std::endl;
    oss << "            write into the output, one band (a range of columns) reduced into one column; supported methods: average (default), sum, min, max; the sum method" << std::endl;
    oss << "            requires the explicit output-depth parameter (one of i,f,d or CV_32S, CV_32F, CV_64F) if the input has low depth" << std::endl;
    oss << "            examples: \"bands-to-cols=12,23,50,30,45,60,100,1,method:average\"; output an image of 4 columns containing the average" << std::endl;
    oss << "                          of columns 12-34 (i.e., 12 + 23 - 1), 50-79, 45-104 (overlap is OK), and 100 (width is 1) from the original image" << std::endl;
    oss << "                      \"bands-to-cols=12,23,50,30,45,60,100,1,method:sum,output-depth:d\"; same bands but output an image of 4 columns containing the sum" << std::endl;
    oss << "                          of the columns data from the original image; use CV_64F depth (double) as the output format" << std::endl;
    oss << "        bands-to-rows=x,w[,x,w][,method:<method-name>,output-depth:<depth>]; same as bands-to-cols but operate on rows of input instead of columns" << std::endl;
    oss << std::endl;
    oss << "        channels-to-cols; opposite to cols-to-channels; unwrap all channels as columns" << std::endl;
    oss << "            example: \"channels-to-cols\" over a 3-channel image: RGB channels of column 0 become columns 0 (single-channel), 1, and 2, RGB channels" << std::endl;
    oss << "            of column 1 become columns 3,4,5, and so on" << std::endl;
    oss << "        channels-to-rows; same as channels-to-cols but operates over rows" << std::endl;
    oss << std::endl;
    oss << "        cols-to-channels=i,j,k[,pad:value,repeat:step]; opposite to channels-to-cols; stores the listed columns as channels in the output file" << std::endl;
    oss << "            input shall be a single-channel stream; up to 4 channels are supported; if 1, 3, or 4 columns are specified, the output would have 1, 3, or 4 channels respectively" << std::endl;
    oss << "            in case of 2 columns, a third empty (zero) channel is added; use the \"pad:value\" option to specify the fill value other then zero" << std::endl;
    oss << "            the repeat option applies the transformation periodically, first for the specified columns, then for columns incremented by one step, and so on; see the examples" << std::endl;
    oss << "            examples: \"cols-to-channels=6,4,pad:128\"; put column 6 into the R channel, column 4 into the G channel, and fill the B channel with 128" << std::endl;
    oss << "                      \"cols-to-channels=0,1,2,repeat:3\"; store columns 0,1,2 as RGB channels of column 0 of the output file, then columns 3,4,5 as RGB" << std::endl;
    oss << "                      channels of column 1 of the output file, etc.; conversion stops when all input column indices exceed the image width" << std::endl;
    oss << "                      if one of the input columns exceed the image width, the respective output channel is filled with zeros (or the padding value)" << std::endl;
    oss << std::endl;
    oss << "        crop-cols=<x>,<w>[,<x>,<w>,...]: output an image consisting of (multiple) columns starting at x with width w" << std::endl;
    oss << "            examples: \"crop-cols=2,10,12,10\"; output an image of width 20 taking 2 width-10 columns starting at 2 and 12" << std::endl;
    oss << "                      \"crop-cols=2,1,12,10\"; width is 1, the output image has width 11" << std::endl;
    oss << "        crop-rows=<y>,<h>[,<y>,<h>,...]: output an image consisting of (multiple) row blocks starting at y with height h" << std::endl;
    oss << "            examples: \"crop-rows=5,10,25,5\"; output an image of height 15 taking 2 row blocks starting at 5 and 25 and with heights 10 and 5, respectively" << std::endl;
    oss << "                      \"crop-rows=5,1,25,1,15,1\"; block height is 1, block starts can be out of order" << std::endl;
    oss << std::endl;
    oss << "        rows-to-channels=1,4,5[,pad:value,repeat:step]; same as cols-to-channels but operates on rows" << std::endl;
    oss << std::endl;
    oss << "    basic drawing on images" << std::endl;
    oss << "        cross[=<x>,<y>]: draw cross-hair at x,y; default: at image center" << std::endl;
    oss << "        circle=<x>,<y>,<radius>[,<r>,<g>,<b>,<thickness>,<line_type>,<shift>]: draw circle; see cv::circle for details on parameters and defaults" << std::endl;
    oss << "        rectangle,box=<x>,<y>,<x>,<y>[,<r>,<g>,<b>,<thickness>,<line_type>,<shift>]: draw rectangle; see cv::rectangle for details on parameters and defaults" << std::endl;
    oss << std::endl;
    oss << "    morphology operations" << std::endl;
    oss << "        blackhat[=<parameters>]; apply black-hat operation with the given parameters" << std::endl;
    oss << "        close[=<parameters>], closing[=<parameters>]; apply closing with the given parameters" << std::endl;
    oss << "        dilate[=<parameters>], dilation[=<parameters>]; apply dilation with the given parameters" << std::endl;
    oss << "        erode[=<parameters>], erosion[=<parameters>]; apply erosion with the given parameters" << std::endl;
    oss << "        gradient[=<parameters>]; apply morphological gradient with the given parameters" << std::endl;
    oss << "        open[=<parameters>], opening[=<parameters>]; apply opening with the given parameters" << std::endl;
    oss << "        tophat[=<parameters>]; apply top-hat operation with the given parameters" << std::endl;
    oss << "        skeleton[=<parameters>], thinning[=<parameters>]; apply skeletonization (thinning) with the given parameters" << std::endl;
    oss << std::endl;
    oss << "            <parameters> for all the above operations have the same syntax; erode as an example is shown below:" << std::endl;
    oss << "                erode=rectangle,<size/x>,<size/y>[,<anchor/x>,<anchor/y>][,iterations]; apply erosion with a rectangular structuring element" << std::endl;
    oss << "                erode=square,<size/x>[,<anchor/x>][,iterations]; apply erosion with a square structuring element of custom size" << std::endl;
    oss << "                erode=ellipse,<size/x>,<size/y>[,<anchor/x>,<anchor/y>][,iterations]; apply erosion with an elliptic structuring element" << std::endl;
    oss << "                erode=circle,<size/x>[,<anchor/x>][,iterations]; apply erosion with a circular structuring element" << std::endl;
    oss << "                erode=cross,<size/x>,<size/y>[,<anchor/x>,<anchor/y>][,iterations]; apply erosion with a circular structuring element" << std::endl;
    oss << "                    note that the structuring element shall usually be symmetric, and therefore, size/s,size/y shall be odd" << std::endl;
    oss << "                    any of the parameters after the shape name can be omitted (left as an empty csv field) to use the defaults:" << std::endl;
    oss << "                        - size/x = 3:" << std::endl;
    oss << "                        - size/y = size/x:" << std::endl;
    oss << "                        - anchor/x = center in x" << std::endl;
    oss << "                        - anchor/y = anchor/x" << std::endl;
    oss << "                        - iterations = 1" << std::endl;
    oss << "                    anchor value of -1 is interpreted as the center of the element" << std::endl;
    oss << "                    alternatively, if only 2 parameters are given for rectangle, ellipse, and cross, they are interpreted as size/x," << std::endl;
    oss << "                    size/y, with the anchor set to default; similarly, if a single parameter is given for square and circle," << std::endl;
    oss << "                    it is used to set size with default anchor" << std::endl;
    oss << std::endl;
    oss << "            examples: \"erode=rectangle,5,3,,\"; apply erosion with a 5x3 rectangle anchored at the center" << std::endl;
    oss << "                      \"erode=rectangle,5,3,,,2\"; same as above but apply 2 iterations of erode" << std::endl;
    oss << "                      \"close\"; apply closing with a 3x3 square structuring element anchored at the center (default)" << std::endl;
    oss << "                      \"tophat=rectangle,11,,3,3\"; apply tophat with a 11x11 square and custom off-center anchor" << std::endl;
    oss << "                      \"dilate=cross,7,,,\"; apply dilation with a 7x7 cross anchored at the center" << std::endl;
    oss << "                      \"open=circle,7\"; apply opening with a radius 7 circle anchored at the center (note single parameter)" << std::endl;
    oss << "                      \"open=rectangle,7,3\"; apply opening with a 7x3 rectangle anchored at the center (note only two parameters)" << std::endl;
    oss << std::endl;
    oss << "    multiple-channel operations" << std::endl;
    oss << "        linear-combination=<a1>r + <a2>g + ... + <ac>: output a grey-scale image that is a linear combination of input channels with given coefficients and optional offset" << std::endl;
    oss << "            example: \"linear-combination=-r+2g-b\", highlights the green channel" << std::endl;
    oss << "            naming conventions are the same as for the ratio operation; use '--help filters::linear-combination' for more examples and a detailed syntax explanation" << std::endl;
    oss << "        linear-combination=expr0,expr1...: output multi-channel image where each output channel is calculated as a linear combination of input channels" << std::endl;
    oss << "            according to the provided comma-separated expressions (expr0 goes to channel 0, expr1 to channel 1, etc.); if any of the expressions is omitted" << std::endl;
    oss << "            (empty), it is interpreted as a pass-through of the respective channel" << std::endl;
    oss << "        ratio=(<a1>r + <a2>g + ... + <ac>)/(<b1>r + <b2>g + ... + <bc>): output a grey-scale image that is a ratio of linear combinations of input channels" << std::endl;
    oss << "            with given coefficients and offsets; see below for examples, use '--help filters::ratio' for the detailed explanation of the syntax and examples" << std::endl;
    oss << "            the naming convention does not depend on the actual image channels: 'r' in the ratio expression is always interpreted as channel[0]," << std::endl;
    oss << "            'g' as channel[1], etc.; in particular, grey-scaled images have a single channel that shall be referred to as 'r', e.g., ratio=r / ( r + 1 )" << std::endl;
    oss << "        ratio=expr0,expr1...: output a multi-channel image where each output channel is calculated as a ratio of linear combinations of input channels" << std::endl;
    oss << "            given by the provided comma-separated expressions (expr0 goes to channel 0, expr1 to channel 1, etc.; if any of the expressions is empty" << std::endl;
    oss << "            (omitted), it is interpreted as a pass-through (see examples)" << std::endl;
    oss << std::endl;
    oss << "            examples: \"ratio=( r + g + b ) / ( 1 + a )\"; output a grey-scale image equal to the sum of the first 3 channels" << std::endl;
    oss << "                          divided by the offset 4th channel" << std::endl;
    oss << "                      \"ratio=( r - b ) / ( r + b )\"; output normalized difference of channels 'r' and 'g'" << std::endl;
    oss << "                      \"ratio=,( r - b ) / ( r + b )\"; output a 2-channel image, output channel 0 (r) is copied from the input as is, while output channel 1 (g)" << std::endl;
    oss << "                          contains normalized difference of channels 'r' and 'g'" << std::endl;
    oss << std::endl;
    oss << "        output of the ratio and linear-combination operations has floating point (CV_32F) precision unless the input is already in doubles (if so, precision is unchanged)" << std::endl;
    oss << std::endl;
    oss << "    cv::Mat image operations" << std::endl;
    oss << "        simple-blob[=<parameters>]: wraps cv::SimpleBlobDetector, outputs as csv key points timestamped by image timestamp" << std::endl;
    oss << "            <parameters>" << std::endl;
    oss << "                output-binary: output key points as binary" << std::endl;
    oss << "                output-fields: print output fields on stdout and exit" << std::endl;
    oss << "                output-format: print binary output format on stdout and exit" << std::endl;
    oss << "                output-default-params,default-params: print default simple blob detector parameters to stdout and exit" << std::endl;
    oss << "                <filename>[:<path>]: simple blob detector params config file and optionally cv file node path in it" << std::endl;
    oss << "                                     e.g: my-params.txt:blob" << std::endl;
    oss << "                                     if not present, defaults will be used" << std::endl;
    oss << std::endl;
    oss << snark::imaging::vegetation::impl::filters< H >::usage() << std::endl;
    return oss.str();
}

template < typename H >
const std::string& impl::filters< H >::usage( const std::string & operation )
{
    if( operation.empty() ) { static const std::string s = usage_impl_< H >(); return s; }
    if ( operation == "ratio" || operation == "linear-combination" ) { static const std::string s = snark::cv_mat::ratios::ratio::describe_syntax(); return s; }
    static std::string s = "filters: no specific help is available for the '" + operation + "' operation";
    return s;
}

} } // namespace snark{ namespace cv_mat {

namespace comma { namespace visiting {

template <> struct traits< map_input_t >
{
    template< typename K, typename V > static void visit( const K&, map_input_t& t, V& v )
    {
        v.apply( "key", t.key );
        v.apply( "value", t.value );
    }
    template< typename K, typename V > static void visit( const K&, const map_input_t& t, V& v )
    {
        v.apply( "key", t.key );
        v.apply( "value", t.value );
    }
};

template < >
struct traits< typename snark::cv_mat::log_impl_< boost::posix_time::ptime >::logger::indexer >
{
    template < typename K, typename V > static void visit( const K&, const typename snark::cv_mat::log_impl_< boost::posix_time::ptime >::logger::indexer& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "file", t.file );
        v.apply( "offset", t.offset );
    }

    template < typename K, typename V > static void visit( const K&, typename snark::cv_mat::log_impl_< boost::posix_time::ptime >::logger::indexer& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "file", t.file );
        v.apply( "offset", t.offset );
    }
};

template < >
struct traits< typename snark::cv_mat::log_impl_< snark::cv_mat::header_type >::logger::indexer >
{
    template < typename K, typename V > static void visit( const K&, const typename snark::cv_mat::log_impl_< snark::cv_mat::header_type >::logger::indexer& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "file", t.file );
        v.apply( "offset", t.offset );
    }

    template < typename K, typename V > static void visit( const K&, typename snark::cv_mat::log_impl_< snark::cv_mat::header_type >::logger::indexer& t, V& v )
    {
        v.apply( "t", t.t );
        v.apply( "file", t.file );
        v.apply( "offset", t.offset );
    }
};
//TODO traits for template type needed

} } // namespace comma { namespace visiting {

template class snark::cv_mat::impl::filters< boost::posix_time::ptime >;
template class snark::cv_mat::impl::filters< snark::cv_mat::header_type >;
