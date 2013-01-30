// This file is part of snark, a generic and flexible library
// for robotics research.
//
// Copyright (C) 2011 The University of Sydney
//
// snark is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// snark is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with snark. If not, see <http://www.gnu.org/licenses/>.

#include <snark/sensors/dc1394/dc1394.h>
#include <snark/timing/time.h>
#include <comma/base/exception.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace snark { namespace camera {

/// default constructor    
dc1394::config::config():
    output( BGR ),
    video_mode( DC1394_VIDEO_MODE_640x480_YUV422 ),
    operation_mode( DC1394_OPERATION_MODE_LEGACY ),
    iso_speed( DC1394_ISO_SPEED_400 ),
    frame_rate( DC1394_FRAMERATE_MAX ),
    relative_shutter( 0 ),
    relative_gain( 0 ),
    shutter( 0 ),
    gain( 0 ),
    exposure( 0 ),
    guid( 0 )
{
    
}

/// get OpenCV type from color coding
int dc1394::config::type() const
{
    if( output == RGB || output == BGR )
    {
        return CV_8UC3;
    }
    else
    {
        return CV_8UC1;
    }
}

/// constructor
/// @param config camera config
dc1394::dc1394( const snark::camera::dc1394::config& config, unsigned int format7_width, unsigned int format7_height, unsigned int format7_size ):
    m_config( config ),
    m_epoch( timing::epoch ),
    m_format7_width( format7_height ),
    m_format7_height( format7_height ),
    m_format7_size( format7_size )
{
    memset( &m_output_frame, 0, sizeof( m_output_frame ) );
    m_output_frame.color_coding = DC1394_COLOR_CODING_RGB8;
    
    init_camera();

    /* First check if relative shutter has been set, if not check absolute
     * shutter. If neither of those are set, use the exposure */
    if( m_config.relative_shutter != 0 )
    {
        set_relative_shutter_gain( m_config.relative_shutter, m_config.relative_gain );
    }
    else if( std::fabs( m_config.shutter ) > 1e-5 )
    {
        set_absolute_shutter_gain( m_config.shutter, m_config.gain );
    }
    else {
        set_exposure(m_config.exposure);
    }

    if( m_config.output == config::Raw )
    {
        dc1394color_coding_t colorCoding;
        dc1394_get_color_coding_from_video_mode( m_camera, m_config.video_mode, &colorCoding );
        unsigned int numBits;
        dc1394_get_color_coding_bit_size( colorCoding, &numBits );
        if( numBits != 8 )
        {
            COMMA_THROW( comma::exception, "only 8 bits per pixel supported in raw output mode, color coding has " << numBits << " bits per pixel " );
        }
    }

    dc1394video_modes_t video_modes;
    if ( dc1394_video_get_supported_modes( m_camera, &video_modes ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "cannot get video modes" );
    }    

    dc1394_get_image_size_from_video_mode( m_camera, m_config.video_mode, &m_width, &m_height );
        
    if ( !dc1394_is_video_mode_scalable( m_config.video_mode ) )
    {
        setup_camera();
    }
    else
    {
        setup_camera_format7();
    }

    if ( dc1394_capture_setup( m_camera, 4, DC1394_CAPTURE_FLAGS_DEFAULT) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not setup the camera" );
    }

    m_image = cv::Mat( m_height, m_width, m_config.type() );
    
    if ( dc1394_video_set_transmission( m_camera, DC1394_ON ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not start the camera iso transmission" );
    }

    dc1394switch_t status = DC1394_OFF;
    unsigned int i = 0;
    while( status != DC1394_OFF )
    {
        ::usleep( 10000 );
        if ( dc1394_video_get_transmission( m_camera, &status ) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "could not get the camera transmission status" );
        }
        if( i == 10 )
        {
            COMMA_THROW( comma::exception, "camera does not start" );
        }
        i++;
    }
}

/// destructor
dc1394::~dc1394()
{
    dc1394_capture_stop( m_camera );
    dc1394_video_set_transmission( m_camera, DC1394_OFF );
    dc1394_camera_free( m_camera );
}

/// acquire a frame from the camera
const cv::Mat& dc1394::read()
{
    
    if ( dc1394_capture_dequeue( m_camera, DC1394_CAPTURE_POLICY_WAIT, &m_frame ) < 0 )
    {
        COMMA_THROW( comma::exception, " no frame in buffer " );
    }
    if ( dc1394_capture_is_frame_corrupt ( m_camera, m_frame ) )
    {
        COMMA_THROW( comma::exception, "frame corrupted" );
    }

    if( ( m_config.output == config::Raw ) || ( m_output_frame.color_coding == m_frame->color_coding ) )
    {
        memcpy( m_image.data, m_frame->image, m_frame->image_bytes );
    }
    else
    {
        if( dc1394_convert_frames( m_frame, &m_output_frame ) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "error converting frame, conversion probably not supported" );
        }
        memcpy( m_image.data, m_output_frame.image, m_output_frame.image_bytes );
        if( m_config.output == config::BGR )
        {
            cv::cvtColor( m_image, m_image, CV_RGB2BGR );
        }
    }
    //Get the time from the frame timestamp
    m_time = m_epoch + boost::posix_time::microseconds( m_frame->timestamp );
    dc1394_capture_enqueue( m_camera, m_frame ); // release the frame
    
    return m_image;
}

bool dc1394::poll()
{
    m_select.wait( m_frame_duration );
    return m_select.read().ready( m_fd ); 
}

void dc1394::list_cameras()
{
    dc1394_t* dc1394 = dc1394_new();
    dc1394camera_list_t * list;

    if ( dc1394_camera_enumerate( dc1394, &list) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "error scanning for cameras" );
    }

    for( unsigned int i = 0; i < list->num; i++ )
    {
        std::cerr << "camera found: " << list->ids[i].guid << std::endl;
    }
}

/// initialize dc1394 camera structure
void dc1394::init_camera()
{
    dc1394_t* dc1394 = dc1394_new();
    dc1394camera_list_t * list;

    if ( dc1394_camera_enumerate( dc1394, &list) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "error scanning for cameras" );
    }

    if ( list->num == 0 )
    {
        COMMA_THROW( comma::exception, "no camera found" );
    }

    unsigned int index = list->num - 1;
    if ( m_config.guid != 0 )
    {
        // look for specified camera on the bus
        bool found = false;
        unsigned int i = 0;
        while( i < list->num && !found )
        {
            if ( list->ids[i].guid == m_config.guid )
            {
                found = true;
                index = i;
            }
            i++;
        }
        if ( !found )
        {
            COMMA_THROW( comma::exception, "camera with guid " << m_config.guid << " not found " );
        }
    }

    m_camera = dc1394_camera_new( dc1394, list->ids[ index ].guid );
    if ( m_camera == NULL )
    {
        COMMA_THROW( comma::exception, "failed to initialize camera with guid " << list->ids[ m_config.guid ].guid );
    }

    dc1394_camera_free_list( list );
}

/// setup the camera capture for non-format7 modes
void dc1394::setup_camera()
{
    if( dc1394_video_set_operation_mode( m_camera, m_config.operation_mode ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set operation mode" );
    }
    if( dc1394_video_set_iso_speed( m_camera, m_config.iso_speed ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set iso speed" );
    }

    dc1394framerate_t framerate;
    dc1394framerates_t framerates;
    if ( dc1394_video_get_supported_framerates( m_camera, m_config.video_mode, &framerates ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "error getting framerate" );
    }
    if( framerates.num == 0 )
    {
        COMMA_THROW( comma::exception, "could not get framerate list" );
    }
    if( m_config.frame_rate == DC1394_FRAMERATE_MAX )
    {
        framerate = framerates.framerates[ framerates.num-1 ];
    }
    else
    {
        unsigned int i = 0;
        while( i < framerates.num && framerates.framerates[ i ] != m_config.frame_rate )
        {
            i++;
        }
        if( i == framerates.num )
        {
            COMMA_THROW( comma::exception, "invalid framerate" );
        }
        else
        {
            framerate = framerates.framerates[ i ];
        }
    }

    if ( dc1394_video_set_mode( m_camera, m_config.video_mode ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set video mode" );
    }

    if ( dc1394_video_set_framerate( m_camera, framerate ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set framerate" );
    }
}

/// setup camera capture for format7 modes
void dc1394::setup_camera_format7()
{    
    if (dc1394_video_set_operation_mode( m_camera, m_config.operation_mode ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set iso mode" );
    }

    if ( dc1394_video_set_iso_speed( m_camera, m_config.iso_speed ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set iso speed" );
    }

    if( dc1394_video_set_mode( m_camera, m_config.video_mode ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set video mode" );
    }
    
    if( m_format7_width != 0 )
    {
        if (dc1394_format7_set_image_size( m_camera, m_config.video_mode, m_format7_width, m_format7_height ) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "could not set image size" );
        }
        if ( dc1394_format7_get_image_size( m_camera, m_config.video_mode, &m_width, &m_height ) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "could not get image size" );
        }
        std::cerr << " size " << m_width << " , " << m_height << std::endl;
    }
    else
    {    
        if( dc1394_format7_get_max_image_size( m_camera, m_config.video_mode, &m_width, &m_height) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "could not get max image size" );
        }
        if (dc1394_format7_set_image_size( m_camera, m_config.video_mode, m_width, m_height) != DC1394_SUCCESS )
        {
            COMMA_THROW( comma::exception, "could not set image size" );
        }
    }

    unsigned int minBytes;
    unsigned int maxBytes;
    if ( dc1394_format7_get_packet_parameters( m_camera, m_config.video_mode, &minBytes, &maxBytes) != DC1394_SUCCESS)
    {
        COMMA_THROW( comma::exception, "could not get packet size" );
    }
    dc1394color_coding_t color;
    dc1394_format7_get_color_coding( m_camera, m_config.video_mode, &color );
    // HACK packet sizes higher than 8160 don't seem to work with the ladybug
    // setting it to 8160 by hand works but the frame rate is only about 5 fps
    // TODO fix to be able to set DC1394_QUERY_FROM_CAMERA again and maybe higher framerate
    // this is fine for other cameras with lower packet size ( e.g. bumblebee ), as it will be set to the next possible value
    if ( dc1394_format7_set_roi( m_camera, m_config.video_mode, color, m_format7_size, 0, 0, m_width, m_height ) != DC1394_SUCCESS )
    {
        COMMA_THROW( comma::exception, "could not set roi" );
    }
}

/// set absolute shutter and gain
void dc1394::set_absolute_shutter_gain( float shutter, float gain )
{
    dc1394error_t err;

    /* 1. Turn exposure off */
    err = dc1394_feature_set_power(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF);
    DC1394_ERR(err, "Failed to set exposure power");

    /* 2. Set gain to manual mode (and ideally minimum gain) */
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL);
    DC1394_ERR(err, "Failed to set gain mode");
    err = dc1394_feature_set_absolute_control(m_camera, DC1394_FEATURE_GAIN, DC1394_ON);
    DC1394_ERR(err, "Failed to set gain absolute control mode");
    err = dc1394_feature_set_absolute_value(m_camera, DC1394_FEATURE_GAIN, gain);
    DC1394_ERR(err, "Failed to set absolute gain value");

    /* 3. Set the shutter to manual mode, and then the requested value. */
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL);
    DC1394_ERR(err, "Failed to set shutter mode");
    err = dc1394_feature_set_absolute_control(m_camera, DC1394_FEATURE_SHUTTER, DC1394_ON);
    DC1394_ERR(err, "Failed to set shutter absolute control mode");
    err = dc1394_feature_set_absolute_value(m_camera, DC1394_FEATURE_SHUTTER, shutter);
    DC1394_ERR(err, "Failed to set absolute shutter value");
}

/// set relative shutter and gain
void dc1394::set_relative_shutter_gain( unsigned int shutter, unsigned int gain )
{    
    dc1394error_t err;

    /* 1. Turn exposure off */
    err = dc1394_feature_set_power(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF);
    DC1394_ERR(err, "Failed to set exposure power");

    /* 2. Set gain to manual mode (and ideally minimum gain) */
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL);
    DC1394_ERR(err, "Failed to set gain mode");
    err = dc1394_feature_set_absolute_control(m_camera, DC1394_FEATURE_GAIN, DC1394_OFF);
    DC1394_ERR(err, "Failed to set gain absolute control mode");
    err = dc1394_feature_set_value(m_camera, DC1394_FEATURE_GAIN, gain);
    DC1394_ERR(err, "Failed to set gain value");

    /* 3. Set the shutter to manual mode, and then the requested value. */
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL);
    DC1394_ERR(err, "Failed to set shutter mode");
    err = dc1394_feature_set_absolute_control(m_camera, DC1394_FEATURE_SHUTTER, DC1394_OFF);
    DC1394_ERR(err, "Failed to set shutter absolute control mode");
    err = dc1394_feature_set_value(m_camera, DC1394_FEATURE_SHUTTER, shutter);
    DC1394_ERR(err, "Failed to set shutter value");
}

/// set exposure controls 
void dc1394::set_exposure( unsigned int exposure )
{
    dc1394error_t err;

    /* Turn shutter/gain to auto */
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_AUTO);
    DC1394_ERR(err, "Failed to set gain mode");
    err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_AUTO);
    DC1394_ERR(err, "Failed to set shutter mode");

    /* Turn auto exposure on */
    err = dc1394_feature_set_power(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_ON);
    DC1394_ERR(err, "Failed to set exposure power");
   
    if(exposure > 0) 
    {
        /* If a particular exposure has been selected, choose it 
         * only capable of relative mode for now */
        err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_MANUAL);
        DC1394_ERR(err, "Failed to set exposure mode");
        err = dc1394_feature_set_absolute_control(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_OFF);
        DC1394_ERR(err, "Failed to set exposure absolute control mode");
        err = dc1394_feature_set_value(m_camera, DC1394_FEATURE_EXPOSURE, exposure);
        DC1394_ERR(err, "Failed to set exposure value");
    }
    else
    {
        /* Otherwise, auto exposure */
        err = dc1394_feature_set_mode(m_camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_AUTO);
        DC1394_ERR(err, "Failed to set exposure mode");
    }  
}


/// list all the dc1394 features of the camera
void dc1394::list_attributes()
{
    dc1394featureset_t featureset;
    dc1394_feature_get_all(m_camera, &featureset);
    dc1394_feature_print_all(&featureset, stderr);
}


} } // namespace snark { namespace camera {
