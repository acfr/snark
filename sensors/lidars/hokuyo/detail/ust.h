#pragma once

#include "device.h"
#include "../message.h"
#include "../output.h"
#include <comma/base/exception.h>
#include "../sensors.h"
#include <chrono>
#include <boost/asio/steady_timer.hpp>

namespace hok = snark::hokuyo;
namespace ip = boost::asio::ip;

namespace snark { namespace hokuyo {

template < int STEPS >
struct ust_device:public laser_device
{
    typedef data_point output_t;
    struct data_t
    {
        double distance;
        comma::uint32 intensity;
        double bearing;
    };
    //response data
    typename di_data< STEPS >::rays rays;
    request_md me;
    int start_step;
    std::string address;
    bool permissive;
    ust_device( const bool permissive ) : permissive( permissive ) {}
    virtual void init(comma::command_line_options options)
    {
        //get args
        address=options.value< std::string >( "--tcp,--laser" );
    }
    virtual boost::shared_ptr< stream_base > connect();
    virtual void setup(stream_base& ios) { } //nothing to do
    virtual void reboot(stream_base& ios);
    void request_scan(stream_base& ios, int start_step, int end_step, int num_of_scans);
    bool receive_response(stream_base& ios);
    data_t get_data(int scan);
    bool is_valid(data_t& data);
    output_t convert(data_t& data);
    int get_steps() const { return STEPS; }
};

static boost::optional< sequence_string > prev_message_id;

template < int STEPS >
void ust_device<STEPS>::request_scan(stream_base& iostream, int start, int end_step, int num_of_scans)
{
    start_step= (start==-1) ? 0 : start;
    if( start_step != 0 && start_step >= int( hok::ust_10lx::step_max - STEPS ) ) { COMMA_THROW( comma::exception, "start step is too high" ); }
    if(end_step!=0&&end_step!=STEPS){COMMA_THROW( comma::exception, "--end-step is not supported for UST implementation" );}

    me=request_md( true );
    me.header.start_step = start_step;
    me.header.end_step = start_step + STEPS-1;
    me.num_of_scans = num_of_scans;
    
    iostream.write( me.data(), request_md::size );
    iostream.flush();

    reply_md state;
    iostream.read( state.data(), reply_md::size );
    // possible scanner bug: occasionally, after completing a request the scanner resends the response message
    // after sending a new request, if a response message for the previous request is received then ignore and re-read
    if( permissive && prev_message_id && state.request.message_id == *prev_message_id )
    {
        //std::cerr << "got response to previous message: " << state.request.message_id.str() << std::endl;
        iostream.read( state.data(), reply_md::size );
    }
    if( state.request.message_id != me.message_id )
    { 
        COMMA_THROW( comma::exception, "message id mismatch for ME status reply, read: " << iostream.bytes_read()
                                        << " bytes, expected: " << me.message_id.str() << " got: " << state.request.message_id.str() );
    }
    if( state.status.status() != 0 ) { COMMA_THROW( comma::exception, "status reply to ME request is not success: " << state.status.status() ); }
    prev_message_id = me.message_id;
}

template < int STEPS >
bool ust_device<STEPS>::receive_response(stream_base& iostream)
{
    reply_me_data< STEPS > response; // reply with data
    // TODO just read the status response first, or timeout on read()
    // iostream.read( response.data(), reply_me_data< STEPS >::size );
    int status = read( response, iostream );
    if( status != status::data_success ) 
    {
        COMMA_THROW( comma::exception, "failure dectected when reading data, status: " << status );
    }
    if( response.header.request.message_id != me.message_id )
    { 
        COMMA_THROW( comma::exception, "message id mismatch for ME data reply, expected: " << me.message_id.str() << " got: " << response.header.request.message_id.str() ); 
    }
    
    response.encoded.get_values( rays );
    return !(response.header.request.num_of_scans == 0);
}

template < int STEPS >
typename ust_device<STEPS>::data_t ust_device<STEPS>::get_data(int i)
{
    typename ust_device::data_t data;
    data.distance = rays.steps[i].distance();
    data.intensity = rays.steps[i].intensity();
    data.bearing = ust_10lx::step_to_bearing( i + start_step );
    return data;
}

template < int STEPS >
bool ust_device<STEPS>::is_valid(data_t& data)
{
    return !( data.distance == ust_10lx::distance_nan || data.distance <= ust_10lx::distance_min );
}

template < int STEPS >
typename ust_device<STEPS>::output_t ust_device<STEPS>::convert(data_t& data)
{
    data_point point3d;
    point3d.set( data.distance, data.intensity, data.bearing );
    return point3d;
}

/// Connect to the TCP server within the allowed timeout
/// Needed because comma::io::iostream is not available
static bool tcp_connect( const std::string& conn_str, 
                ip::tcp::iostream& io, 
                const boost::posix_time::time_duration& timeout=boost::posix_time::seconds(1)
)
{
    std::vector< std::string > v = comma::split( conn_str, ':' );
    boost::asio::io_service service;
    ip::tcp::resolver resolver( service );
    ip::tcp::resolver::query query( v[0] == "localhost" ? "127.0.0.1" : v[0], v[1] );
    ip::tcp::resolver::iterator it = resolver.resolve( query );
    
    io.expires_from_now( std::chrono::microseconds(timeout.total_microseconds()) );
    io.connect( it->endpoint() );
    
    io.expires_at( boost::asio::steady_timer::time_point::max() );
    
    return io.error().value() == boost::system::errc::success;
} 

template < int STEPS >
boost::shared_ptr<stream_base> ust_device<STEPS>::connect()
{
    //setup communication
    boost::shared_ptr<tcp_stream> tcp( new tcp_stream() );
    if( !tcp_connect( address, tcp->ios ) ) {
        COMMA_THROW( comma::exception, "failed to connect to the hokuyo laser at: " << address );
    }
    return tcp;
}
    
template < int STEPS >
void ust_device<STEPS>::reboot(stream_base& ios)
{
    // it must be sent twice within one second
    ios << "RB\n"; ios.flush();
    ios << "RB\n"; ios.flush();
    sleep( 1 );
}

} }  // namespace snark { namespace hokuyo {
    
