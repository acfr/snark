// This file is part of rur, a generic and flexible library
// for robotics research.
//
// Copyright (C) 2011 The University of Sydney
//
// rur is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// rur is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License
// for more details.
//
// You should have received a copy of the GNU General Public
// License along with rur. If not, see <http://www.gnu.org/licenses/>.

#include "./VertexBuffer.h"


namespace rur { namespace graphics { namespace Qt3D {

VertexBuffer::VertexBuffer ( std::size_t size ):
    m_readIndex( 0 ),
    m_writeIndex( 0 ),
    m_readSize( 0 ),
    m_writeSize( 0 ),
    m_bufferSize( size ),
    m_block( 0 )
{
    m_points.resize( 2 * size );
    m_color.resize( 2 * size );
}

void VertexBuffer::addVertex ( const QVector3D& point, const QColor4ub& color, unsigned int block )
{
    if( block != m_block )
    {
        m_block = block;

        m_writeIndex += m_bufferSize;
        m_writeIndex %= 2 * m_bufferSize;
        if( m_readIndex == m_writeIndex )
        {
            m_readIndex+= m_bufferSize;
            m_readIndex %= 2 * m_bufferSize;
            m_readSize = m_writeSize;
        }
        m_writeSize = 0;
    }
    m_points[ m_writeIndex + m_writeSize ] = point;
    m_color[ m_writeIndex + m_writeSize ] = color;
    m_writeSize++;
    if( block == 0 )
    {
        m_readSize++;
    }
    if( m_writeSize > m_bufferSize )
    {
        m_writeSize = 0;
        m_readSize = m_bufferSize;
    }
}

const QVector3DArray& VertexBuffer::points() const
{
    return m_points;
}

const QArray< QColor4ub >& VertexBuffer::color() const
{
    return m_color;
}

const unsigned int VertexBuffer::size() const
{
    return m_readSize;
}

const unsigned int VertexBuffer::index() const
{
    return m_readIndex;
}


    
} } } // namespace rur { namespace graphics { namespace View {
