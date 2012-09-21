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

#ifndef RUR_GRAPHICS_APPLICATIONS_VIEWPOINTS_MODEL_READER_H_
#define RUR_GRAPHICS_APPLICATIONS_VIEWPOINTS_MODEL_READER_H_


#include "./Reader.h"
#include "./PlyLoader.h"

class QGLAbstractScene;

namespace rur { namespace graphics { namespace View {

/// display 3d models ( obj or 3ds ), set its position from an input csv stream       
class ModelReader : public Reader
{
    public:
        ModelReader( QGLView& viewer, Csv::Options& options, const std::string& file, bool z_up, coloured* c, const std::string& label );

        void start();
        void update( const Eigen::Vector3d& offset );
        const Eigen::Vector3d& somePoint() const;
        bool readOnce();
        void render( QGLPainter *painter );
        bool empty() const;

    protected:
        boost::scoped_ptr< Csv::InputStream< PointWithId > > m_stream;
        const std::string m_file;
        QGLAbstractScene* m_scene;
        bool m_z_up; // z-axis points up
        boost::optional< PlyLoader > m_plyLoader;
};

} } } // namespace rur { namespace graphics { namespace View {

#endif /*RUR_GRAPHICS_APPLICATIONS_VIEWPOINTS_MODEL_READER_H_*/
