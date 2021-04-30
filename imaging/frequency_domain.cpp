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

#include <opencv2/imgproc/imgproc.hpp>
#include "frequency_domain.h"

namespace snark{ namespace imaging {

/// constructor
/// @param mask window to be applied in the frequencty
frequency_domain::frequency_domain( const cv::Mat& mask ):
    m_mask( mask )
{
    shift( m_mask );
}

/// transform image into frequency domain, apply window and transform back to space domain
cv::Mat frequency_domain::filter( const cv::Mat& image, const cv::Mat& mask )
{
    double maxValue;
    cv::minMaxIdx( image, NULL, &maxValue );
    cv::Mat padded;
    cv::Mat paddedMask;                      //expand input image to optimal size
    int m = cv::getOptimalDFTSize( image.rows );
    int n = cv::getOptimalDFTSize( image.cols ); // on the border add zero values
    cv::copyMakeBorder( image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );
    cv::copyMakeBorder( m_mask, paddedMask, 0, m - m_mask.rows, 0, n - m_mask.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );

    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::merge( planes, 2, m_complexImage );         // Add to the expanded another plane with zeros
    cv::dft( m_complexImage, m_complexImage );            // this way the result may fit in the source matrix
    
    cv::Mat masked;
    m_complexImage.copyTo( masked, paddedMask );
    
    cv::Mat inverse;
    cv::dft( masked, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT );
    cv::Mat abs = cv::abs( inverse );
    if( mask.rows != 0 )
    {
        cv::Mat maskF;// = cv::Mat_<float>( mask );
        cv::Mat paddedMaskF;
        mask.convertTo( maskF, CV_32F, 1.0/255.0 );
        if( abs.cols > maskF.cols || abs.rows > maskF.rows )
        {
            cv::copyMakeBorder( maskF, paddedMaskF, 0, abs.rows - maskF.rows, 0, abs.cols - maskF.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );
        }
        else
        {
            paddedMaskF = maskF;
        }
        cv::multiply( abs, paddedMaskF, abs );
    }
    cv::Mat result;
    cv::normalize( abs, abs, 0, maxValue, cv::NORM_MINMAX ); // scale to get the same max value as the original TODO better scaling ?
    abs.convertTo( result, CV_8U );
    return result;
}

/// get the magnitude of the image in frequency domain
cv::Mat frequency_domain::magnitude() const
{    
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::Mat planes[2];
    cv::split( m_complexImage, planes );                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude( planes[0], planes[1], planes[0] );// planes[0] = magnitude
    cv::Mat magnitudeImage = planes[0];

    magnitudeImage += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log( magnitudeImage, magnitudeImage );

    // crop the spectrum, if it has an odd number of rows or columns
    magnitudeImage = magnitudeImage( cv::Rect( 0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2 ) );
    shift( magnitudeImage );
    
 // transform the matrix with float values into a viewable image form (float between values 0 and 1).
    cv::normalize( magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX );
    return magnitudeImage;
}

/// rearrange the quadrants of Fourier image  so that the origin is at the image center
void frequency_domain::shift( cv::Mat& image )
{
    if( image.rows == 1 && image.cols == 1)
    {
        return;
    }

    std::vector< cv::Mat > planes;
    cv::split( image, planes );

    int xMid = image.cols >> 1;
    int yMid = image.rows >> 1;

    for( unsigned int i = 0; i < planes.size(); i++ )
    {
        cv::Mat tmp;
        cv::Mat q0(planes[i], cv::Rect(0,    0,    xMid, yMid));
        cv::Mat q1(planes[i], cv::Rect(xMid, 0,    xMid, yMid));
        cv::Mat q2(planes[i], cv::Rect(0,    yMid, xMid, yMid));
        cv::Mat q3(planes[i], cv::Rect(xMid, yMid, xMid, yMid));

        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    cv::merge( planes, image );

}
    

} }

