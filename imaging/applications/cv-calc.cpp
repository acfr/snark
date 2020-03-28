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

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <tbb/parallel_for.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/scoped_ptr.hpp>
#include <comma/base/exception.h>
#include <comma/io/stream.h>
#include <comma/csv/stream.h>
#include <comma/math/interval.h>
#include <comma/name_value/parser.h>
#include <comma/string/string.h>
#include <comma/visiting/traits.h>
#include "../../imaging/cv_mat/filters.h"
#include "../../imaging/cv_mat/serialization.h"
#include "../../imaging/cv_mat/detail/life.h"
#include "../../visiting/eigen.h"

const char* name = "cv-calc: ";
static bool verbose = false;

static void usage( bool verbose=false )
{
    std::cerr << std::endl;
    std::cerr << "performs verious image manipulation or calculations on cv image streams" << std::endl;
    std::cerr << std::endl;
    std::cerr << "essential difference between cv-calc and  cv-cat" << std::endl;
    std::cerr << "    cv-cat takes one image in, applies operations, outputs one image out; e.g. cv-cat cannot skip images" << std::endl;
    std::cerr << "    cv-calc, depending on operation, may output multiple images per an one input image, skip images, or have just numeric outputs, etc" << std::endl;
    std::cerr << std::endl;
    std::cerr << "usage: cat images.bin | cv-calc <operation> [<options>] > processed.bin " << std::endl;
    std::cerr << std::endl;
    std::cerr << "operations" << std::endl;
    std::cerr << "    chessboard-corners: detect and output corners of a chessboard calibration image" << std::endl;
    std::cerr << "    crop-random,roi-random,random-crop,random-roi: output random patches of given size, e.g. to create a machine learning test dataset" << std::endl;
    std::cerr << "    draw: draw on the image primitives defined in the image header; skip a primitive if its dimensions are zero" << std::endl;
    std::cerr << "    format: output header and data format string in ascii" << std::endl;
    std::cerr << "    grep: output only images that satisfy conditions" << std::endl;
    std::cerr << "    header: output header information in ascii csv" << std::endl;
    std::cerr << "    histogram: output image histogram for all image channels appended to image header" << std::endl;
    std::cerr << "    life: take image on stdin, output game of life on each channel" << std::endl;
    std::cerr << "    mean: output image means for all image channels appended to image header" << std::endl;
    std::cerr << "    roi: given cv image data associated with a region of interest, either set everything outside the region of interest to zero or crop it" << std::endl;
    std::cerr << "    stride: stride through the image, output images of kernel size for each pixel" << std::endl;
    std::cerr << "    thin: thin image stream by discarding some images" << std::endl;
    std::cerr << "    unstride: take stride images, compose output image with a given overlap handling policy" << std::endl;
    std::cerr << "    unstride-positions: take stride index and positions within the stride, append positions in original (unstrided) image" << std::endl;
    std::cerr << std::endl;
    std::cerr << "options" << std::endl;
    std::cerr << "    --binary=[<format>]: binary format of header; default: operation dependent, see --header-format" << std::endl;
    std::cerr << "    --fields=<fields>; fields in header; default: operation dependent, see --header-fields" << std::endl;
    std::cerr << "    --flush; flush after every image" << std::endl;
    std::cerr << "    --input=<options>; default values for image header; e.g. --input=\"rows=1000;cols=500;type=ub\", see serialization options" << std::endl;
    std::cerr << "    --header-fields; show header fields and exit" << std::endl;
    std::cerr << "    --header-format; show header format and exit" << std::endl;
    std::cerr << "    --output-fields; show output fields and exit" << std::endl;
    std::cerr << "    --output-format; show output format and exit" << std::endl;
    std::cerr << std::endl;
    std::cerr << "serialization options" << std::endl;
    if( verbose ) { std::cerr << snark::cv_mat::serialization::options::usage() << std::endl; }
    else { std::cerr << "    run --help --verbose for more details..." << std::endl; }
    std::cerr << std::endl;
    std::cerr << "operation options" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    chessboard-corners" << std::endl;
    std::cerr << "        --draw; outputs image with detected corners drawn" << std::endl;
    std::cerr << "        --permissive; outputs corners when chessboard is not detected (default: only output corners when entire chessboard is found)" << std::endl;
    std::cerr << "        --select; filters images, only outputs images where chessboards were detected" << std::endl;
    std::cerr << "        --size=<rows,cols>; size of internal grid of corners in chessboard" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    crop-random,roi-random,random-crop,random-roi" << std::endl;
    std::cerr << "        --count=<n>; default=1; how many crops per image to output" << std::endl;
    std::cerr << "        --height=<y>: crop height, unless --size given" << std::endl;
    std::cerr << "        --padding=<x>,<y>: minimum crop offset from image borders" << std::endl;
    std::cerr << "        --seed=<seed>; random seed, if not specified, the seed will be randomized as much as possible" << std::endl;
    std::cerr << "        --size=<x>,<y>: crop size" << std::endl;
    std::cerr << "        --width=<x>: crop width, unless --size given" << std::endl;
    std::cerr << "        --permissive; discard images smaller than crop size" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    draw" << std::endl;
    std::cerr << "        --circles=<options>|<deprecated-options>: draw circles given in the image header; fields: centre/x,centre/y,radius" << std::endl;
    std::cerr << "        --labels=<options>|<deprecated-options>: draw labels given in the image header; fields: position/x,position/y,text" << std::endl;
    std::cerr << "        --rectangles=<options>|<deprecated-options>: rectangles as min and max given in the image header; fields: min/x,min/y,max/x,max/y" << std::endl;
    std::cerr << "            <options>:<filename>[;size=<size>][;normalized][;weight=<weight>][;color/(r|g|b)]*[;<csv_options>]; fields: t,index,<shape_fields>" << std::endl;
    std::cerr << "            <deprecated-options>:<size>[,normalized][,weight=<weight>][,color/(r|g|b)=<component>]*]" << std::endl;
    std::cerr << "                <filename>: file name as in csv options." << std::endl;
    std::cerr << "                <csv_options>: acfr csv options like fields, format etc." << std::endl;
    std::cerr << "                <size>: number of primitives" << std::endl;
    std::cerr << "                <component>: r, g or b color component. default: 0" << std::endl;
    std::cerr << "                <weight>: line weight; default: 1" << std::endl;
    std::cerr << "                normalized: if present, the input points are expected in [0,1) interval and will be rescaled to the image size" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            if (reverse) index field present, all the shapes in one block from each file is drawn, otherwise only one shape from each file is drawn" << std::endl;
    std::cerr << "            if t (timestamp) field present, then shapes are drawn on the frame with matching timestamp, otherwise they are drawn on next available image" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        examples:" << std::endl;
    std::cerr << "            > # reading one circle at a time, draw on next available image" << std::endl;
    std::cerr << "            > cv-calc draw --circles=\"circles.csv;fields=centre/x,centre/y,radius;weight=3;normalized\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # reading a block (based on reverse 'index' field) of labels at a time from a binary file, draw on next available image" << std::endl;
    std::cerr << "            > cv-calc draw --labels=\"labels.csv;fields=index,text,position/x,position/y;binary=ui,s[128],2d;color/r=255;color/b=255\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # draw circles and labels (one block each at a time) from files on images with matching timestamp" << std::endl;
    std::cerr << "            > cv-calc draw --circles=\"circles.csv;fields=t,index,centre/x,centre/y,radius;weight=3;normalized\" \\" << std::endl;
    std::cerr << "                --labels=\"labels.csv;fields=t,index,text,position/x,position/y;binary=t,ui,s[128],2d;color/r=255;color/b=255\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # read rectangles from headers and circles from file" << std::endl;
    std::cerr << "            > cv-calc draw --rectangles=\";size=5;color/g=255;weight=2\" \\" << std::endl;
    std::cerr << "                --circles=\"circles.csv;fields=t,index,centre/x,centre/y,radius;weight=3;normalized\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # read rectangles from headers (using deprecated semantics) and circles from file" << std::endl;
    std::cerr << "            > cv-calc draw --rectangles=\"5,color/g=255,weight=2\" \\" << std::endl;
    std::cerr << "                --circles=\"circles.csv;fields=t,index,centre/x,centre/y,radius;weight=3;normalized\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        fields: t,rows,cols,type,circles,labels,rectangles" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    grep" << std::endl;
    std::cerr << "        --filter,--filters=[<filters>]; apply --non-zero logic to the image with filters applied, not to image itself" << std::endl;
    std::cerr << "                                        run cv-cat --help --verbose for filters available" << std::endl;
    std::cerr << "        --non-zero=[<what>]; output only images that have non-zero pixels" << std::endl;
    std::cerr << "            <what>" << std::endl;
    std::cerr << "                ratio,[<min>][,<max>]: output only images with number of non-zero pixels within the limits of given ratios, e.g:" << std::endl;
    std::cerr << "                                           --non-zero=ratio,0.2,0.8: output images that have from 20 to 80% of non-zero pixels" << std::endl;
    std::cerr << "                                           --non-zero=ratio,,0.8: output images that have up to 80% of non-zero pixels" << std::endl;
    std::cerr << "                                           --non-zero=ratio,0.8: output images that have at least 80% of non-zero pixels" << std::endl;
    std::cerr << "                size,[<min>][,<max>]: output only images with number of non-zero pixels within given limits" << std::endl;
    std::cerr << "                                      lower limit inclusive, upper limit exclusive; e.g" << std::endl;
    std::cerr << "                                          --non-zero=size,10,1000: output images that have from 10 to 999 non-zero pixels" << std::endl;
    std::cerr << "                                          --non-zero=size,10: output images that have at least 10 non-zero pixels" << std::endl;
    std::cerr << "                                          --non-zero=size,,1000: output images that have not more than 999 non-zero pixels" << std::endl;
    std::cerr << "                                          --non-zero=size,,1: output images with all pixels zero (makes sense only when used with --filters" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    life" << std::endl;
    std::cerr << "        --exit-on-stability: exit, if no change" << std::endl;
    std::cerr << "        --procreation-treshold,--procreation=[<threshold>]: todo: document; default: 3.0" << std::endl;
    std::cerr << "        --stability-treshold,--stability=[<threshold>]: todo: document; default: 4.0" << std::endl;
    std::cerr << "        --step=[<step>]: todo: document; default: 1.0" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    mean" << std::endl;
    std::cerr << "        --threshold=[<thresh>]: apply a mask (binary threshold) and only calculate mean on pixel matching the mask." << std::endl;
    std::cerr << "              default: calculate a mean on all pixels" << std::endl;
    std::cerr << "        default output fields: t,rows,cols,type,mean,count" << std::endl;
    std::cerr << "                               count: total number of non-zero pixels used in calculating the mean" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    roi" << std::endl;
    std::cerr << "        --crop: crop to roi and output instead of setting region outside of roi to zero" << std::endl;
    std::cerr << "        --no-discard; do not discards frames where the roi is not seen" << std::endl;
    std::cerr << "        --permissive,--show-partial; allow partial overlaps of roi and input image, default: if partial roi and image overlap, set entire image to zeros." << std::endl;
    std::cerr << "        --rectangles=<options>|<deprecated-options>: rectangles as min and max given in the image header; fields: min/x,min/y,max/x,max/y" << std::endl;
    std::cerr << "            <options>:<filename>[;size=<size>][;normalized][;<csv_options>]; fields: t,index,<shape_fields>" << std::endl;
    std::cerr << "            <deprecated-options>:<size>[,normalized]" << std::endl;
    std::cerr << "                <filename>: file name as in csv options." << std::endl;
    std::cerr << "                <csv_options>: acfr csv options like fields, format etc." << std::endl;
    std::cerr << "                <size>: number of primitives" << std::endl;
    std::cerr << "                             all images in the input stream must have same number of regions" << std::endl;
    std::cerr << "                             regions with zero width or height will be ignored" << std::endl;
    std::cerr << "                normalized: if present, the input points are expected in [0,1) interval and will be rescaled to the image size" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            if (reverse) index field present, all the shapes in one block from each file is drawn, otherwise only one shape from each file is drawn" << std::endl;
    std::cerr << "            if t (timestamp) field present, then shapes are drawn on the frame with matching timestamp, otherwise they are drawn on next available image" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        examples:" << std::endl;
    std::cerr << "            > # reading one rectangle at a time, mask image with matching timestamp" << std::endl;
    std::cerr << "            > cv-calc roi --rectangles=\"boxes.csv;fields=t,min/x,min/y,max/x,max/y\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # reading block of rectangles at a time, mask next available image" << std::endl;
    std::cerr << "            > cv-calc roi --rectangles=\"boxes.csv;fields=index,min/x,min/y,max/x,max/y\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # reading rectangles from headers, crop relevant image" << std::endl;
    std::cerr << "            > cv-calc roi --rectangles=\";size=5;normalized\" --crop" << std::endl;
    std::cerr << std::endl;
    std::cerr << "            > # reading rectangles from headers (deprecated semantics), crop relevant image" << std::endl;
    std::cerr << "            > cv-calc roi --rectangles=\"5,normalized\" --crop" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        fields: t,rows,cols,type,rectangles" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    stride" << std::endl;
    std::cerr << "        --filter,--filters=[<filters>]; see grep operation; added to stride for performance" << std::endl;
    std::cerr << "        --fit-last; fit last stride exactly to the image size, i.e. last stride may be irregular" << std::endl;
    std::cerr << "        --input=[<options>]; input options; run cv-cat --help --verbose for details" << std::endl;
    std::cerr << "        --non-zero=[<what>]; see grep operation; added to stride for performance" << std::endl;
    std::cerr << "        --output=[<options>]; output options; run cv-cat --help --verbose for details" << std::endl;
    std::cerr << "        --padding=[<padding>]; padding, 'same' or 'valid' (see e.g. tensorflow for the meaning); default: valid" << std::endl;
    std::cerr << "        --shape,--kernel,--size=<x>,<y>; image size" << std::endl;
    std::cerr << "        --strides=[<x>,<y>]; stride size; default: 1,1" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    thin" << std::endl;
    std::cerr << "        by thinning rate" << std::endl;
    std::cerr << "            --deterministic; output frames at a given thinning rate with as uniform intervals as possible" << std::endl;
    std::cerr << "                             default: output frames at a given thinning rate at random with uniform distribution" << std::endl;
    std::cerr << "            --rate=<rate>; thinning rate between 0 and 1" << std::endl;
    std::cerr << "        by frames per second" << std::endl;
    std::cerr << "            --from-fps,--input-fps=<fps>; input fps (since it is impossible to know it upfront)" << std::endl;
    std::cerr << "            --to-fps,--fps=<fps>; thin to a given <fps>, same as --rate=<to-fps>/<from-fps> --deterministic" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    unstride" << std::endl;
    std::cerr << "        --fit-last; last stride is fit exactly to the image size, i.e. last stride may be irregular" << std::endl;
    std::cerr << "        --how-to-handle-overlaps,--how=<how>; default: last; how to handle overlaps" << std::endl;
    std::cerr << "            last: simply put the next tile on top on the previous one" << std::endl;
    std::cerr << "            linear: linearly interpolate across stride overlaps (todo)" << std::endl;
    std::cerr << "            other policies: todo" << std::endl;
    std::cerr << "        --input=[<options>]; input options; run cv-cat --help --verbose for details" << std::endl;
    std::cerr << "        --output=[<options>]; output options; run cv-cat --help --verbose for details" << std::endl;
    std::cerr << "        --output-number-of-strides,--number-of-strides: output number of strides as <x>,<y> to stdout and exit" << std::endl;
    //std::cerr << "        --padding=[<padding>]; padding, 'same' or 'valid' (see e.g. tensorflow for the meaning); default: valid" << std::endl;
    std::cerr << "        --shape,--kernel,--size=<x>,<y>; image size" << std::endl;
    std::cerr << "        --strides=[<x>,<y>]; stride size; default: 1,1" << std::endl;
    std::cerr << "        --unstrided-size,--unstrided=<width>,<height>; original (unstrided) image size" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    unstride-positions" << std::endl;
    std::cerr << "        options" << std::endl;
    std::cerr << "            --permissive: discard records with invalid stride id" << std::endl;
    std::cerr << "            --positions=<n>: number of positions in stride; fields: positions[i]/x,positions[i]/y, (default: 1)" << std::endl;
    std::cerr << "            --shape,--kernel,--size=<x>,<y>; (tile) image size (see: stride operation)" << std::endl;
    std::cerr << "            --strides=[<x>,<y>]; stride size; default: 1,1 (see: stride operation)" << std::endl;
    std::cerr << "            --unstrided-size,--unstrided=<width>,<height>; original (unstrided) image size" << std::endl;
    std::cerr << "        fields" << std::endl;
    std::cerr << "            index: stride index counting strides row by row" << std::endl;
    std::cerr << "            x,y: todo: stride 2d index" << std::endl;
    std::cerr << "            positions: pixel positions to unstride, e.g: positions[0]/x,positions[0]/y,positions[1]/x,positions[1]/y; also see --positions" << std::endl;
    std::cerr << std::endl;
    std::cerr << "examples" << std::endl;
    std::cerr << "    draw" << std::endl;
    std::cerr << "        # read rectangles and labels from header and cirles from file" << std::endl;
    std::cerr << "        cv-cat --file image.jpg \\" << std::endl;
    std::cerr << "            | csv-paste 'value=50,100,100,200,150,150,250,250,250,150,some_text;binary=10ui,s[64]' \"-;binary=t,3ui,s[$image_size_in_bytes]\" \\" << std::endl;
    std::cerr << "            | cv-calc draw --binary=10ui,s[64],t,3ui \\" << std::endl;
    std::cerr << "                           --fields rectangles,labels,t,rows,cols,type  \\" << std::endl;
    std::cerr << "                           --rectangles=2,weight=3,color/r=255 \\" << std::endl;
    std::cerr << "                           --circles=circles.csv;fields=t,index,centre/x,centre/y,radius;weight=3,color/b=255 \\" << std::endl;
    std::cerr << "                           --labels=1,weight=2,color/r=255 \\" << std::endl;
    std::cerr << "            | tail -c<original image size + 20> \\" << std::endl;
    std::cerr << "            | cv-cat --output no-header 'encode=jpg' > output.jpg" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    header" << std::endl;
    std::cerr << "        cat data.bin | cv-calc header" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    format" << std::endl;
    std::cerr << "        cat data.bin | cv-calc format" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    life" << std::endl;
    std::cerr << "        a combination of parameters producing fairly long-living colony:" << std::endl;
    std::cerr << "        cv-cat --file ~/tmp/some-image.jpg | cv-calc life --procreation 3 --stability 5.255 --step 0.02 | cv-cat \"count;view;null\"" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    roi" << std::endl;
    std::cerr << "        Setting everything but the roi rectangle to 0 for all images" << std::endl;
    std::cerr << "        ROI fields must be pre-pended. This roi is is a square of (100,100) to (300,300)" << std::endl;
    std::cerr << "        Given a cv-cat image stream with format 't,3ui,s[1572864]'." << std::endl;
    std::cerr << std::endl;
    std::cerr << "        cat data.bin | csv-paste \"value=100,100,300,300;binary=4i\" \"-;binary=t,3ui,s[1572864]\" \\" << std::endl;
    std::cerr << "            | cv-calc roi -v | csv-bin-cut '4i,t,3ui,s[1572864]' --fields 5-9 >output.bin" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        Explicity specifying fields. Image payload data field is not specified for cv-calc, not set for --binary either" << std::endl;
    std::cerr << "        The user must explcitly list all four roi fields. Using 'min,max' is not possible." << std::endl;
    std::cerr << std::endl;
    std::cerr << "        cat data.bin | csv-paste \"value=100,100,999,300,300;binary=5i\" \"-;binary=t,3ui,s[1572864]\" \\" << std::endl;
    std::cerr << "            | cv-calc roi --fields min/x,min/y,,max/x,max/y,t,rows,cols,type --binary '5i,t,3ui' \\" << std::endl;
    std::cerr << "            | csv-bin-cut '5i,t,3ui,s[1572864]' --fields 6-10 >output.bin" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        Multiple roi" << std::endl;
    std::cerr << "        cv-cat --file image.jpg \\" << std::endl;
    std::cerr << "            | cv-calc roi --rectangles=\"boxes.bin;fields=t,index,min/x,min/y,max/x,max/y;binary=t,ui,4d;normalized\" \\" << std::endl;
    std::cerr << "            | tail -c<original image size + 20> \\" << std::endl;
    std::cerr << "            | cv-cat --output no-header 'encode=jpg' > output.jpg" << std::endl;
    std::cerr << std::endl;
    std::cerr << "        Multiple roi (deprecated)" << std::endl;
    std::cerr << "        cv-cat --file image.jpg \\" << std::endl;
    std::cerr << "            | csv-paste 'value=50,100,100,200,150,150,250,250;binary=8ui' \"-;binary=t,3ui,s[$image_size_in_bytes]\" \\" << std::endl;
    std::cerr << "            | cv-calc roi --binary=8ui,t,3ui --fields rectangles,t,rows,cols,type --rectangles=2 \\" << std::endl;
    std::cerr << "            | tail -c<original image size + 20> \\" << std::endl;
    std::cerr << "            | cv-cat --output no-header 'encode=jpg' > output.jpg" << std::endl;
    std::cerr << std::endl;
    std::cerr << std::endl;
    std::cerr << "    unstride-positions" << std::endl;
    std::cerr << "        <process strides, output stride index and rectangle> |" << std::endl;
    std::cerr << "            cv-calc unstride-positions --fields index,positions --positions 2 --size 600,400 --strides 300,200 --unstrided-size 1200,800" << std::endl;
    std::cerr << std::endl;
    exit( 0 );
}

struct positions_t
{
    struct config
    {
        comma::uint32 size;
        config() : size( 1 ) {}
    };

    std::vector< cv::Point2d > positions;

    positions_t() {}
    positions_t( const comma::command_line_options& options )
    {
        std::string config_string = options.value< std::string >( "--positions", "1" );
        if( config_string.empty() ) { return; }
        const auto& config = comma::name_value::parser( "size", ',', '=' ).get< positions_t::config >( config_string );
        positions.resize( config.size );
    }
};

struct stride_positions_t : public positions_t
{
    comma::uint32 index;
    stride_positions_t() {}
    stride_positions_t( const comma::command_line_options& options ) : positions_t( options ) {}
};

struct extents
{
    cv::Point2f min;
    cv::Point2f max;
    extents(): min( 0, 0 ), max( 0, 0 ) {}

    void unnormalise( int const rows, int const cols )
    {
        min.x = std::round( min.x * cols ); min.y = std::round( min.y * rows );
        max.x = std::round( max.x * cols ); max.y = std::round( max.y * rows );
    }
};

struct chessboard_corner_t
{
    boost::posix_time::ptime t;
    comma::uint32 block;
    comma::uint32 row;
    comma::uint32 col;
    Eigen::Vector2d position;
};

namespace region
{
template< typename T >
struct indexed : public T
{
    boost::posix_time::ptime timestamp;
    comma::uint32 index;

    indexed() : index( 0 ) {}
    indexed( T const& value) : T(value), index( 0U ) {}
};

struct properties
{
    bool normalized;
    properties() : normalized( false ) {}
};

struct config : public comma::csv::options
{
    comma::uint32 size;
    config() : size( 0 ) {}
};

using rectangle = ::extents;

struct circle
{
    cv::Point2f centre;
    double radius;
    circle() : centre( 0, 0 ), radius( 0 ) {}
};

struct label
{
    cv::Point2f position;
    std::string text;

    label() : position( 0, 0 ) {}
};

template < typename T >
class join_filter
{
public:
    using key_type = boost::posix_time::ptime;
    using record_type = indexed< T >;
    using stream_type = comma::csv::input_stream< record_type >;

    join_filter() {}

    template < typename properties_type >
    join_filter( comma::csv::options const& csv, properties_type properties )
        : m_ifstrm( csv.filename, csv.binary() ? comma::io::mode::binary : comma::io::mode::ascii )
        , m_istrm( *m_ifstrm, csv, record_type( T( properties ) ) )
        , has_timestamp( csv.has_field( "t" ) )
        , has_index( csv.has_field( "index" ) )
    {
        if( m_ifstrm.fd() == comma::io::invalid_file_descriptor ) { std::cerr << "cv-calc: draw: failed to open '" << csv.filename << "'" << std::endl; exit( 1 ); }
    }

    std::vector< record_type > read_at( key_type const& timestamp )
    {
        std::vector< record_type > result;
        if( has_timestamp )
        {
            if( m_record )
            {
                if( timestamp < m_record->timestamp ) { return result; }
                if( timestamp == m_record->timestamp ) { result.push_back( *m_record ); m_record = boost::none; }
                else if( timestamp > m_record->timestamp ) { m_record = boost::none; }
            }
            while( m_ifstrm->good() && !m_ifstrm->eof() )
            {
                auto rec = m_istrm.read(); if( !rec ) { break; }
                if( timestamp > rec->timestamp ) continue;
                if( timestamp < rec->timestamp ) { m_record = *rec; break; }
                result.push_back( *rec );
                if( 0 == rec->index && has_index ) { break; } // TODO: use config::size semantics here
            }
        }
        else
        {
            assert( !m_record );
            while( m_ifstrm->good() && !m_ifstrm->eof() )
            {
                auto rec = m_istrm.read(); if( !rec ) { break; }
                result.push_back( *rec );
                if( 0 == rec->index ) { break; } // TODO: use config::size semantics here
            }
        }
        return result;
    }

    void read_at( std::vector< T >& list, key_type const& timestamp )
    {
        if( has_timestamp )
        {
            if( m_record )
            {
                if( timestamp < m_record->timestamp ) { return; }
                if( timestamp == m_record->timestamp ) { list.push_back( *m_record ); m_record = boost::none; }
                else if( timestamp > m_record->timestamp ) { m_record = boost::none; }
            }
            while( m_ifstrm->good() && !m_ifstrm->eof() )
            {
                auto rec = m_istrm.read(); if( !rec ) { break; }
                if( timestamp > rec->timestamp ) continue;
                if( timestamp < rec->timestamp ) { m_record = *rec; return; }
                list.push_back( *rec );
                if( 0 == rec->index && has_index ) { return; } // TODO: use config::size here
            }
        }
        else
        {
            assert( !m_record );
            while( m_ifstrm->good() && !m_ifstrm->eof() )
            {
                auto rec = m_istrm.read(); if( !rec ) { break; }
                list.push_back( *rec );
                if( 0 == rec->index ) { break; } // TODO: use config::size here
            }
        }
    }

private:
    comma::io::istream m_ifstrm;
    stream_type m_istrm;
    boost::optional< record_type > m_record;
    bool has_timestamp;
    bool has_index;
};

template < typename shape_type, typename config_type > static void init_( std::vector< shape_type >& header_shapes , std::unique_ptr< join_filter< shape_type > >& stream_shapes
        , const comma::command_line_options& options, const std::string& what )
{
    std::string config_string = options.value< std::string >( what, "" );
    if( config_string.empty() ) { return; }

    char delimiter = std::string::npos == config_string.find( ';' ) && std::string::npos != config_string.find( ',' ) ? ',' : ';';
    if( ',' == delimiter ) { std::cerr << "cv-calc: warning: ',' as delimiter in shape attributes is deprecated, use ';'. Got attributes string: \"" << config_string << "\"" << std::endl; }

    std::string unnamed_attr = config_string.substr( 0U, config_string.find_first_of( delimiter ) );
    std::string unnamed_attr_name = !unnamed_attr.empty() && std::find_if_not( unnamed_attr.cbegin(), unnamed_attr.cend(), []( char const c ) { return 0 != std::isdigit( c ); } ) == unnamed_attr.cend()
        ? "size" : "filename";

    if( "size" == unnamed_attr_name ) { std::cerr << "cv-calc: warning: using 'size' as the unnamed shape attribute is deprecated, the unnamed attribute should be filename. Attribute string: " << config_string << std::endl; }
    bool data_from_stdin = unnamed_attr.empty() || unnamed_attr == "-" || unnamed_attr.find_first_of( '=' ) != std::string::npos;
    if( verbose ) { std::cerr << "cv-calc: draw: shapes from " << ( data_from_stdin ? std::string( "stdin" ) : unnamed_attr ) << std::endl; }
    auto cfg = data_from_stdin
             ? comma::name_value::parser( delimiter, '=' ).get< config_type >( config_string )
             : comma::name_value::parser( unnamed_attr_name, delimiter, '=' ).get< config_type >( config_string );
    if( !cfg.filename.empty() ) { cfg.full_xpath = true; stream_shapes = std::unique_ptr< region::join_filter< shape_type > >( new region::join_filter< shape_type >( cfg, cfg.properties ) ); }
    if( 0 < cfg.size ) header_shapes.resize( cfg.size, shape_type( cfg.properties ) );
}

} //namespace region

namespace roi {

enum class status
{
    error = -2,
    success = 0,
    ignore = 1
};

struct shapes
{
    struct config : public region::config
    {
        region::properties properties;
    };

    struct rectangle : public region::rectangle
    {
        region::properties properties; // to do, per record properties
        rectangle() {}
        rectangle( region::properties const& properties ) : properties( properties ) {}

        std::pair< status, cv::Rect > validate( int const rows, int const cols, bool const permissive )
        {
            ::extents ext = *this;
            if( properties.normalized ) { ext.unnormalise( rows, cols ); }
            if( permissive )
            {
                ext.min.x = std::max( int( ext.min.x ), 0 );
                ext.min.y = std::max( int( ext.min.y ), 0 );
                ext.max.x = std::min( int( ext.max.x ), cols - 1 );
                ext.max.y = std::min( int( ext.max.y ), rows - 1 );
            }
            cv::Rect rect( ext.min.x, ext.min.y, ext.max.x - ext.min.x, ext.max.y - ext.min.y );
            if( 0 > rect.width || 0 > rect.height ) { return std::make_pair( status::error, cv::Rect() ); }
            if( 0 == rect.width || 0 == rect.height ) { return std::make_pair( status::ignore, cv::Rect() ); }
            if( ext.min.x < 0 || ext.min.y < 0 || ext.max.x >= cols || ext.max.y >= rows ) { return std::make_pair( status::ignore, cv::Rect() ); }
            return std::make_pair( status::success, rect );
        }

    };

    struct header_shapes { std::vector< shapes::rectangle > rectangles; };
    struct stream_shapes { std::unique_ptr< region::join_filter< shapes::rectangle > > rectangles; };
    header_shapes hdr_shapes;
    stream_shapes strm_shapes;

    shapes() {}
    shapes( const comma::command_line_options& options ) { region::init_< rectangle, config >( hdr_shapes.rectangles, strm_shapes.rectangles, options, "--rectangles" ); }
};

} // namespace roi {

namespace draw {

class shapes
{
public:
    struct color
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        color(): r( 0 ), g( 0 ), b( 0 ) {}
        operator cv::Scalar() const { return cv::Scalar( b, g, r ); }
    };

    struct properties : public region::properties
    {
        shapes::color color;
        comma::uint32 weight;
        properties() : weight( 1 ) {}
    };

    struct config : public region::config
    {
        shapes::properties properties;
    };

        //void draw( cv::Mat m ) const { cv::circle( m, center, radius, color, thickness, line_type, shift ); }

    struct rectangle : public region::rectangle
    {
        shapes::properties properties; // todo: record-wise support
        rectangle() {}
        rectangle( const shapes::properties& properties ): properties( properties ) {}
        bool draw( cv::Mat m ) const
        {
            if( max.x == 0 && min.x == 0 && max.y == 0 && max.y == 0 ) { return false; }
            if( properties.normalized ) { cv::rectangle( m, cv::Point2i( min.x * m.cols, min.y * m.rows ), cv::Point2i( max.x * m.cols, max.y * m.rows ), properties.color, properties.weight ); } // CV_AA );
            else { cv::rectangle( m, min, max, properties.color, properties.weight ); } // CV_AA );
            return true;
        }
    };

    struct circle : public region::circle
    {
        shapes::properties properties; // todo: record-wise support
        circle() {}
        circle( const shapes::properties& properties ): properties( properties ) {}
        bool draw( cv::Mat m ) const
        {
            if( !comma::math::less( 0, radius ) ) { return false; }
            if( properties.normalized ) { cv::circle( m, cv::Point2i( centre.x * m.cols, centre.y * m.rows ), radius * m.cols, properties.color, properties.weight, cv::LINE_AA ); }
            else { cv::circle( m, centre, radius, properties.color, properties.weight, cv::LINE_AA ); }
            return true;
        }
    };

    struct label : public region::label
    {
        shapes::properties properties; // todo: record-wise support
        label() {}
        label( const shapes::properties& properties ): properties( properties ) {}
        bool draw( cv::Mat m ) const
        {
            if( text.empty() ) { return false; }
            cv::putText( m, text, properties.normalized ? cv::Point2f( position.x * m.cols, position.y * m.rows ) : position, cv::FONT_HERSHEY_SIMPLEX, 1.0, properties.color, properties.weight, cv::LINE_AA );
            return true;
        }
    };

    struct header_shapes
    {
        std::vector< shapes::circle > circles;
        std::vector< shapes::label > labels;
        std::vector< shapes::rectangle > rectangles;

        void draw( cv::Mat m )
        {
            draw_( m, circles );
            draw_( m, labels );
            draw_( m, rectangles );
        }

    private:
        template < typename T > void draw_( cv::Mat m, const std::vector< T >& hdr_shapes ) const { for( auto& ii : hdr_shapes ) { ii.draw( m ); } }
    };

    struct stream_shapes
    {
        std::unique_ptr< region::join_filter< shapes::circle > > circles;
        std::unique_ptr< region::join_filter< shapes::label > > labels;
        std::unique_ptr< region::join_filter< shapes::rectangle > > rectangles;

        void draw( cv::Mat m, boost::posix_time::ptime const& t )
        {
            draw_( m, t, circles );
            draw_( m, t, labels );
            draw_( m, t, rectangles );
        }
    private:
        template < typename T > void draw_( cv::Mat m, boost::posix_time::ptime const& t, std::unique_ptr< region::join_filter< T > >& strm_shapes )
        {
            if( strm_shapes ) { auto list = strm_shapes->read_at( t ); for( auto& ii : list ) { ii.draw( m ); } }
        }
    };

    header_shapes hdr_shapes;
    stream_shapes strm_shapes;

    shapes() {}

    shapes( const comma::command_line_options& options )
    {
        region::init_< circle, config >( hdr_shapes.circles, strm_shapes.circles, options, "--circles" );
        region::init_< label, config >( hdr_shapes.labels, strm_shapes.labels, options, "--labels" );
        region::init_< rectangle, config >( hdr_shapes.rectangles, strm_shapes.rectangles, options, "--rectangles" );

    }
};

} // namespace draw {

namespace comma { namespace visiting {

template <> struct traits< positions_t::config >
{
    template < typename Key, class Visitor > static void visit( const Key&, positions_t::config& p, Visitor& v ) { v.apply( "size", p.size ); }
    template < typename Key, class Visitor > static void visit( const Key&, const positions_t::config& p, Visitor& v ) { v.apply( "size", p.size ); }
};

template <> struct traits< positions_t >
{
    template < typename Key, class Visitor > static void visit( const Key&, positions_t& p, Visitor& v ) { v.apply( "positions", p.positions ); }
    template < typename Key, class Visitor > static void visit( const Key&, const positions_t& p, Visitor& v ) { v.apply( "positions", p.positions ); }
};

template <> struct traits< stride_positions_t >
{
    template < typename Key, class Visitor > static void visit( const Key& k, stride_positions_t& p, Visitor& v )
    {
        v.apply( "index", p.index );
        traits< ::positions_t >::visit( k, p, v );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const stride_positions_t& p, Visitor& v )
    {
        v.apply( "index", p.index );
        traits< ::positions_t >::visit( k, p, v );
    }
};

template < typename T > struct traits< cv::Point_< T > >
{
    template < typename Key, class Visitor > static void visit( const Key&, cv::Point_< T >& p, Visitor& v )
    {
        v.apply( "x", p.x );
        v.apply( "y", p.y );
    }

    template < typename Key, class Visitor > static void visit( const Key&, const cv::Point_< T >& p, Visitor& v )
    {
        v.apply( "x", p.x );
        v.apply( "y", p.y );
    }
};

template <> struct traits< ::extents >
{
    template < typename Key, class Visitor > static void visit( const Key&, ::extents& p, Visitor& v )
    {
        v.apply( "min", p.min );
        v.apply( "max", p.max );
    }

    template < typename Key, class Visitor > static void visit( const Key&, const ::extents& p, Visitor& v )
    {
        v.apply( "min", p.min );
        v.apply( "max", p.max );
    }
};

template < typename T > struct traits< region::indexed< T > >
{
    template < typename Key, class Visitor > static void visit( const Key& k, region::indexed< T >& p, Visitor& v )
    {
        traits< T >::visit( k, p, v );
        v.apply( "t", p.timestamp );
        v.apply( "index", p.index );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const region::indexed< T >& p, Visitor& v )
    {
        traits< T >::visit( k, p, v );
        v.apply( "t", p.timestamp );
        v.apply( "index", p.index );
    }
};

template <> struct traits< region::properties >
{
    template < typename Key, class Visitor > static void visit( const Key&, region::properties& p, Visitor& v ) { v.apply( "normalized", p.normalized ); }
    template < typename Key, class Visitor > static void visit( const Key&, const region::properties& p, Visitor& v ) { v.apply( "normalized", p.normalized ); }
};

template <> struct traits< region::config >
{
    template < typename Key, class Visitor > static void visit( const Key& k, region::config& p, Visitor& v )
    {
        traits< comma::csv::options >::visit( k, p, v );
        v.apply( "size", p.size );
    }
    template < typename Key, class Visitor > static void visit( const Key& k, const region::config& p, Visitor& v )
    {
        traits< comma::csv::options >::visit( k, p, v );
        v.apply( "size", p.size );
    }
};

template <> struct traits< region::circle >
{
    template < typename Key, class Visitor > static void visit( const Key& k, region::circle& p, Visitor& v )
    {
        v.apply( "centre", p.centre );
        v.apply( "radius", p.radius );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const region::circle& p, Visitor& v )
    {
        v.apply( "centre", p.centre );
        v.apply( "radius", p.radius );
    }
};

template <> struct traits< region::label >
{
    template < typename Key, class Visitor > static void visit( const Key& k, region::label& p, Visitor& v )
    {
        v.apply( "position", p.position );
        v.apply( "text", p.text );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const region::label& p, Visitor& v )
    {
        v.apply( "position", p.position );
        v.apply( "text", p.text );
    }
};

template <> struct traits< roi::shapes::config >
{
    template < typename Key, class Visitor > static void visit( const Key& k, roi::shapes::config& p, Visitor& v )
    {
        traits< region::config >::visit( k, p, v );
        traits< region::properties >::visit( k, p.properties, v );
    }
    template < typename Key, class Visitor > static void visit( const Key& k, const roi::shapes::config& p, Visitor& v )
    {
        traits< region::config >::visit( k, p, v );
        traits< region::properties >::visit( k, p.properties, v );
    }
};

template <> struct traits< roi::shapes::rectangle >
{
    template < typename Key, class Visitor > static void visit( const Key& k, roi::shapes::rectangle& p, Visitor& v ) { traits< ::extents >::visit( k, p, v ); }
    template < typename Key, class Visitor > static void visit( const Key& k, const roi::shapes::rectangle& p, Visitor& v ) { traits< ::extents >::visit( k, p, v ); }
};

template <> struct traits< roi::shapes::header_shapes >
{
    template < typename Key, class Visitor > static void visit( const Key&, roi::shapes::header_shapes& p, Visitor& v )
    {
        //v.apply( "circles", p.circles );
        v.apply( "rectangles", p.rectangles );
    }

    template < typename Key, class Visitor > static void visit( const Key&, const roi::shapes::header_shapes& p, Visitor& v )
    {
        //v.apply( "circles", p.circles );
        v.apply( "rectangles", p.rectangles );
    }
};

template <> struct traits< draw::shapes::color >
{
    template < typename Key, class Visitor > static void visit( const Key&, draw::shapes::color& p, Visitor& v )
    {
        unsigned int r = p.r;
        v.apply( "r", r );
        p.r = r;
        unsigned int g = p.g;
        v.apply( "g", g );
        p.g = g;
        unsigned int b = p.b;
        v.apply( "b", b );
        p.b = b;
    }

    template < typename Key, class Visitor > static void visit( const Key&, const draw::shapes::color& p, Visitor& v )
    {
        v.apply( "r", comma::uint32( p.r ) );
        v.apply( "g", comma::uint32( p.g ) );
        v.apply( "b", comma::uint32( p.b ) );
    }
};

template <> struct traits< draw::shapes::properties >
{
    template < typename Key, class Visitor > static void visit( const Key& k, draw::shapes::properties& p, Visitor& v )
    {
        traits< region::properties >::visit( k, p, v );
        v.apply( "color", p.color );
        v.apply( "weight", p.weight );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const draw::shapes::properties& p, Visitor& v )
    {
        traits< region::properties >::visit( k, p, v );
        v.apply( "color", p.color );
        v.apply( "weight", p.weight );
    }
};

template <> struct traits< draw::shapes::config >
{
    template < typename Key, class Visitor > static void visit( const Key& k, draw::shapes::config& p, Visitor& v )
    {
        traits< region::config >::visit( k, p, v );
        traits< draw::shapes::properties >::visit( k, p.properties, v );
    }

    template < typename Key, class Visitor > static void visit( const Key& k, const draw::shapes::config& p, Visitor& v )
    {
        traits< region::config >::visit( k, p, v );
        traits< draw::shapes::properties >::visit( k, p.properties, v );
    }
};

template <> struct traits< draw::shapes::rectangle >
{
    template < typename Key, class Visitor > static void visit( const Key& k, draw::shapes::rectangle& p, Visitor& v ) { traits< ::extents >::visit( k, p, v ); }
    template < typename Key, class Visitor > static void visit( const Key& k, const draw::shapes::rectangle& p, Visitor& v ) { traits< ::extents >::visit( k, p, v ); }
};

template <> struct traits< draw::shapes::circle >
{
    template < typename Key, class Visitor > static void visit( const Key& k, draw::shapes::circle& p, Visitor& v ) { traits< region::circle >::visit( k, p, v ); }
    template < typename Key, class Visitor > static void visit( const Key& k, const draw::shapes::circle& p, Visitor& v ) { traits< region::circle >::visit( k, p, v ); }
};

template <> struct traits< draw::shapes::label >
{
    template < typename Key, class Visitor > static void visit( const Key& k, draw::shapes::label& p, Visitor& v ) { traits< region::label >::visit( k, p, v ); }
    template < typename Key, class Visitor > static void visit( const Key& k, const draw::shapes::label& p, Visitor& v ) { traits< region::label >::visit( k, p, v ); }
};

template <> struct traits< draw::shapes::header_shapes >
{
    template < typename Key, class Visitor > static void visit( const Key&, draw::shapes::header_shapes& p, Visitor& v )
    {
        v.apply( "circles", p.circles );
        v.apply( "labels", p.labels );
        v.apply( "rectangles", p.rectangles );
    }

    template < typename Key, class Visitor > static void visit( const Key&, const draw::shapes::header_shapes& p, Visitor& v )
    {
        v.apply( "circles", p.circles );
        v.apply( "labels", p.labels );
        v.apply( "rectangles", p.rectangles );
    }
};

template <> struct traits< chessboard_corner_t >
{
    template < typename Key, class Visitor > static void visit( const Key&, chessboard_corner_t& p, Visitor& v)
    {
        v.apply("t", p.t);
        v.apply("block", p.block);
        v.apply("row", p.row);
        v.apply("col", p.col);
        v.apply("position", p.position);
    }
    template < typename Key, class Visitor > static void visit( const Key&, const chessboard_corner_t& p, Visitor& v)
    {
        v.apply("t", p.t);
        v.apply("block", p.block);
        v.apply("row", p.row);
        v.apply("col", p.col);
        v.apply("position", p.position);
    }
};

} } // namespace comma { namespace visiting {

namespace grep {

class non_zero
{
    public:
        non_zero() {}
        non_zero( const std::string& s )
        {
            if( s.empty() ) { return; }
            const std::vector< std::string >& v = comma::split( s, ',' );
            if( v[0] == "ratio" )
            {
                if( v.size() < 2 ) { COMMA_THROW( comma::exception, "expected --non-zero=ratio,<min>,<max>; got --non-zero=ratio" ); }
                if( !v[1].empty() ) { ratio_.first = boost::lexical_cast< double >( v[1] ); }
                if( v.size() > 2 && !v[2].empty() ) { ratio_.second = boost::lexical_cast< double >( v[2] ); }
                return;
            }
            if( v[0] == "size" )
            {
                if( v.size() < 2 ) { COMMA_THROW( comma::exception, "expected --non-zero=size,<min>,<max>; got --non-zero=size" ); }
                if( !v[1].empty() ) { size_.first = boost::lexical_cast< unsigned int >( v[1] ); }
                if( v.size() > 2 && !v[2].empty() ) { size_.second = boost::lexical_cast< unsigned int >( v[2] ); }
                return;
            }
            COMMA_THROW( comma::exception, "--non-zero: expected 'ratio' or 'size', got: '" << v[0] << "'" );
        }
        operator bool() const { return static_cast< bool >( ratio_.first ) || static_cast< bool >( ratio_.second ) || static_cast< bool >( size_.first ) || static_cast< bool >( size_.second ); }
        void size( unsigned int image_size )
        {
            if( ratio_.first ) { size_.first = image_size * *ratio_.first; }
            if( ratio_.second ) { size_.second = image_size * *ratio_.second; }
        }
        bool keep( unsigned int count ) const { return ( !size_.first || *size_.first <= count ) && ( !size_.second || count < *size_.second ); }
        bool keep( const cv::Mat& m ) const { return keep( count( m ) ); }
        unsigned int count( const cv::Mat& m ) const
        {
            static std::vector< char > zero_pixel( m.elemSize(), 0 );
            std::vector< unsigned int > counts( m.rows, 0 );
            tbb::parallel_for( tbb::blocked_range< std::size_t >( 0, m.rows )
                                , [&]( const tbb::blocked_range< std::size_t >& r )
                                {
                                    for( unsigned int i = r.begin(); i < r.end(); ++i )
                                    {
                                        for( const unsigned char* ptr = m.ptr( i ); ptr < m.ptr( i + 1 ); ptr += m.elemSize() )
                                        {
                                            if( ::memcmp( ptr, &zero_pixel[0], zero_pixel.size() ) != 0 ) { ++counts[i]; }
                                        }
                                    }
                                } );
            return std::accumulate( counts.begin(), counts.end(), 0 );
        }
        const uchar* ptr;

    private:
        std::pair< boost::optional< double >, boost::optional< double > > ratio_;
        std::pair< boost::optional< unsigned int >, boost::optional< unsigned int > > size_;
        bool empty_;
        bool keep_counting_( unsigned int count ) const
        {
            if( size_.second ) { return *size_.second < count; }
            return size_.first && ( count < *size_.first );
        }
};

} // namespace grep {

namespace thin {

class keep
{
    public:
        keep( double rate, bool deterministic )
            : rate_( rate )
            , deterministic_( deterministic )
            , distribution_( 0, 1 )
            , random_( generator_, distribution_ )
            , size_( 1.0e+9 )
            , step_( 0 )
            , count_( 0 )
        {
        }

        operator bool() { return deterministic_ ? deterministic_impl_() : random_() < rate_; }

    private:
        double rate_;
        bool deterministic_;
        boost::mt19937 generator_;
        boost::uniform_real<> distribution_;
        boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_;
        comma::uint64 size_;
        comma::uint64 step_;
        comma::uint64 count_;
        bool deterministic_impl_()
        {
            ++count_;
            if( count_ < ( step_ + 1 ) / rate_ ) { return false; }
            ++step_;
            if( step_ == size_ ) { count_ = 0; step_ = 0; }
            return true;
        }
};

} // namespace thin {

static snark::cv_mat::serialization::options handle_fields_and_format( const comma::csv::options& csv, snark::cv_mat::serialization::options input_options )
{
    if( !csv.fields.empty() && !input_options.fields.empty() ) { COMMA_THROW(comma::exception, "cv-calc: please set fields in --fields or --input, not both"); }
    if( csv.binary() && !input_options.format.elements().empty() ) { COMMA_THROW(comma::exception, "cv-calc: please set binary format in --binary or --input, not both"); }
    if( !csv.fields.empty() && input_options.fields.empty() ) { input_options.fields = csv.fields; }
    if( csv.binary() && input_options.format.string().empty() ) { input_options.format = csv.format(); }
    return input_options;
}

namespace snark { namespace unstride { namespace overlap {

struct base
{
    virtual ~base() {}
    
    virtual void append( cv::Mat image, cv::Mat tile, unsigned int x, unsigned int y ) = 0;
};
    
struct last: public overlap::base
{
    void append( cv::Mat image, cv::Mat tile, unsigned int x, unsigned int y ) { tile.copyTo( cv::Mat( image, cv::Rect( x, y, tile.cols, tile.rows ) ) ); }
};
    
class linear: public overlap::base
{
    public:
        linear(): x_( 0 ), y_( 0 ), y_prev_( 0 ) {}
        
        void append( cv::Mat image, cv::Mat tile, unsigned int x, unsigned int y )
        {
            if( image.type() != CV_32F ) { std::cerr << "cv-calc: unstride: --how=linear: currently works only on images of type f (CV_32F, " << CV_32F << "); got: " << image.type() << std::endl; exit( 1 ); }
            if( x == 0 ) { y_ = y_prev_; }
            cv::Mat s = y == 0 ? tile : vertical_( image, tile, x, y );
            cv::Mat t = x == 0 ? s : horizontal_( image, s, x, y );
            t.copyTo( cv::Mat( image, cv::Rect( x, y, tile.cols, tile.rows ) ) );
            x_ = x;
            y_prev_ = y;
        }
        
    private:
        unsigned int x_;
        unsigned int y_;
        unsigned int y_prev_;
        cv::Mat horizontal_( cv::Mat image, cv::Mat tile, unsigned int x, unsigned int y ) // todo: more reuse between horizontal_() and vertical_()
        {
            cv::Mat t;
            tile.copyTo( t );
            unsigned int overlap = x_ + tile.cols - x;
            tbb::parallel_for( tbb::blocked_range< std::size_t >( 0, tile.rows ), [&]( const tbb::blocked_range< std::size_t >& r )
            {
                for( unsigned int i = r.begin(); i < r.end(); ++i )
                {
                    for( unsigned int j = 0; j < overlap; ++j )
                    {
                        double ratio = double( j ) / overlap;
                        t.at< float >( i, j ) = t.at< float >( i, j ) * ratio + image.at< float >( y + i, x + j ) * ( 1 - ratio );
                    }
                }
            } );
            return t;
        }
        cv::Mat vertical_( cv::Mat image, cv::Mat tile, unsigned int x, unsigned int y )
        {
            cv::Mat t;
            tile.copyTo( t );
            unsigned int overlap = y_ + tile.rows - y;
            tbb::parallel_for( tbb::blocked_range< std::size_t >( 0, overlap ), [&]( const tbb::blocked_range< std::size_t >& r )
            {
                for( unsigned int i = r.begin(); i < r.end(); ++i )
                {
                    double ratio = double( i ) / overlap;
                    for( unsigned int j = 0; int( j ) < tile.cols; ++j )
                    {
                        t.at< float >( i, j ) = t.at< float >( i, j ) * ratio + image.at< float >( y + i, x + j ) * ( 1 - ratio );
                    }
                }
            } );
            return t;
        }
};

base* make( const std::string& how )
{
    if( how == "last" ) { return new last; }
    if( how == "linear" ) { return new linear; }
    std::cerr << "cv-calc: unstride: expected policy to handle overlaps, got: '" << how << "' (more policies: todo)" << std::endl;
    exit( 1 );
}
    
} } } // namespace snark { namespace unstride { namespace overlap {

int main( int ac, char** av )
{
    try
    {
        comma::command_line_options options( ac, av, usage );
        comma::csv::options csv( options );
        csv.full_xpath = true;
        verbose = options.exists("--verbose,-v");
        //std::vector< std::string > ops = options.unnamed("-h,--help,-v,--verbose,--flush,--input-fields,--input-format,--output-fields,--output-format,--show-partial", "--fields,--binary,--input,--output,--strides,--padding,--shape,--size,--kernel");
        std::vector< std::string > ops = options.unnamed("-h,--help,-v,--verbose,--flush,--header-fields,--header-format,--output-fields,--output-format,--exit-on-stability,--crop,--no-discard,--show-partial,--permissive,--deterministic,--fit-last,--output-number-of-strides,--number-of-strides", "-.*");
        if( ops.empty() ) { std::cerr << name << "please specify an operation." << std::endl; return 1;  }
        if( ops.size() > 1 ) { std::cerr << name << "please specify only one operation, got " << comma::join( ops, ' ' ) << std::endl; return 1; }
        std::string operation = ops.front();
        const snark::cv_mat::serialization::options input_parsed = comma::name_value::parser( ';', '=' ).get< snark::cv_mat::serialization::options >( options.value< std::string >( "--input", "" ) );
        snark::cv_mat::serialization::options input_options = handle_fields_and_format(csv, input_parsed );
        std::string output_options_string = options.value< std::string >( "--output", "" );
        snark::cv_mat::serialization::options output_options = output_options_string.empty() ? input_options : comma::name_value::parser( ';', '=' ).get< snark::cv_mat::serialization::options >( output_options_string );
        if( input_options.no_header && !output_options.fields.empty() && input_options.fields != output_options.fields )
        {
            if( output_options.fields != snark::cv_mat::serialization::header::default_fields() ) { std::cerr << "cv-calc: when --input has no-header option, --output fields can only be fields=" << snark::cv_mat::serialization::header::default_fields() << ", got: " << output_options.fields << std::endl; return 1; }
        }
        else
        {
            if( !output_options.fields.empty() && input_options.fields != output_options.fields ) { std::cerr << "cv-calc: customised output header fields not supported (todo); got: input fields: \"" << input_options.fields << "\" output fields: \"" << output_options.fields << "\"" << std::endl; return 1; }
        }
        if( output_options.fields.empty() ) { output_options.fields = input_options.fields; } // output fields and format will be empty when the user specifies only --output no-header or --output header-only
        if( !output_options.format.elements().empty() && input_options.format.string() != output_options.format.string() ) { std::cerr << "cv-calc: customised output header format not supported (todo); got: input format: \"" << input_options.format.string() << "\" output format: \"" << output_options.format.string() << "\"" << std::endl; return 1; }
        if( output_options.format.elements().empty() ) { output_options.format = input_options.format; };
        if( operation == "chessboard-corners")
        {
            if (options.exists("--output-fields")) { std::cout << comma::join(comma::csv::names<chessboard_corner_t>(), ',') << std::endl; return 0; }
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );

            const std::vector< std::string >& s = comma::split( options.value< std::string >( "--size" ), ',' );
            if( s.size() != 2 ) { std::cerr << "cv-calc: chessboard-corners: expected --size=<rows>,<cols>, got: \"" << options.value< std::string >( "--size" ) << std::endl; return 1; }
            comma::uint32 rows = boost::lexical_cast<comma::uint32>(s[0]);
            comma::uint32 cols = boost::lexical_cast<comma::uint32>(s[1]);
            cv::Size pattern_size(rows, cols);
            comma::uint32 block = 0;
            while( std::cin.good() )
            {
                auto p = input_serialization.read< boost::posix_time::ptime >( std::cin );
                if( p.second.empty() ) { break; }
                cv::Mat out;
                bool found = cv::findChessboardCorners(p.second, pattern_size, out);
                if (options.exists("--select"))
                {
                    if (found) { output_serialization.write_to_stdout(p, csv.flush ); }
                }
                else if (options.exists("--draw"))
                {
                    cv::drawChessboardCorners(p.second, pattern_size, out, found);
                    output_serialization.write_to_stdout(p, csv.flush );
                }
                else
                {
                    if (found || options.exists("--permissive"))
                    {
                        comma::csv::output_stream< chessboard_corner_t > output(std::cout, csv);
                        chessboard_corner_t corner;
                        corner.t = p.first;
                        corner.block = block;
                        for (comma::uint32 c = 0; int( c ) < out.rows; c++)
                        {
                            corner.row = c / rows;
                            corner.col = c % rows;
                            corner.position = Eigen::Vector2d(out.row(c).at<float>(0), out.row(c).at<float>(1));
                            output.write(corner);
                        }
                        ++block;
                    }
                    else
                    {
                        if ( verbose ) { std::cerr << "chessboard not detected in frame " << boost::posix_time::to_iso_string(p.first) << ", found " << out.rows << " of " << rows * cols << " points" << std::endl; }
                    }
                }
            }
            return 0;
        }
        if( operation == "crop-random" || operation == "roi-random" || operation == "random-crop" || operation == "random-roi" )
        {
            bool permissive = options.exists( "--permissive" );
            unsigned int count = options.value( "--count", 1 );
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            options.assert_mutually_exclusive( "--size", "--width,--height" );
            boost::optional< unsigned int > width = options.optional< unsigned int >( "--width" );
            boost::optional< unsigned int > height = options.optional< unsigned int >( "--height" );
            if( !width || !height )
            {
                const std::vector< std::string >& v = comma::split( options.value< std::string >( "--size" ), ',' );
                if( v.size() != 2 ) { std::cerr << "cv-calc: expected --size=<width>,<height>, got: '" << comma::join( v, ',' ) << std::endl; return 1; }
                width = boost::lexical_cast< unsigned int >( v[0] );
                height = boost::lexical_cast< unsigned int >( v[1] );
            }
            const std::vector< std::string >& v = comma::split( options.value< std::string >( "--padding", "0,0" ), ',' );
            if( v.size() != 2 ) { std::cerr << "cv-calc: expected --size=<width>,<height>, got: '" << comma::join( v, ',' ) << std::endl; return 1; }
            unsigned int padding_x = boost::lexical_cast< unsigned int >( v[0] );
            unsigned int padding_y = boost::lexical_cast< unsigned int >( v[1] );
            boost::optional< unsigned int > seed = options.optional< unsigned int >( "--seed" );
            if( !seed ) { seed = ( boost::posix_time::microsec_clock::universal_time() - boost::posix_time::from_iso_string( "20180101T000000" ) ).total_microseconds(); } // quick and dirty, use std::chrono instead?
            std::default_random_engine generator( *seed );
            std::uniform_real_distribution< float > distribution( 0, 1 );
            while( std::cin.good() && !std::cin.eof() )
            {
                std::pair< boost::posix_time::ptime, cv::Mat > p = input_serialization.read< boost::posix_time::ptime >( std::cin );
                if( p.second.empty() ) { return 0; }
                if( p.second.cols < int( *width + 2 * padding_x ) ) { std::cerr << "cv-calc: " << ( permissive ? "warning: " : "" ) << " expected image width at least " << ( *width + 2 * padding_x ) << " got: " << p.second.cols << std::endl; if( !permissive ) { return 1; } }
                if( p.second.rows < int( *height + 2 * padding_y ) ) { std::cerr << "cv-calc: " << ( permissive ? "warning: " : "" ) << " expected image height at least " << ( *height + 2 * padding_y ) << " got: " << p.second.rows << std::endl; if( !permissive ) { return 1; } }
                for( unsigned int i = 0; i < count; ++i )
                {
                    unsigned int x = padding_x + distribution( generator ) * ( p.second.cols - padding_x - *width );
                    unsigned int y = padding_y + distribution( generator ) * ( p.second.rows - padding_y - *height );
                    cv::Mat m;
                    p.second( cv::Rect( x, y, *width, *height ) ).copyTo( m );
                    output_serialization.write_to_stdout( std::make_pair( p.first, m ) );
                }
                std::cout.flush();
            }
            return 0;
        }
        if( operation == "draw" )
        {
            draw::shapes sample( options );
            input_options.fields = comma::join( comma::csv::names< draw::shapes::header_shapes >( input_options.fields, true, sample.hdr_shapes ), ',' );
            output_options.fields = comma::join( comma::csv::names< draw::shapes::header_shapes >( output_options.fields, true, sample.hdr_shapes ), ',' );
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            auto binary = csv.binary()
                ? comma::csv::binary< draw::shapes::header_shapes >( csv, sample.hdr_shapes )
                : comma::csv::binary< draw::shapes::header_shapes >(); // todo: sample seems not to take effect; debug

            while( std::cin.good() )
            {
                auto p = input_serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                if( p.second.empty() ) { break; }
                draw::shapes::header_shapes hdr_shapes = sample.hdr_shapes; // todo: quick and dirty; tear down, once todo above is sorted
                binary.get( hdr_shapes, &input_serialization.header_buffer()[0] );
                hdr_shapes.draw( p.second );
                sample.strm_shapes.draw( p.second, input_serialization.get_header( &input_serialization.header_buffer()[0] ).timestamp );
                output_serialization.write_to_stdout( p, csv.flush );
            }
            return 0;
        }
        if( operation == "grep" )
        {
            // Need to be created inside, some operation (roi) has other default fields. If not using --binary also requires --fields
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            grep::non_zero non_zero( options.value< std::string >( "--non-zero", "" ) );
            const std::vector< snark::cv_mat::filter >& filters = snark::cv_mat::filters::make( options.value< std::string >( "--filter,--filters", "" ) );
            if( !non_zero && !filters.empty() ) { std::cerr << "cv-calc: grep: warning: --filters specified, but --non-zero is not; --filters will have no effect" << std::endl; }
            while( std::cin.good() && !std::cin.eof() )
            {
                std::pair< boost::posix_time::ptime, cv::Mat > p = input_serialization.read< boost::posix_time::ptime >( std::cin );
                if( p.second.empty() ) { return 0; }
                std::pair< boost::posix_time::ptime, cv::Mat > filtered;
                if( filters.empty() ) { filtered = p; } else { p.second.copyTo( filtered.second ); }
                for( auto& filter: filters ) { filtered = filter( filtered ); }
                non_zero.size( filtered.second.rows * filtered.second.cols );
                if( non_zero.keep( filtered.second ) ) { output_serialization.write_to_stdout( p ); }
                std::cout.flush();
            }
            return 0;
        }
        if( operation == "life" )
        {
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            double procreation_threshold = options.value( "--procreation-threshold,--procreation", 3.0 );
            double stability_threshold = options.value( "--stability-threshold,--stability", 4.0 );
            double step = options.value( "--step", 1.0 );
            bool exit_on_stability = options.exists( "--exit-on-stability" );
            snark::cv_mat::impl::life< boost::posix_time::ptime > life( procreation_threshold, stability_threshold, step, exit_on_stability );
            auto iterations = options.optional< unsigned int >( "--iterations,-i" );
            std::pair< boost::posix_time::ptime, cv::Mat > p = input_serialization.read< boost::posix_time::ptime >( std::cin );
            if( p.second.empty() ) { return 0; }
            for( unsigned int i = 0; ( !iterations || i < *iterations ) && std::cout.good(); ++i )
            {
                output_serialization.write_to_stdout( life( p ) ); // todo: decouple step from output
                std::cout.flush();
            }
            return 0;
        }
        if( operation == "histogram" )
        {
            if( options.exists("--output-fields") ) { std::cout << "t,rows,cols,type,histogram" << std::endl;  exit(0); }
            if( options.exists("--output-format") ) { std::cout << "t,3ui,d,256ui" << std::endl;  exit(0); }
            snark::cv_mat::serialization serialization( input_options );
            while( std::cin.good() && !std::cin.eof() )
            {
                std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                if( p.second.empty() ) { return 0; }
                if( p.second.type() != CV_8UC1 && p.second.type() != CV_8UC2 && p.second.type() != CV_8UC3 && p.second.type() != CV_8UC4 ) { std::cerr << "cv-calc: histogram: expected an unsigned char image type; got type: " << p.second.type() << std::endl; exit( 1 ); }
                typedef boost::array< comma::uint32, 256 > channel_t;
                std::vector< channel_t > channels( p.second.channels() );
                for( unsigned int i = 0; i < channels.size(); ++i ) { ::memset( ( char* )( &channels[i][0] ), 0, sizeof( comma::uint32 ) * 256 ); }
                cv::Mat mat = p.second;
                for( int r = 0; r < p.second.rows; ++r )
                {
                    const unsigned char* m = mat.ptr< unsigned char >( r );
                    for( int c = 0; c < mat.cols; ++c ) { for( unsigned int i = 0; i < channels.size(); ++channels[i][*m], ++i, ++m ); }
                }
                if( !serialization.no_header() ) { std::cout.write( &serialization.header_buffer()[0], serialization.header_buffer().size() ); }
                for( unsigned int i = 0; i < channels.size(); ++i ) { std::cout.write( ( char* )( &channels[i][0] ), sizeof( comma::uint32 ) * 256 ); }
                std::cout.flush();
            }
            return 0;
        }
        if( operation == "mean" )
        {
            if( options.exists("--output-fields") ) { std::cout << "t,rows,cols,type,mean,count" << std::endl;  exit(0); }
            if( options.exists("--output-format") ) { std::cout << "t,3ui,d,ui" << std::endl;  exit(0); }
            auto threshold = options.optional< double >("--threshold");
            snark::cv_mat::serialization serialization( input_options );
            while( std::cin.good() && !std::cin.eof() )
            {

                std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                if( p.second.empty() ) { return 0; }

                cv::Mat mask;
                comma::uint32 count = p.second.rows * p.second.cols;
                if( threshold )
                {
                    cv::threshold(p.second, mask, *threshold, 255, cv::THRESH_BINARY);
                    if( mask.type() != CV_8U )
                    {
                        cv::Mat swap;
                        mask.convertTo( swap, CV_8U );
                        mask=swap;
                    }
                    count = cv::countNonZero(mask);
                }
                cv::Scalar mean = cv::mean( p.second, !threshold ? cv::noArray() : mask );
                std::cout.write( &serialization.header_buffer()[0], serialization.header_buffer().size() );
                for( int i = 0; i < p.second.channels(); ++i ) { std::cout.write( reinterpret_cast< char* >( &mean[i] ), sizeof( double ) ); std::cout.write( reinterpret_cast< char* >( &count ), sizeof( comma::uint32 ) ); }
                std::cout.flush();
            }
            return 0;
        }
        if( operation == "stride" )
        {
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            bool fit_last = options.exists( "--fit-last" );
            const std::vector< std::string >& strides_vector = comma::split( options.value< std::string >( "--strides", "1,1" ), ',' );
            if( strides_vector.size() != 2 ) { std::cerr << "cv-calc: stride: expected strides as <x>,<y>, got: \"" << options.value< std::string >( "--strides" ) << std::endl; return 1; }
            std::pair< unsigned int, unsigned int > strides( boost::lexical_cast< unsigned int >( strides_vector[0] ), boost::lexical_cast< unsigned int >( strides_vector[1] ) );
            const std::vector< std::string >& shape_vector = comma::split( options.value< std::string >( "--shape,--size,--kernel" ), ',' );
            if( shape_vector.size() != 2 ) { std::cerr << "cv-calc: stride: expected shape as <x>,<y>, got: \"" << options.value< std::string >( "--shape,--size,--kernel" ) << std::endl; return 1; }
            std::pair< unsigned int, unsigned int > shape( boost::lexical_cast< unsigned int >( shape_vector[0] ), boost::lexical_cast< unsigned int >( shape_vector[1] ) );
            unsigned int shape_size = shape.first * shape.second;
            struct padding_types { enum values { same, valid }; };
            std::string padding_string = options.value< std::string >( "--padding", "valid" );
            padding_types::values padding = padding_types::same;
            if( padding_string == "same" || padding_string == "SAME" ) { padding = padding_types::same; std::cerr << "cv-calc: stride: padding 'same' not implemented; please use --padding=valid" << std::endl; return 1; }
            else if( padding_string == "valid" || padding_string == "VALID" ) { padding = padding_types::valid; }
            else { std::cerr << "cv-calc: stride: expected padding type, got: \"" << padding_string << "\"" << std::endl; return 1; }
            grep::non_zero non_zero( options.value< std::string >( "--non-zero", "" ) );
            typedef snark::cv_mat::serialization::header::buffer_t first_t; // typedef boost::posix_time::ptime first_t;
            typedef std::pair< first_t, cv::Mat > pair_t;
            typedef snark::cv_mat::filter_with_header filter_t; // typedef snark::cv_mat::filter filter_t;
            typedef snark::cv_mat::filters_with_header filters_t; // typedef snark::cv_mat::filters filters_t;
            const comma::csv::binary< snark::cv_mat::serialization::header >* binary = input_serialization.header_binary();
            auto get_timestamp_from_header = [&]( const snark::cv_mat::serialization::header::buffer_t& h )->boost::posix_time::ptime
            {
                if( h.empty() || !binary ) { return boost::posix_time::not_a_date_time; }
                snark::cv_mat::serialization::header d;
                return binary->get( d, &h[0] ).timestamp;
            };
            const std::vector< filter_t >& filters = filters_t::make( options.value< std::string >( "--filter,--filters", "" ), get_timestamp_from_header );
            if( !non_zero && !filters.empty() ) { std::cerr << "cv-calc: stride: warning: --filters specified, but --non-zero is not; --filters will have no effect" << std::endl; }
            while( std::cin.good() && !std::cin.eof() )
            {
                pair_t p = input_serialization.read< first_t >( std::cin );
                if( p.second.empty() ) { return 0; }
                pair_t filtered;
                if( !filters.empty() )
                {
                    p.second.copyTo( filtered.second );
                    for( auto& filter: filters ) { filtered = filter( filtered ); }
                    if( filtered.second.rows != p.second.rows || filtered.second.cols != p.second.cols ) { std::cerr << "cv-calc: stride: expected original and filtered images of the same size, got " << p.second.rows << "," << p.second.cols << " vs " << filtered.second.rows << "," << filtered.second.cols << std::endl; return 1; }
                }
                switch( padding )
                {
                    case padding_types::same: // todo
                        break;
                    case padding_types::valid:
                    {
                        if( p.second.cols < int( shape.first ) || p.second.rows < int( shape.second ) ) { std::cerr << "cv-calc: stride: expected image greater than rows: " << shape.second << " cols: " << shape.first << "; got rows: " << p.second.rows << " cols: " << p.second.cols << std::endl; return 1; }
                        pair_t q;
                        q.first = p.first;
                        bool is_last_row = false;
                        for( unsigned int j = 0; !is_last_row; j += strides.second )
                        {
                            if( fit_last && int( j ) < p.second.rows && j > p.second.rows - shape.second ) { j = p.second.rows - shape.second; is_last_row = true; }
                            if( j >= ( p.second.rows + 1 - shape.second ) ) { break; }
                            bool is_last_col = false;
                            for( unsigned int i = 0; !is_last_col; i += strides.first )
                            {
                                if( fit_last && int( i ) < p.second.cols && i > p.second.cols - shape.first ) { i = p.second.cols - shape.first; is_last_col = true; }
                                if( i >= ( p.second.cols + 1 - shape.first ) ) { break; }
                                if( !filtered.second.empty() )
                                {
                                    filtered.second( cv::Rect( i, j, shape.first, shape.second ) ).copyTo( q.second );
                                    non_zero.size( shape_size );
                                    if( !non_zero.keep( q.second ) ) { continue; }
                                }
                                p.second( cv::Rect( i, j, shape.first, shape.second ) ).copyTo( q.second );
                                output_serialization.write_to_stdout( q, csv.flush );
                            }
                        }
                        break;
                    }
                }
            }
            return 0;
        }
        if( operation == "unstride" )
        {
            boost::scoped_ptr< snark::unstride::overlap::base > overlap( snark::unstride::overlap::make( options.value< std::string >( "--how-to-handle-overlaps,--how", "last" ) ) );
            bool fit_last = options.exists( "--fit-last" );
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            const std::vector< std::string >& unstrided_vector = comma::split( options.value< std::string >( "--unstrided-size,--unstrided" ), ',' );
            if( unstrided_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected --unstrided-size as <width>,<height>, got: \"" << options.value< std::string >( "--unstrided-size,--unstrided" ) << std::endl; return 1; }
            cv::Point2i unstrided( boost::lexical_cast< unsigned int >( unstrided_vector[0] ), boost::lexical_cast< unsigned int >( unstrided_vector[1] ) );
            const std::vector< std::string >& strides_vector = comma::split( options.value< std::string >( "--strides", "1,1" ), ',' );
            if( strides_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected strides as <x>,<y>, got: \"" << options.value< std::string >( "--strides" ) << std::endl; return 1; }
            cv::Point2i strides( boost::lexical_cast< unsigned int >( strides_vector[0] ), boost::lexical_cast< unsigned int >( strides_vector[1] ) );
            const std::vector< std::string >& shape_vector = comma::split( options.value< std::string >( "--shape,--size,--kernel" ), ',' );
            if( shape_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected shape as <x>,<y>, got: \"" << options.value< std::string >( "--shape,--size,--kernel" ) << std::endl; return 1; }
            cv::Point2i shape( boost::lexical_cast< unsigned int >( shape_vector[0] ), boost::lexical_cast< unsigned int >( shape_vector[1] ) );
            unsigned int stride_rows = ( unstrided.y - shape.y ) / strides.y + 1;
            unsigned int stride_cols = ( unstrided.x - shape.x ) / strides.x + 1;
            if( fit_last )
            {
                if( int( stride_rows - 1 ) * strides.y + shape.y < unstrided.y ) { ++stride_rows; }
                if( int( stride_cols - 1 ) * strides.x + shape.x < unstrided.x ) { ++stride_cols; }
            }
            if( options.exists( "--output-number-of-strides,--number-of-strides" ) ) { std::cout << stride_cols << "," << stride_rows << std::endl; exit( 0 ); }
            typedef std::pair< boost::posix_time::ptime, cv::Mat > pair_t;
            pair_t output;
            unsigned int ix = 0;
            unsigned int iy = 0;
            while( std::cin.good() && !std::cin.eof() )
            {
                pair_t p = input_serialization.read< boost::posix_time::ptime >( std::cin );
                if( p.second.empty() ) { return 0; }
                if( output.second.empty() ) { output.second = cv::Mat( unstrided.y, unstrided.x, p.second.type() ); }
                if( output.second.type() != p.second.type() ) { std::cerr << "cv-calc: unstride: expected input image type " << output.second.type() << "; got: " << p.second.type() << std::endl; exit( 1 ); }
                if( p.second.rows != shape.y ) { std::cerr << "cv-calc: unstride: expected input image with " << shape.y << "; got: " << p.second.rows << std::endl; exit( 1 ); }
                if( p.second.cols != shape.x ) { std::cerr << "cv-calc: unstride: expected input image with " << shape.x << "; got: " << p.second.cols << std::endl; exit( 1 ); }
                if( ix == 0 && iy == 0 ) { output.first = p.first; }
                unsigned int x = ix * strides.x;
                unsigned int y = iy * strides.y;
                if( fit_last )
                {
                    if( ix + 1 == stride_cols ) { x = unstrided.x - shape.x; }
                    if( iy + 1 == stride_rows ) { y = unstrided.y - shape.y; }
                }
                overlap->append( output.second, p.second, x, y );
                ++ix;
                if( ix >= stride_cols )
                {
                    ix = 0;
                    ++iy;
                    if( iy >= stride_rows ) { iy = 0; }
                }
                if( ix == 0 && iy == 0 ) { output_serialization.write_to_stdout( output, true ); }
            }
            return 0;
        }
        if( operation == "thin" )
        {
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            options.assert_mutually_exclusive( "--rate,--to-fps,--fps" );
            options.assert_mutually_exclusive( "--deterministic,--to-fps,--fps" );
            if( !options.exists( "--rate" ) && !options.exists( "--to-fps,--fps" ) ) { std::cerr << "cv-calc: thin: please specify either --rate or --to-fps" << std::endl; }
            bool deterministic = options.exists( "--to-fps,--fps" ) || options.exists( "--deterministic" );
            double rate = options.exists( "--rate" ) ? options.value< double >( "--rate" ) : options.value< double >( "--to-fps,--fps" ) / options.value< double >( "--from-fps" );
            thin::keep keep( rate, deterministic );
            while( std::cin.good() && !std::cin.eof() )
            {
                std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = input_serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                if( p.second.empty() ) { return 0; }
                if( keep ) { output_serialization.write_to_stdout( p ); std::cout.flush(); }
            }
            return 0;
        }
        if( operation == "unstride-positions" ) // TODO move to another utility since it doesn't operate on an image ?
        {
            stride_positions_t input( options );
            positions_t output( options );

            if( options.exists("--input-fields") ) { std::cout << comma::join( comma::csv::names< stride_positions_t >( true, input ), ',' ) << std::endl; return 0; }
            if( options.exists("--input-format") ) { std::cout << comma::csv::format::value< stride_positions_t >( "", true, input ) << std::endl; return 0; }
            if( options.exists("--output-fields") ) { std::cout << comma::join( comma::csv::names< positions_t >( true, output ), ',' ) << std::endl; return 0; }
            if( options.exists("--output-format") ) { std::cout << comma::csv::format::value< positions_t >( "", true, output ) << std::endl; return 0; }

            const std::vector< std::string >& unstrided_vector = comma::split( options.value< std::string >( "--unstrided-size,--unstrided" ), ',' );
            if( unstrided_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected --unstrided-size as <width>,<height>, got: \"" << options.value< std::string >( "--unstrided-size,--unstrided" ) << std::endl; return 1; }
            cv::Point2i unstrided( boost::lexical_cast< unsigned int >( unstrided_vector[0] ), boost::lexical_cast< unsigned int >( unstrided_vector[1] ) );
            const std::vector< std::string >& strides_vector = comma::split( options.value< std::string >( "--strides", "1,1" ), ',' );
            if( strides_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected strides as <x>,<y>, got: \"" << options.value< std::string >( "--strides" ) << std::endl; return 1; }
            cv::Point2i strides( boost::lexical_cast< unsigned int >( strides_vector[0] ), boost::lexical_cast< unsigned int >( strides_vector[1] ) );
            const std::vector< std::string >& shape_vector = comma::split( options.value< std::string >( "--shape,--size,--kernel" ), ',' );
            if( shape_vector.size() != 2 ) { std::cerr << "cv-calc: unstride-positions: expected shape as <x>,<y>, got: \"" << options.value< std::string >( "--shape,--size,--kernel" ) << std::endl; return 1; }
            cv::Point2i shape( boost::lexical_cast< unsigned int >( shape_vector[0] ), boost::lexical_cast< unsigned int >( shape_vector[1] ) );

            // TODO --padding
            unsigned int stride_rows = ( unstrided.y - shape.y ) / strides.y + 1;
            unsigned int stride_cols = ( unstrided.x - shape.x ) / strides.x + 1;
            unsigned int num_strides = stride_rows * stride_cols;
            if( verbose ) { std::cerr << name << "unstride-positions: stride rows: " << stride_rows << " stride cols: " << stride_cols << " number of strides: " << num_strides << std::endl; }

            comma::csv::options icsv( options );
            icsv.full_xpath = true;
            comma::csv::input_stream< stride_positions_t > is( std::cin, icsv, input );
            comma::csv::options ocsv;
            ocsv.fields = comma::join( comma::csv::names< positions_t >( "positions", true, output ), ',' );
            ocsv.flush = true;
            ocsv.full_xpath = true;
            if( options.exists( "--binary" ) ) { ocsv.format( comma::csv::format::value< positions_t >( ocsv.fields, true, output ) ); }
            comma::csv::output_stream< positions_t > os( std::cout, ocsv, output );
            comma::csv::tied< stride_positions_t, positions_t > tied( is, os );
            bool permissive = options.exists( "--permissive" );

            while( is.ready() || std::cin.good() )
            {
                const stride_positions_t* p = is.read();
                if( !p ) { break; }
                if( p->index >= num_strides )
                { 
                    if( permissive )
                    { 
                        if( verbose ) { std::cerr << "cv-calc: unstride-positions: expected stride index less than : " << num_strides << "; got: " << p->index << "; discarded" << std::endl; }
                        continue;
                    }
                    else
                    { 
                        std::cerr << "cv-calc: unstride-positions: expected stride index less than : " << num_strides << "; got: " << p->index << "; use --permissive to discard" << std::endl;
                        return 1;
                    }
                }
                unsigned int stride_col = p->index % stride_cols;
                unsigned int stride_row = p->index / stride_cols;
                cv::Point2d offset( stride_col * strides.x, stride_row * strides.y );
                for( unsigned int i = 0; i < p->positions.size(); ++i ) { output.positions[i] = p->positions[i] + offset; }
                tied.append( output );
            }
            return 0;
        }
        if( operation == "header" )
        {
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );

            if( options.exists("--header-fields") ) { std::cout << "t,rows,cols,type" << std::endl;  exit(0); }
            if( options.exists("--header-format") ) { std::cout << "t,3ui" << std::endl;  exit(0); }
            if( verbose ) { std::cerr << name << "fields: " << input_options.fields << std::endl; std::cerr << name << "format: " << input_options.format.string() << std::endl; }
            if( options.exists("--output-fields") ) { std::cout << "rows,cols,type" << std::endl;  return 0; }

            snark::cv_mat::serialization serialization( input_options );
            std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = serialization.read< snark::cv_mat::serialization::header::buffer_t >(std::cin);
            if( p.second.empty() ) { std::cerr << name << "failed to read input stream" << std::endl; return 1; }
            comma::csv::options out;
            out.fields = "rows,cols,type";
            comma::csv::output_stream< snark::cv_mat::serialization::header > ascii( std::cout, out );
            ascii.write( serialization.get_header( &serialization.header_buffer()[0] ) );
            return 0;
        }
        if( operation == "format" )
        {
            snark::cv_mat::serialization input_serialization( input_options );
            snark::cv_mat::serialization output_serialization( output_options );
            if( options.exists("--header-fields") ) { std::cout << "t,rows,cols,type" << std::endl;  exit(0); }
            if( options.exists("--header-format") ) { std::cout << "t,3ui" << std::endl;  exit(0); }
            if( verbose ) { std::cerr << name << "fields: " << input_options.fields << std::endl; std::cerr << name << "format: " << input_options.format.string() << std::endl; }

            snark::cv_mat::serialization serialization( input_options );
            std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = serialization.read< snark::cv_mat::serialization::header::buffer_t >(std::cin);
            if( p.second.empty() ) { std::cerr << name << "failed to read input stream" << std::endl; return 1; }
            snark::cv_mat::serialization::header header = serialization.get_header( &serialization.header_buffer()[0] );
            comma::csv::format format = input_options.format.elements().empty() ? comma::csv::format("t,3ui") : input_options.format ;
            format += "s[" + boost::lexical_cast<std::string>( comma::uint64(header.rows) * header.cols * p.second.elemSize() )  + "]";
            std::cout << format.string() << std::endl;
            return 0;
        }
        if( operation == "roi" )
        {
            // TODO: in order to do this, extents need to be built into serialisation::options
            if( input_options.no_header ) { std::cerr << "cv-calc: --input with no-header cannot be used with 'roi' operation, as roi extents is passed in with the header" << std::endl; return 1; }
            if( options.exists("--header-fields") ) { std::cout << comma::join( comma::csv::names<extents>(), ',' ) << "," << "t,rows,cols,type" << std::endl;  exit(0); }
            if( options.exists("--header-format") ) { std::cout << comma::csv::format::value<extents>() << "," << "t,3ui" << std::endl;  exit(0); }
            options.assert_mutually_exclusive( "--crop,--no-discard" );

            bool crop = options.exists( "--crop" );
            bool flush = options.exists("--flush");
            bool permissive = options.exists("--show-partial,--permissive");
            bool no_discard = options.exists( "--no-discard" );

            if( csv.has_some_of_paths( "rectangles" ) || options.exists( "--rectangles" ) )
            {
                roi::shapes sample( options );
                input_options.fields = comma::join( comma::csv::names< roi::shapes::header_shapes >( input_options.fields, true, sample.hdr_shapes ), ',' );
                output_options.fields = comma::join( comma::csv::names< roi::shapes::header_shapes >( output_options.fields, true, sample.hdr_shapes ), ',' );
                if( crop ) { output_options.fields = std::string(); output_options.format = comma::csv::format(); }// comma::join( comma::csv::names< snark::cv_mat::serialization::header >( output_options.fields, true ), ',' ); }

                snark::cv_mat::serialization input_serialization( input_options );
                snark::cv_mat::serialization output_serialization( output_options );
                auto binary = csv.binary()
                    ? comma::csv::binary< roi::shapes::header_shapes >( csv, sample.hdr_shapes )
                    : comma::csv::binary< roi::shapes::header_shapes >(); // todo: sample seems not to take effect; debug

                cv::Mat mask;
                comma::uint64 count = 0;

                while( std::cin.good() )
                {
                    auto p = input_serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                    if( p.second.empty() ) { break; }

                    cv::Mat& mat = p.second;
                    ++count;

                    roi::shapes::header_shapes hdr_shapes = sample.hdr_shapes; // todo: quick and dirty; tear down, once todo above is sorted
                    binary.get( hdr_shapes, &input_serialization.header_buffer()[0] );
                    auto timestamp = input_serialization.get_header( &input_serialization.header_buffer()[0] ).timestamp;
                    if( sample.strm_shapes.rectangles ) { sample.strm_shapes.rectangles->read_at( hdr_shapes.rectangles, timestamp ); }

                    if( mask.rows != mat.rows || mask.cols != mat.cols ) { mask = cv::Mat::ones( mat.rows, mat.cols, CV_8U ); }
                    auto do_discard = true;
                    for( auto& ext : hdr_shapes.rectangles )
                    {
                        auto result = ext.validate( mat.rows, mat.cols, permissive );

                        if( roi::status::error == result.first )
                        {
                            std::cerr << name << "roi's width and height can not be negative; failed on image/frame number " << count
                                << ", min: " << ext.min << ", max: " <<  ext.max << ", width: " << result.second.width << ", height: " << result.second.height << std::endl;
                            return 1;
                        }
                        if( roi::status::ignore == result.first ) { continue; }

                        if( crop )
                        {
                            cv::Mat cropped;
                            p.second( result.second ).copyTo( cropped );
                            output_serialization.write_to_stdout( std::pair< boost::posix_time::ptime, cv::Mat >( timestamp, cropped ) );
                        }
                        else
                        {
                            do_discard = false;
                            mask( result.second ) = cv::Scalar(0);
                        }

                    }
                    if( !crop )
                    {
                        if( !do_discard )
                        {
                            mat.setTo( cv::Scalar(0), mask );
                            //mask.setTo( cv::Scalar(1), mask ); //not working
                            mask = cv::Scalar( 1 );
                            output_serialization.write_to_stdout( p, csv.flush );
                        }
                        else if( no_discard )
                        {
                            mat.setTo( cv::Scalar(0) );
                            output_serialization.write_to_stdout( p, flush );
                        }
                    }
                }
            }
            else
            {
                if( input_options.fields.empty() ) { input_options.fields = "min/x,min/y,max/x,max/y,t,rows,cols,type"; }
                if( input_options.format.elements().empty() ) { input_options.format = comma::csv::format( "4i,t,3ui" ); }
                if( output_options_string.empty() ) { output_options = input_options; }
                if( verbose ) { std::cerr << name << "fields: " << input_options.fields << std::endl; std::cerr << name << "format: " << input_options.format.string() << std::endl; }

                snark::cv_mat::serialization input_serialization( input_options );
                snark::cv_mat::serialization output_serialization( output_options );

                csv.fields = input_options.fields;
                csv.format( input_options.format );
                comma::csv::binary< ::extents > binary( csv );
                ::extents ext;
                cv::Mat mask;
                comma::uint64 count = 0;
                while( std::cin.good() && !std::cin.eof() )
                {
                    std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat > p = input_serialization.read< snark::cv_mat::serialization::header::buffer_t >( std::cin );
                    if( p.second.empty() ) { break; }
                    cv::Mat& mat = p.second;
                    ++count;
                    binary.get( ext, &input_serialization.header_buffer()[0] );
                    if( mask.rows != mat.rows || mask.cols != mat.cols ) { mask = cv::Mat::ones( mat.rows, mat.cols, CV_8U ); }
                    if( ext.max.x < 0 || ext.min.x >= mat.cols || ext.max.y < 0 || ext.min.y >= mat.rows )
                    {
                        if( no_discard )
                        {
                            mat.setTo( cv::Scalar(0) );
                            output_serialization.write_to_stdout( p, flush );
                        }
                        continue;
                    }
                    if( permissive )
                    {
                        ext.min.x = std::max( int( ext.min.x ), 0 );
                        ext.min.y = std::max( int( ext.min.y ), 0 );
                        ext.max.x = std::min( int( ext.max.x ), mat.cols );
                        ext.max.y = std::min( int( ext.max.y ), mat.rows );
                    }
                    int width = ext.max.x - ext.min.x;
                    int height = ext.max.y - ext.min.y;
                    if( width < 0 || height < 0 ) { std::cerr << name << "roi's width and height can not be negative; failed on image/frame number " << count << ", min: " << ext.min << ", max: " << ext.max << ", width: " << width << ", height: " << height << std::endl; return 1; }
                    if( ext.min.x >= 0 && ext.min.y >=0 && ext.max.x <= mat.cols && ext.max.y <= mat.rows )
                    {
                        if( crop )
                        {
                            cv::Mat cropped;
                            mat( cv::Rect( ext.min.x, ext.min.y, width, height ) ).copyTo( cropped );
                            output_serialization.write_to_stdout( std::pair< snark::cv_mat::serialization::header::buffer_t, cv::Mat >( p.first, cropped ) );
                        }
                        else
                        {
                            mask( cv::Rect( ext.min.x, ext.min.y, width , height ) ) = cv::Scalar(0);
                            mat.setTo( cv::Scalar(0), mask );
                            mask( cv::Rect( ext.min.x, ext.min.y, width , height ) ) = cv::Scalar(1);
                            output_serialization.write_to_stdout( p, flush );
                        }
                    }
                }
            }
            return 0;
        }
        std::cerr << name << " unknown operation: '" << operation << "'" << std::endl;
    }
    catch( std::exception& ex ) { std::cerr << "cv-calc: " << ex.what() << std::endl; }
    catch( ... ) { std::cerr << "cv-calc: unknown exception" << std::endl; }
    return 1;
}
