/**
	Findpeaks: small library to find local extrema in 1D or 2D datasets
 	based on the persistent homology method.
	Ported Stefan Huber's code from Python: https://git.sthu.org/?p=persistence.git;hb=HEAD

    Copyright (C) 2022  University College London
	developed by: Balázs Dura-Kovács (b.dura-kovacs@ucl.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef FINDPEAK_FINDPEAKS_HPP
#define FINDPEAK_FINDPEAKS_HPP

namespace findpeaks {

typedef size_t linear_index_t; // identifies pixels

template <typename pixel_coordinate_type = size_t>
struct pixel_index_t {
	pixel_coordinate_type x;
	pixel_coordinate_type y;
};

// 3D
template <typename pixel_coordinate_type = size_t>
struct  pixel_index_3d {
    pixel_coordinate_type x;
    pixel_coordinate_type y;
    pixel_coordinate_type z;
};

/**
 * Converts linear index to 2d indices
 * *(m + linea;r_index) to m[x][y]
 * @param linear_index
 * @param width matrix size in x dimension (unused)
 * @param height matrix size in y dimension, ie m[WIDTH][HEIGHT]
 * @param x output
 * @param y output
 */
inline void index_1d_to_2d(linear_index_t linear_index, const size_t width, const size_t height, size_t &x, size_t &y){
	x = linear_index / height;
	y = linear_index % height;
}


// 3D
inline void index_1d_to_3d(linear_index_t linear_index, const size_t width, const size_t height, const size_t depth, size_t &x, size_t &y, size_t &z){
    x = linear_index / (height * depth);
    y = (linear_index / depth) % height;
    z = linear_index % depth;
}


/**
 * Converts linear index to 2d indices
 * *(m + linear_index) to m[x][y]
 * @param linear_index
 * @param width matrix size in x dimension (unused)
 * @param height matrix size in y dimension, ie m[WIDTH][HEIGHT]
 * @param x output
 * @param y output
 */
inline pixel_index_t<> index_1d_to_2d(linear_index_t linear_index, const size_t width, const size_t height){
	size_t x = linear_index / height;
	size_t y = linear_index % height;
	return {x, y};
}

// 3D
inline pixel_index_3d<> index_1d_to_3d(linear_index_t linear_index, const size_t width, const size_t height, const size_t depth){
    size_t x = linear_index / (height * depth);
    size_t y = (linear_index / depth) % height;
    size_t z = linear_index % depth;
    return {x, y, z};
}

/**
 * Converts 2d index to linear index
 * m[x][y] to *(m + linear_index]
 * @param x
 * @param y
 * @param width matrix size in x dimension (unused)
 * @param height matrix size in y dimension, ie m[WIDTH][HEIGHT]
 * @return
 */
inline linear_index_t index_2d_to_1d(const size_t x, const size_t y, const size_t width, const size_t height){
	return x*height + y;
}

// 3D
inline linear_index_t index_3d_to_1d(const size_t x, const size_t y, const size_t z, const size_t width, const size_t height, const size_t depth){
    return x * height * depth + y * depth + z;
}

inline linear_index_t index_2d_to_1d(const pixel_index_t<> p, const size_t width, const size_t height){
	return index_2d_to_1d(p.x, p.y, width, height);
}

// 3D
inline linear_index_t index_3d_to_1d(const pixel_index_3d<> p, const size_t width, const size_t height, const size_t depth){
    return index_3d_to_1d(p.x, p.y, p.z, width, height, depth);
}

// 2D
template <typename pixel_data_type>
struct image_t {
	const size_t width;
	const size_t height;
	const pixel_data_type * data;
	inline pixel_data_type get_pixel_value(size_t x, size_t y){
		return *(data + index_2d_to_1d(x, y, width, height));
	}
	inline pixel_data_type get_pixel_value(pixel_index_t<> pixel){
		return get_pixel_value(pixel.x, pixel.y);
	}
	inline pixel_data_type get_pixel_value(linear_index_t pixel){
		return *(data + pixel);
	}
};

// 3D
template <typename pixel_data_type>
struct volume_t {
    const size_t width;
    const size_t height;
    const size_t depth;
    const pixel_data_type * data;
    inline pixel_data_type get_pixel_value(size_t x, size_t y, size_t z){
        return *(data + index_3d_to_1d(x, y, z, width, height, depth));
    }
    inline pixel_data_type get_pixel_value(pixel_index_3d<> pixel){
        return get_pixel_value(pixel.x, pixel.y, pixel.z);
    }
    inline pixel_data_type get_pixel_value(linear_index_t pixel){
        return *(data + pixel);
    }
};

template <typename pixel_data_type, typename pixel_coordinate_type = size_t>
struct peak_t {
	pixel_data_type birth_level; //i.e. peak value
	pixel_data_type persistence;
	pixel_index_t<pixel_coordinate_type> birth_position;
	pixel_index_t<pixel_coordinate_type> death_position;
};

// 3D
template <typename pixel_data_type, typename pixel_coordinate_type = size_t>
struct peak_3d {
    pixel_data_type birth_level; //i.e. peak value
    pixel_data_type persistence;
    pixel_index_3d<pixel_coordinate_type> birth_position;
    pixel_index_3d<pixel_coordinate_type> death_position;
};

template <typename pixel_data_type>
struct pixel_t {
	pixel_data_type value;
	linear_index_t position;
};

// same as pixel_t, but with 2d position
template <typename pixel_data_type, typename pixel_coordinate_type = size_t>
struct pixel_t2 {
	pixel_data_type value;
	pixel_index_t<pixel_coordinate_type> position;
};

// 3D
template <typename pixel_data_type, typename pixel_coordinate_type = size_t>
struct pixel_t3 {
    pixel_data_type value;
    pixel_index_3d<pixel_coordinate_type> position;
};


enum extremum_t {minimum, maximum};


}

#endif //FINDPEAK_FINDPEAKS_HPP
