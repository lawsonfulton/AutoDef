// Copyright (C) 2009 Cornell University
// All rights reserved.
// Original Author: Steven An (http://www.cs.cornell.edu/~stevenan)

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#ifndef __STEVELIB_UTILS_H__
#define __STEVELIB_UTILS_H__

#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>


/*
	 Stuff in this library should depend on nothing but standard C/++ libraries!
 */
namespace SteveLib
{
	using namespace std;

	void bin_read( std::istream& s, int& dst );
	void bin_read( std::istream& s, double& dst );

	void bin_write( std::ostream& s, const int& dst );
	void bin_write( std::ostream& s, const double& dst );

	/**
	 * Useful for getting a globally unqiue number.
	 * Usage example:
	 *
	 * char timeBuf[32];
	 * char idBuf[128];
	 *
	 * sprintf( idBuf, "id #%s", raw_time_string(timeBuf) );
	 */
	char* raw_time_string( char* buf );

	// Returns a human-readable form of the given number of seconds.
	// Ie. it converts to the lowest unit of time (from days to seconds)
	// that keeps the figures greater than one.
	std::string human_time( float secs );

	/**
	 * Returns the current working directory.
	 * WARNING: This returns a pointer to a global buffer,
	 * so if you want to keep it, make your own copy.
	 */
	const char* cwd();

	int get_files_in_dir( string dir_path, vector<string>& files );

	/**
	 * Use this to enforce run-time checks for app bugs.
	 */
	void halt_if_false( bool success, const char* onfail );

	bool ends_with( const string& s, const string& end );

	bool begins_with( const string& s, const string& begin );

	// This splits 's' into parts delimited by any character in 'splitters' and returns the i-th part/substr.
	// i is base 0
	string get_split_part( const string& s, const string& splitters, int i );

	vector<string> split( const string& s, const string& splitters );

	// returns the number of times that any char in 'chars' occurs in 's'
	int count_occurs( const string& s, const string& chars );

	/**
	 * Displays prompt to the user, and adds "(press ENTER for default) >> ".
	 * If the user enters blank, s will be set to "def" and return false.
	 * Otherwise, s contains the user's input, and returns true.
	 */
	bool prompt_with_default( string& s, const string& def, const char* prompt );

	time_t seed_random_generator_with_time();

	inline double clock2secs( clock_t tics )
	{
		return (double)tics / (double)CLOCKS_PER_SEC;
	}

	//----------------------------------------
	//  Returns a string with the human-readable date-time suitable for filenames  
	//----------------------------------------
	string datetime_for_filename();

	//----------------------------------------
	//  Tries to create a directory of the given path.
	//  Returns false if the directory already existed.
	//----------------------------------------
	enum CREATE_DIR_RESULT { OK, EXISTS, FAIL };

	enum CREATE_DIR_RESULT create_dir( const string& name );

	//----------------------------------------
	//  Platform independent file listing.
	//  2008-02-02 (17-08) - only impl'd for WIN32 so far
	//----------------------------------------
	int collect_files_with_extension( string dir, string ext, vector<string>& filesOut );

	//----------------------------------------
	//  Ex: dir_for_path("a/b/c/f.exe") --> "a/b/c/"
	//----------------------------------------
	std::string dir_for_path( char* path );

	//----------------------------------------
	//  Same as normal ctime, but removes the trailing \n
	//----------------------------------------
	char* myctime( time_t* t );

	std::string short_hostname();

	template < typename T > bool vector_contains( vector<T> vec, T val ) {
		bool contains = false;
		for( int i = 0; i < vec.size(); i++ )
		{
			if( vec[i] == val ) {
				contains = true;
				break;
			}
		}

		return contains;
	}

	template < typename T > void print_vector( vector<T>& vec, string sep ) {
		for( int i = 0; i < vec.size(); i++ ) {
			cout << i << ":" << vec[i] << sep;
		}
	}

	template < typename T > int find_max( T* array, int size, T& maxVal ) {
		int maxIdx = 0;
		maxVal = array[0];
		for( int i = 1; i < size; i++ ) {
			if( array[i] > maxVal ) {
				maxVal = array[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}

	template < typename T > int find_max( T* array, int size ) {
		T maxVal;
		int maxIdx = 0;
		maxVal = array[0];
		for( int i = 1; i < size; i++ ) {
			if( array[i] > maxVal ) {
				maxVal = array[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}

	template < typename T > T* set_to_new_array( set<T>& s ) {
		T* arr = new T[ s.size() ];
		copy( s.begin(), s.end(), arr );
		return arr;
	}

	//----------------------------------------
	//  Not very accurate.
	//----------------------------------------
	class StopWatch
	{
		public:

			StopWatch() :
				totalTics( 0 ),
				inCycle( false ),
				ticTime( 0 )
		{
		}

			void tic()
			{
				if( inCycle )
				{
					cerr << "** WARNING: StopWatch::tic was called again before toc was called!" << endl;
				}

				inCycle = true;
				ticTime = clock();
			}

			//----------------------------------------
			//  Returns the time of this last tic/toc cycle.
			//----------------------------------------
			double toc()
			{
				if( !inCycle )
				{
					cerr << "** WARNING: StopWatch::toc called before tic was called! Returning bogus time!" << endl;
					return -1.0;
				}

				inCycle = false;
				clock_t now = clock();
				clock_t cycle = (now - ticTime);

				// accumulate
				totalTics += cycle;

				return clock2secs( cycle );
			}

			double getTotalSecs()
			{
				return clock2secs( totalTics );
			}

			void reset()
			{
				if( inCycle )
				{
					cerr << "** WARNING: StopWatch::reset was called between a tic/toc cycle!" << endl;
				}
				totalTics = 0;
			}

		private:

			clock_t totalTics;

			bool inCycle;
			clock_t ticTime;
	};
}

// useful macros

#define NULL_SAFE_DELETE(x) {if( x != NULL ) { delete x; x = NULL; }}

// excuse the name..
// to be used like: cout << SDUMP(x+y) << SDUMP(foo->bar) << endl;
#define SDUMP(x)	" " << #x << "=[ " << x << " ] "
#define LDUMP(x)	" " << #x << "=[ " << endl << x << " ] "

#define __FL__	__FILE__ ":" __LINE__

// for libconfig++
#define LOAD_CFG_VAR( cfg, var, defval ) { if( (cfg).exists(#var) ) cfg.lookupValue( #var, var ); else var = (defval); }

#endif
