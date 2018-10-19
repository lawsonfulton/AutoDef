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

#include "steve.h"

#include <ctime>
#include <time.h>
#include <stdlib.h>// for _MAX_PATH
#include <stdio.h>
#include <string.h>
#include <sstream>

#if defined(__unix__) || defined (__LINUX__)
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#endif

#ifdef WIN32
#include <windows.h>
#endif

using namespace std;
using namespace SteveLib;

void SteveLib::bin_read( istream& s, int& dst ) { s.read( (char*)&dst, sizeof(int) ); }
void SteveLib::bin_read( istream& s, double& dst ) { s.read( (char*)&dst, sizeof(double) ); }

void SteveLib::bin_write( ostream& s, const int& dst ) { s.write( (char*)&dst, sizeof(int) ); }
void SteveLib::bin_write( ostream& s, const double& dst ) { s.write( (char*)&dst, sizeof(double) ); }

char* SteveLib::raw_time_string( char* buf )
{
	time_t rawTime;
	time( &rawTime );
	sprintf( buf, "%f", rawTime );
	return buf;
}

#ifdef _WIN32
#include <direct.h> // for _getcwd
static char CWD_BUFFER[_MAX_PATH];

const char* SteveLib::cwd()
{
	_getcwd( CWD_BUFFER, _MAX_PATH );
	return CWD_BUFFER;
}
#else
static char CWD_BUFFER[1024];

const char* SteveLib::cwd()
{
	getcwd( CWD_BUFFER, 1024 );
	return CWD_BUFFER;
}
#endif

void SteveLib::halt_if_false( bool success, const char* onfail )
{
	if( !success )
	{
		cerr << onfail << endl;
		char dummy;
		cin >> dummy;
	}
}

int SteveLib::get_files_in_dir( string dir_path, vector<string>& files )
{
	halt_if_false( false, "ERROR ID 2007-10-30 (14-48)" );
	return 1;
}

#if 0
int SteveLib::get_files_in_dir( string dir_path, vector<string>& files )
{
    DIR *dir = NULL;
    struct dirent *dirp = NULL;

	dir = opendir( dir_path.c_str() );
    if( dir == NULL )
	{
        cout << "Error(" << errno << ") opening " << dir_path << endl;
        return errno;
    }
	else
	{
		while( (dirp = readdir(dir)) != NULL )
		{
			files.push_back(string(dirp->d_name));
		}
		closedir( dir );
		return 0;
	}
}
#endif

bool SteveLib::ends_with( const string& s, const string& end )
{
	int sl = s.length();
	int el = end.length();

	if( sl < el ) return false;

	bool endsWith = true;

	for( int i = 0; i < end.length(); i++ )
	{
		if( s[ sl-1 - i ] != end[ el-1 - i ] )
		{
			endsWith = false;
		}
	}

	return endsWith;
}

bool SteveLib::prompt_with_default( string& s, const string& def, const char* prompt )
{
	cout << prompt << " (enter nothing for default) >> ";
	getline( cin, s );

	if( s.length() == 0 )
	{
		cout << "Using default: \"" << def << "\"" << endl;
		s = def;
		return false;
	}
	else
	{
		return true;
	}
}

bool SteveLib::begins_with( const string& s, const string& begin )
{
	if( s.length() < begin.length() ) return false;

	for( int i = 0; i < begin.length(); i++ )
	{
		if( s[i] != begin[i] ) return false;
	}

	return true;
}

string SteveLib::get_split_part( const string& s, const string& splitters, int i )
{
	int start = 0;
	
	while( i != 0 )
	{
		int occur = s.find_first_of( splitters, start );

		if( occur == string::npos )
		{
			cerr << "couldn't find any occurrance of '" << splitters << "' in '" << s << "'" << endl;
			return string("");
		}

		start = occur + 1;
		i--;
	}

	unsigned long len = 0;
	unsigned long occur = s.find_first_of( splitters, start );
	if( occur == string::npos )
	{
		len = string::npos;
	}
	else
	{
		len = occur - start;
	}

	return s.substr( start, len );
}

vector<string> SteveLib::split( const string& s, const string& splitters )
{
	vector<string> splat;
	int start = 0;
	while( 1 )
	{
		int occur = s.find_first_of( splitters, start );

		if( occur == string::npos ) {
			// we're done. add the last string
			splat.push_back( s.substr( start, string::npos ) );
			break;
		}
		else {
			splat.push_back( s.substr( start, occur-start ) );
			start = occur + 1;
		}	
	}

	return splat;
}

int SteveLib::count_occurs( const string& s, const string& chars )
{
	int count = 0;

	for( int i = 0; i < s.length(); i++ )
	{
		if( chars.find( s[i] ) != string::npos )
		{
			count++;
		}
	}

	return count;
}

time_t SteveLib::seed_random_generator_with_time()
{
	time_t randSeed = time(NULL);
	cout << "Random seed = " << randSeed << endl;
	srand( (unsigned int)randSeed );
	return randSeed;
}

string SteveLib::datetime_for_filename()
{
	time_t rawtime;
	tm* ptm;
	char buf[256];

	time( &rawtime );

	// convert to EST	
	rawtime -= 5*60*60;

	ptm = gmtime( &rawtime );

	sprintf( buf, "%d-%02d-%02d %02d-%02d-%02d",
			ptm->tm_year + 1900, 
			ptm->tm_mon + 1,
			ptm->tm_mday,
			ptm->tm_hour,
			ptm->tm_min,
			ptm->tm_sec );

	return string(buf);
}

#if defined( WIN32 )

//----------------------------------------
//  Stolen from http://www.godpatterns.com/article/how-to-list-files-in-a-directory-in-c/
//----------------------------------------
static string wchar2string( const WCHAR * const wcharArray )
{
    stringstream ss;

    int i = 0;
    char c = (char) wcharArray[i];
    while(c != '\0') {
        ss <<c;
        i++;
        c = (char) wcharArray[i];
    }

    string convert = ss.str();
    return convert;
}

enum CREATE_DIR_RESULT SteveLib::create_dir( const string& name )
{
	wchar_t* wideStr = NULL;
	size_t numWideChars = 0;
	BOOL ok = false;

	wideStr = new wchar_t[ name.size() ];
	mbstowcs_s( &numWideChars, wideStr, name.size()+1, name.c_str(), _TRUNCATE );
	// FIXME
	//ok = CreateDirectory( wideStr, NULL );
	string temp = wchar2string( wideStr );
	ok = CreateDirectory( temp.c_str(), NULL );
	delete wideStr;

	if( ok )
	{
		return OK;
	}
	else
	{
		cerr << "** Creating directory " << SDUMP(name) << " failed!" << endl;
		// TODO - use GetLastError to get more detailed errror info
		return FAIL;
	}
}

int SteveLib::collect_files_with_extension( string dir, string ext, vector<string>& filesOut )
{
	size_t dirSize = MAX_PATH;
	//TCHAR lpcwstrPattern[ MAX_PATH ];
	wchar_t lpcwstrPattern[ MAX_PATH ];
	WIN32_FIND_DATA findFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	int numCollected = 0;

	string findPattern = dir + string("\\*.") + ext;

	// FIXME - a bunch of changes here to get things compiling...

	// convert to wchar
	mbstowcs_s( &dirSize, lpcwstrPattern, findPattern.size()+1, findPattern.c_str(), _TRUNCATE );

	// FIXME
	string temp = wchar2string( lpcwstrPattern );

	//hFind = FindFirstFile( lpcwstrPattern, &findFileData );
	hFind = FindFirstFile( temp.c_str(), &findFileData );

	if( hFind != INVALID_HANDLE_VALUE )
	{
		//filesOut.push_back( wchar2string( findFileData.cFileName ) );
		filesOut.push_back( string( findFileData.cFileName ) );
		numCollected++;

		while( FindNextFile( hFind, &findFileData ) )
		{
			//filesOut.push_back( wchar2string( findFileData.cFileName ) );
			filesOut.push_back( string( findFileData.cFileName ) );
			numCollected++;
		}
	}

	FindClose( hFind );

	return numCollected;
}

#elif defined( __unix__ )

enum CREATE_DIR_RESULT SteveLib::create_dir( const string& name )
{
	int status = mkdir( name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );

	if( status == 0 )
	{
		return OK;
	}
	else
	{
		switch( errno )
		{
			case EEXIST: return EXISTS;
			default:
			{
				cerr << "** Creating directory " << SDUMP(name) << " failed!" << endl;
				return FAIL;
			}
		}
	}
}

int SteveLib::collect_files_with_extension( string dir, string ext, vector<string>& filesOut )
{
	// TRACE_ERROR( "UNIMPLEMENTED" );
	return -1;
}

string SteveLib::dir_for_path( char* path )
{
	string in(path);
	return in.substr( 0, in.rfind('/')+1 );
}

static char ctime_buf[1024];

char* SteveLib::myctime( time_t* t )
{
	char* s = ctime(t);
	strcpy( ctime_buf, s );
	ctime_buf[ strlen(s)-1 ] = NULL;
	return ctime_buf;
}

string SteveLib::human_time( float secs )
{
	ostringstream ss;
	float x = secs;

	if( x < 60.0 )
	{
		ss << x << "s";
		return ss.str();
	}

	x /= 60.0;
	if( x < 60.0 )
	{
		ss << x << "m";
		return ss.str();
	}

	x /= 60.0;
	if( x < 24 )
	{
		ss << x << "h";
		return ss.str();
	}

	x /= 24.0;
	ss << x << "d";
	return ss.str();
}


string SteveLib::short_hostname()
{
#ifdef WIN32
	return string("NO HOSTNAME FOR WIN32");
#else
	char buf[256];
	gethostname( buf, 256 );
	char* dot = strchr( buf, '.' );
	if( dot != NULL )
		// terminate string at the first dot
		*dot = NULL;
	return string(buf);
#endif
}

#endif


