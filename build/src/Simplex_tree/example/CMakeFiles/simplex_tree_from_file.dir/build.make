# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/frg/Bureau/mWorkingDirectory

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frg/Bureau/mWorkingDirectory/build

# Include any dependencies generated for this target.
include src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/depend.make

# Include the progress variables for this target.
include src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/progress.make

# Include the compile flags for this target's objects.
include src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/flags.make

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/flags.make
src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o: ../src/Simplex_tree/example/simplex_tree_from_file.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/frg/Bureau/mWorkingDirectory/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o"
	cd /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o -c /home/frg/Bureau/mWorkingDirectory/src/Simplex_tree/example/simplex_tree_from_file.cpp

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.i"
	cd /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/frg/Bureau/mWorkingDirectory/src/Simplex_tree/example/simplex_tree_from_file.cpp > CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.i

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.s"
	cd /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/frg/Bureau/mWorkingDirectory/src/Simplex_tree/example/simplex_tree_from_file.cpp -o CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.s

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.requires:
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.requires

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.provides: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.requires
	$(MAKE) -f src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/build.make src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.provides.build
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.provides

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.provides.build: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o

# Object files for target simplex_tree_from_file
simplex_tree_from_file_OBJECTS = \
"CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o"

# External object files for target simplex_tree_from_file
simplex_tree_from_file_EXTERNAL_OBJECTS =

src/Simplex_tree/example/simplex_tree_from_file: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o
src/Simplex_tree/example/simplex_tree_from_file: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/build.make
src/Simplex_tree/example/simplex_tree_from_file: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable simplex_tree_from_file"
	cd /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simplex_tree_from_file.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/build: src/Simplex_tree/example/simplex_tree_from_file
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/build

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/requires: src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/simplex_tree_from_file.cpp.o.requires
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/requires

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/clean:
	cd /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example && $(CMAKE_COMMAND) -P CMakeFiles/simplex_tree_from_file.dir/cmake_clean.cmake
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/clean

src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/depend:
	cd /home/frg/Bureau/mWorkingDirectory/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frg/Bureau/mWorkingDirectory /home/frg/Bureau/mWorkingDirectory/src/Simplex_tree/example /home/frg/Bureau/mWorkingDirectory/build /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example /home/frg/Bureau/mWorkingDirectory/build/src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/Simplex_tree/example/CMakeFiles/simplex_tree_from_file.dir/depend

