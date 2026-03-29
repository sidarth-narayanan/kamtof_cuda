cmake_minimum_required( VERSION 3.20 )

project( "Input Framework" )

set( INPUT_FRAMEWORK_DIR  $ENV{INPUT_FRAMEWORK_DIR} )

aux_source_directory( ${INPUT_FRAMEWORK_DIR}/src InputFrameworkSrcTop)

add_library( input_framework ${InputFrameworkSrcTop} )

# FIXME: Do we need this to be PUBLIC ?
target_include_directories( input_framework PUBLIC
   ${INPUT_FRAMEWORK_DIR}/include
)
