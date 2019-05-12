set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
find_package(Boost)

include_directories(
    ${CMAKE_SOURCE_DIR}/include LINK_PUBLIC ${Boost_INCLUDE_DIRS} ${Boost_INCLUDE_DIR}
)

FIND_PACKAGE( OpenCV REQUIRED )
add_executable(laneDetector demo.cpp)
TARGET_LINK_LIBRARIES( laneDetector ${OpenCV_LIBS} ${Boost_LIBRARIES} )



