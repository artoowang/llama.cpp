set(TARGET llama-simple)
add_executable(${TARGET} simple.cpp color_log_sink.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET}
  PRIVATE
    llama
    ${CMAKE_THREAD_LIBS_INIT}
    absl::flags
    absl::flags_parse
    absl::flags_usage
    absl::log
    absl::log_initialize
    absl::log_sink_registry)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
