file(REMOVE_RECURSE
  "libglfw.pdb"
  "libglfw.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/glfw.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
