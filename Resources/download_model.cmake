if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR must be defined")
endif()

set(MODELS_DIR "${OUTPUT_DIR}/models")
file(MAKE_DIRECTORY "${MODELS_DIR}")

# Option 1: Copy local pre-exported TorchScript model from your repo's resources/models/
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/models/efficientat_mn10_as.pt")
  file(COPY "${CMAKE_CURRENT_LIST_DIR}/models/efficientat_mn10_as.pt" DESTINATION "${MODELS_DIR}")
endif()

# Option 2: Or download it from a release URL using file(DOWNLOAD ...)
# file(DOWNLOAD "https://.../efficientat_mn10_as.pt" "${MODELS_DIR}/efficientat_mn10_as.pt" SHOW_PROGRESS)
