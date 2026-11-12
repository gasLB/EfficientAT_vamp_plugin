if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR must be defined")
endif()

set(MODELS_DIR "${OUTPUT_DIR}/models")
if(REFRESH_MODEL AND EXISTS "${MODELS_DIR}")
  file(REMOVE_RECURSE "${MODELS_DIR}")
endif()
file(MAKE_DIRECTORY "${MODELS_DIR}")

set(RELEASE_URL "https://github.com/Ircam-Partiels/efficientat-vamp-plugin/releases/download/models")

set(MODEL_FILES
	"01_mn04_as.ts"
	"02_mn10_as.ts"
	"03_mn20_as.ts"
)

foreach(MODEL_FILE IN LISTS MODEL_FILES)
  set(DEST_FILE "${MODELS_DIR}/${MODEL_FILE}")
  if(NOT EXISTS "${DEST_FILE}")
    message(STATUS "Downloading ${MODEL_FILE} to ${DEST_FILE}...")
    file(DOWNLOAD
      "${RELEASE_URL}/${MODEL_FILE}"
      "${DEST_FILE}"
      SHOW_PROGRESS
      TLS_VERIFY ON
      STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
      file(REMOVE "${DEST_FILE}")
      message(FATAL_ERROR "Failed to download ${MODEL_FILE} (error ${STATUS_CODE})")
    endif()
  endif()
endforeach()
