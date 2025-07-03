INC  += $(BUILD_TOP)/internal/opencv/include/opencv4

INC  += $(BUILD_TOP)/internal/Eigen

ST_DEP := common  iniparser
LIBS += -lmi_ipu -lcam_fs_wrapper

#static library path
LIBS += -L$(BUILD_TOP)/internal/opencv/lib -L$(BUILD_TOP)/internal/opencv/lib/opencv4/3rdparty


#shared library path
#LIBS += -L$(DB_BUILD_TOP)/internal/opencv/shared_lib_${TOOLCHAIN_VERSION}
#shared opencv library
#LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
#static library
LIBS += -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -llibjasper  -llibjpeg-turbo  -llibpng -llibtiff  -llibwebp  -lzlib

