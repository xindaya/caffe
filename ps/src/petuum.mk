
PS_DIR = $(SRC)/petuum_ps
PS_COMMON_DIR = $(SRC)/petuum_ps_common
PS_SN_DIR = $(SRC)/petuum_ps_sn
ML_DIR = $(SRC)/ml

PS_SRC = $(shell find $(PS_DIR) -type f -name "*.cpp")
PS_HEADERS = $(shell find $(PS_DIR) -type f -name "*.hpp")
PS_OBJ = $(PS_SRC:.cpp=.o)

PS_COMMON_SRC = $(shell find $(PS_COMMON_DIR) -type f -name "*.cpp")
PS_COMMON_HEADERS = $(shell find $(PS_COMMON_DIR) -type f -name "*.hpp")
PS_COMMON_OBJ = $(PS_COMMON_SRC:.cpp=.o)

PS_SN_SRC = $(shell find $(PS_SN_DIR) -type f -name "*.cpp")
PS_SN_HEADERS = $(shell find $(PS_SN_DIR) -type f -name "*.hpp")
PS_SN_OBJ = $(PS_SN_SRC:.cpp=.o)

ML_SRC = $(shell find $(ML_DIR) -type f -name "*.cpp")
ML_HEADERS = $(shell find $(ML_DIR) -type f -name "*.hpp")
ML_OBJ = $(ML_SRC:.cpp=.o)

# ================== client library ==================

PS_LIB = $(LIB)/libpetuum-ps.a

PS_SN_LIB = $(LIB)/libpetuum-ps-sn.a

ML_LIB = $(LIB)/libpetuum-ml.a

ps_lib: $(PS_LIB)

$(PS_LIB): $(PS_OBJ) $(PS_COMMON_OBJ) path
	ar csrv $@ $(PS_OBJ) $(PS_COMMON_OBJ)

$(PS_OBJ): %.o: %.cpp $(PS_HEADERS) $(PS_COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c $< -o $@

ps_sn_lib: $(PS_SN_LIB)

$(PS_SN_LIB): $(PS_SN_OBJ) $(PS_COMMON_OBJ) path
	ar csrv $@ $(PS_SN_OBJ) $(PS_COMMON_OBJ)

$(PS_SN_OBJ): %.o: %.cpp $(PS_SN_HEADERS) $(PS_COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c $< -o $@

$(PS_COMMON_OBJ): %.o: %.cpp $(PS_COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c $< -o $@

ml_lib: $(ML_LIB)

$(ML_LIB): $(ML_OBJ) path
	ar csrv $@ $(ML_OBJ)

$(ML_OBJ): %.o: %.cpp $(ML_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c $< -o $@

.PHONY: ps_lib ps_sn_lib ml_lib
