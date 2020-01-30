EXEC = test.x
CPP = c++
FLAGS = -O2

SRC_DIR = ./src
INC_DIR = -I./include
BUILD_DIR = ./build

SRC := $(wildcard $(SRC_DIR/*.cpp)
INC := $(wildcard $(SRC_DIR)/*.h)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

$(EXEC): $(OBJ)
	@$(CPP) $(OBJ) $(LIB) -o $@
	@echo $(EXEC) "compiled."

$(OBJ): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	@$(CPP) $(FLAGS) $(INC_DIR) -c $< -o $@
	@echo "Compiled "$<" successfully :)"

clear:
	@rm -rf @(OBJ) $(EXEC) $(BUILD_DIR)
	@echo "Cleanup complete!"
