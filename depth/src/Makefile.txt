# CXX = g++
# CXXFLAGS = -fpic -O2 -std=c++17
# LDFLAGS = -shared -lstdc++

# MODULES: ddalpha depth_wrapper ACA_wrapper

# all: $(MODULES:%=%.dll)

# %.o: %.cpp %.h
# 	$(CXX) $(CXXFLAGS) -DBUILDING_$* -c $< -o $@

# %.dll: %.o
# 	$(CXX) $(LDFLAGS) -o $@ $<


# clean:
# 	rm -f *.o *.dll