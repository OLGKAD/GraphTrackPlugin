# GraphTrackPlugin
GraphTrack plugin for the LiveSketch project

All the functions can be found in main.cpp. Some of the main functions are described below: 

1. read_video() - reads the video file and stores the frames into a matrix. 
2. compress_all_patches() - computes the PCA basis of all the patches in the video, and projects every patch into that basis, producing 16-dimensional vectors representing each patch. 
3. precomputations() = read_video() + compress_all_patches(). 
4. mark_all_interest_points() - reads the coordinates of the manually marked interest patches from a file. (They were priorly saved into that file by a C# function). 
5. select_candidates() - Constructs the graph. Chooses 250 patches from every frame that are the most likely to be interest patches. 
6. djikstra() - runs the Djikstra's algorithm on the graph and returns the shortest path (== the track).  
