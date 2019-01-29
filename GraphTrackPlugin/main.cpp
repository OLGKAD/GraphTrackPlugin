#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <vector>
#include <ctime>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "main.pch"

using namespace cv;
using namespace std;

// GLOBAL VARIABLES
Mat frames_unnormalized[269] = {};
Mat frames[269] = {}; // video has 269 frames (270, but the first frame is blank)
Mat current_frame;
Mat pca_patches[40000 * 45] = {};
Mat compressed_patches[45][(528 - 14)][(720 - 14)] = {};
Mat positive_patches[10] = {}; // one per frame in only some frames
Mat negative_patches[10 * 20] = {}; // in each frame where interest points were marked (5), we select a number of negative examples (20 - CHOSEN RANDOMLY)
Mat candidate_nodes[45][200] = {}; // 200 candidates patches, per each frame
Point candidate_nodes_coordinates[45][200] = {};
uchar expanded[45][200] = {}; // 1 - if expanded; 0 - not expanded; -1 - node doesn't exist

int width = 720;
int height = 528;
int patch_width = 15;  // becomes too slow at 20. LiveSketch used 9x9 patches
int patch_height = 15; // 15x15 is what they used in GraphTrack (page. 7 in the paper)
int patches_per_frame = 40000; // 1/8 of all patches, needed for PCA.
int last_frame_number = 45;
int retraced_path[45];
int current_frame_number = 0;
float lambda_f = 1;
float lambda_b = 1;
float lambda_s = 10;
float lambda_d = 10;
int sigma_b = 4096;
int interest_points_per_video = 10; // in how many frames an interest point should be marked
int positive_patch_counter = 0; // the next positve patch will be saved at this index in positive_patches
int negative_patches_per_frame = 20;
int negative_patch_counter = 0; // the next negative patch will be saved at this index in negative_patches
int nodes_per_frame = 200;
bool has_interest_point[45] = {}; // true if an interest point was marked in the frame
float distance_from_source[45][200] = {}; // stores distances to each node from the source. Initialized to +inf (HUGE_VALF).
int parent_pointers[45][200] = {}; // needed to recover the path after a run of Djikstra. Every element stores the index of parent node in the previous row.
int sink_parent_pointer;


/* computes the average color of the video: sum all the pixel RGB values (over all pixels, over all frames) and divide by the #pixels (width * height * frames).  AND
 * Substracts the average color from every pixel in the video, before it's used in PCA.
 */
void compute_average_color() {
    Mat average_frame(height, width, CV_32FC3, Scalar(0));
    Scalar average_color;
    Mat temp1;
    for (int i = 0; i < last_frame_number; i++) {
        frames_unnormalized[i].convertTo(temp1, CV_32FC3);
        average_frame += temp1;
    }
    average_frame /= last_frame_number;
    average_color = sum(average_frame) / (height * width);
    
    // subtract average color from every pixel of every frame
    Mat temp2;
    for (int i = 0; i < last_frame_number; i++) {
        frames_unnormalized[i].convertTo(temp2, CV_32FC3);
        frames[i] = temp2 - average_color;
    }
    
}

void write_mat_to_file(Mat mat, string filename) {
    FileStorage file("/Users/kadyrakunovolzhas/Desktop/LiveSketch/Unity_C++_communication/" + filename, FileStorage::WRITE);
    
    file << filename << mat; // the matrix in the file will be named the same as the filename here.
}

// Transforms a 2D Mat into a 1D Mat
Mat flatten(Mat patch) {
    Mat bgr[3];   //destination array
    split(patch,bgr);//split source
    Mat result;
    // stack the channels into a new mat:
    for (int i=0; i<3; i++)
        result.push_back(bgr[i]);
    result = result.reshape(1,1);
    
    return result;
    
}

// reads the video and fills out the "frames" array by storing the frames there
void read_video() {
    VideoCapture cap("rabbit_fast.avi");
    int frameNumber = -1;
    Mat frame;
    while (true)
    {
        // skip the first frame, since it's blank
        if (frameNumber == -1) {
            frameNumber += 1;
            cap.read(frame);
            continue;
        }
        
        bool isSuccess = cap.read(frame); // read a new frame from the video
        if (isSuccess == false || frameNumber >= last_frame_number)
        {
            break;
        }
        frames_unnormalized[frameNumber] = frame.clone();
        frameNumber += 1;
    }
    cap.release();
    compute_average_color();
    
    write_mat_to_file(frames_unnormalized[0], "frames_unnormalized_0");
    write_mat_to_file(frames[0], "frames_0");
    
    // store all frames
    for (int i = 0; i < last_frame_number; i++) {
        write_mat_to_file(frames_unnormalized[i], "all_frames_unnormalized");
    }
}

// perform the PCA algorithm
PCA computePCA_basis() {
    Mat frame;
    Mat frame_temp;
    for (int i = 0; i < last_frame_number; i++) {
        frame_temp = frames[i];
        frame_temp.convertTo(frame, CV_32FC3);
        Mat patch;
        for (int j = 0; j < patches_per_frame; j++) {
            srand(time(0));
            int patch_x = rand() % (width / patch_width);
            int patch_y = rand() % (height / patch_height);
            pca_patches[i * patches_per_frame + j] = flatten(frame(Rect( patch_x * patch_width, patch_y * patch_height, patch_width, patch_height))).clone();
        }
    }
    
    Mat patches(patches_per_frame * last_frame_number, patch_width * patch_height * 3, CV_32FC1);
    vconcat( pca_patches, patches_per_frame * last_frame_number, patches );
    
    PCA pca(patches, Mat(), PCA::DATA_AS_ROW, 16);
    
    return pca;
}

// use the PCA basis (16 eigenvectors) to project each patch vector in every frame to a 16-vector space
void compress_all_patches(PCA pca) {
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors;
    write_mat_to_file(eigenvalues, "eigenvalues");
    write_mat_to_file(eigenvalues, "eigenvectors");
    
    Mat frame_temp;
    Mat frame;
    Mat transposed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    Mat compressed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    for (int i = 0; i < last_frame_number; i++) {
        frame_temp = frames[i];
        frame_temp.convertTo(frame, CV_32FC3);
        
        for (int j = 0; j < height - patch_height + 1; j++) {
            for (int k = 0; k < width - patch_width + 1; k++) {
                transpose(flatten(frame(Rect( k, j, patch_width, patch_height))), transposed_patch);
                compressed_patch = eigenvectors * transposed_patch;
                compressed_patches[i][j][k] = compressed_patch.clone();
            }
        }
        
    }
    write_mat_to_file(compressed_patches[0][383][330], "compressed_patches_0_383_330");
    
}

void mark_interest_point(int frame_number, int x, int y)
{
    // the point of click is the center of the rectangle. However patches are defined by their top-left corner => x and y should be adjusted.
    candidate_nodes_coordinates[frame_number][0] = Point(x, y);
    // save the positive patch
    positive_patches[positive_patch_counter] = compressed_patches[frame_number][y][x].clone();
    positive_patch_counter += 1;
    // save the negative patches. THIS PART IS INCOMPLETE
    int x_temp;
    int y_temp;
    for (int i = 0; i < negative_patches_per_frame; i++) {
        srand(time(0));
        // loop makes sure that the negative pathc is "far away" from the positive patch
        while(true) {
            x_temp = rand() % (width - patch_width + 1);
            y_temp = rand() % (height - patch_height + 1);
            // if far away from the positive patch
            if (abs(x_temp - x) > 50 && abs(y_temp - y) > 50) {
                break;
            }
        }
        negative_patches[negative_patch_counter] = compressed_patches[frame_number][y_temp][x_temp].clone();
        negative_patch_counter += 1;
    }
    
    // record which frames contain interst points
    has_interest_point[frame_number] = true;
    for (int i = 1; i < nodes_per_frame; i++) {
        expanded[frame_number][i] = -1;
    }
    
}

// reads the coordinates of the interest points from a txt file (that was filled in a C# function in Unity)
void mark_all_interest_points() {
    // get data from the txt file
    string line;
    ifstream myfile ("/Users/kadyrakunovolzhas/Desktop/LiveSketch/Unity_C++_communication/interest_points.txt");
    stringstream ssin;
    int frame_number;
    int x;
    int y;
    
    while ( getline (myfile,line) ){
        ssin = stringstream(line);
        ssin >> frame_number;
        ssin >> x;
        ssin >> y;
        mark_interest_point(frame_number, x, y);
    }
}


/*
 * Given coordinates of a patch in copressed_patches, it returns its (approximate) distance (sum of absoulute differences) to the closest positive patch.
 * Used in candidate_selection().
 */
float approximate_distance_to_positive_patches(int frame, int y, int x) {
    float closest_distance = HUGE_VALF;
    float this_distance;
    
    for (int i = 0; i < positive_patch_counter; i++) {
        this_distance = sum(abs(positive_patches[i] - compressed_patches[frame][y][x]))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    return closest_distance;
}

// given a node, defined by its layer (frame) and number, determine its distance to the closest positive patch.
float distance_to_positive_patches(int frame, int node) {
    float closest_distance = HUGE_VALF;
    float this_distance;
    Mat temp;
    
    for (int i = 0; i < positive_patch_counter; i++) {
        temp = positive_patches[i] - candidate_nodes[frame][node];
        this_distance = sum(temp.mul(temp))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    return closest_distance;
}

// given a node, defined by its layer (frame) and number, determine its distance to the closest negative patch.
float distance_to_negative_patches(int frame, int node) {
    float closest_distance = HUGE_VALF;
    float this_distance;
    Mat temp;
    
    for (int i = 0; i < negative_patch_counter; i++) {
        temp = negative_patches[i] - candidate_nodes[frame][node];
        this_distance = sum(temp.mul(temp))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    if (closest_distance > sigma_b) {
        closest_distance = sigma_b;
    }
    
    return closest_distance;
}

// selects 200-200 nodes (patches) in each frame, resembling the interest points the most.
void select_candidates() {
    /* we'll have (w-4) * (h-4) values, and we'll need the 200 smallest. Sorting will cost O(w*h * log(w*h)). Finding 200 smallest costs O(w*h * 200), which is faster.
     So, we'll first find the min, then the min of the rest, etc.
     */
    Mat distances(height - patch_height + 1, width - patch_width + 1, CV_32FC1);
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    for (int i = 0; i < last_frame_number; i++) {
        // if an interest point was marked in this frame => then only one node will be chosen
        if (has_interest_point[i]) {
            // before the outer loop begins positive_patch_counter = 5 (or whatever the #interest points marked in all frames is).
            Point temp_interest_point = candidate_nodes_coordinates[i][0];
            candidate_nodes[i][0] = compressed_patches[i][temp_interest_point.y][temp_interest_point.x];
        } else {
            // fill up "distances" of each patch to the closest positive patch
            for (int j = 0; j < height - patch_height + 1; j++) {
                for (int k = 0; k < width - patch_width + 1; k++) {
                    distances.row(j).col(k) = approximate_distance_to_positive_patches(i, j, k);
                }
            }
            // select nodes_per_frame candidate nodes in each frame, each time choosing the most likely candidate (closest)
            for (int j = 0; j < nodes_per_frame; j++) {
                minMaxLoc( distances, &minVal, &maxVal, &minLoc, &maxLoc );
                candidate_nodes_coordinates[i][j] = minLoc;
                candidate_nodes[i][j] = compressed_patches[i][minLoc.y][minLoc.x].clone();
                // replace the entry with the maxVal to make sure it never gets selected again
                distances.row(minLoc.y).col(minLoc.x) = maxVal;
            }
        }
        
    }
}

// each patch will be represented using 16 numbers (instead of 450), and everything will be stored in compressed_video array;
void compress_video() {
    PCA pca = computePCA_basis();
    compress_all_patches(pca); // runs at 5-10 sec/frame
}

// assumes that candidate nodes have already been selected, and interest points marked
void djikstra() {
    // Initialize all the appropriate matrices
    for (int i = 0; i < last_frame_number; i++) {
        if (has_interest_point[i] != true) {
            has_interest_point[i] = false;
        }
    }
    for (int i = 0; i < last_frame_number; i++) {
        for (int j = 0; j < nodes_per_frame; j++) {
            if (expanded[i][j] != -1) {
                expanded[i][j] = 0;
            }
            distance_from_source[i][j] = HUGE_VALF;
        }
    }
    
    float distance_from_source_to_sink = HUGE_VALF;
    bool source_expanded = false;
    int last_layer_expanded = -1;
    float next_closest_node = HUGE_VALF;
    int min_node_frame;
    int min_node_number;
    float temp_edge;
    float temp_node;
    bool any_nodes_left = true;
    int const_shift = 10000; // Some of the distances to the source could be negative, thus Djikstra may not be able to find the shortest path. To confront this we add a large constant value to each node, which doesn't change the relative path lengths, but lets Djikstra ran smoothly.
    
    //IS IT CORRECT TO ONLY CHECK FORWARD EDGES, OR SHOULD I CHECK PREVIOUS LAYERS AS WELL? - YES. The graph is directed.
    while(true) {
        if (source_expanded == false) {
            if (has_interest_point[0]) {
                distance_from_source[0][0] = lambda_f * distance_to_positive_patches(0, 0) - lambda_b * distance_to_negative_patches(0, 0) + const_shift;
            } else {
                for (int i = 0; i < nodes_per_frame; i++) {
                    distance_from_source[0][i] = lambda_f * distance_to_positive_patches(0, i) - lambda_b * distance_to_negative_patches(0, i) + const_shift;
                }
            }
            last_layer_expanded += 1; // meaningless variable
            source_expanded = true;
        } else {
            any_nodes_left = false;
            // find the node with the min distance to the source that hasn't been expanded yet
            for (int i = 0; i < last_layer_expanded + 1; i++) { // don't go beyond last_layer_expanded as nodes are not expanded there
                for (int j = 0; j < nodes_per_frame; j++) {
                    if ((expanded[i][j] == 0) && (next_closest_node < distance_from_source[i][j])) {
                        next_closest_node = distance_from_source[i][j];
                        min_node_frame = i;
                        min_node_number = j;
                        // such a node is found
                        any_nodes_left = true;
                    }
                }
            }
            // if no nodes were found
            if (any_nodes_left == false) {
                break;
            }
            
            // update dist_to_source of its neighbours (nodes in the next layer)
            // if this was the last layer
            if (min_node_frame == last_frame_number - 1) {
                if (distance_from_source_to_sink > distance_from_source[min_node_frame][min_node_number]) {
                    distance_from_source_to_sink = distance_from_source[min_node_frame][min_node_number];
                    sink_parent_pointer = min_node_number;
                }
            // if the next layer has only 1 node
            } else if (has_interest_point[min_node_frame + 1] && expanded[min_node_frame + 1][0] == 0) {
                Mat temp;
                Point node_pos_1 = candidate_nodes_coordinates[min_node_frame][min_node_number];
                Point node_pos_2 = candidate_nodes_coordinates[min_node_frame + 1][0];
                float euclidian_distance = pow(node_pos_1.x - node_pos_2.x, 2) + pow(node_pos_1.y - node_pos_2.y, 2);
                temp = candidate_nodes[min_node_frame][min_node_number] - candidate_nodes[min_node_frame + 1][0];
                temp_edge = lambda_s * sum(temp.mul(temp))[0] + lambda_d * euclidian_distance;
                temp_node = lambda_f * distance_to_positive_patches(min_node_frame + 1, 0) - lambda_b * distance_to_negative_patches(min_node_frame + 1, 0) + const_shift;
                
                if (distance_from_source[min_node_frame + 1][0] > distance_from_source[min_node_frame][min_node_number] + temp_edge + temp_node) {
                    distance_from_source[min_node_frame + 1][0] = distance_from_source[min_node_frame][min_node_number] + temp_edge + temp_node;
                    parent_pointers[min_node_frame + 1][0] = min_node_number;
                }
            // the next layer has many nodes
            } else {
                Mat temp;
                Point node_pos_1 = candidate_nodes_coordinates[min_node_frame][min_node_number];
                Point node_pos_2;
                float euclidian_distance;
                
                for (int i = 0; i < nodes_per_frame; i++) {
                    if (expanded[min_node_frame + 1][i] == 0) {
                        node_pos_2 = candidate_nodes_coordinates[min_node_frame + 1][i];
                        euclidian_distance = pow(node_pos_1.x - node_pos_2.x, 2) + pow(node_pos_1.y - node_pos_2.y, 2);
                        temp = candidate_nodes[min_node_frame][min_node_number] - candidate_nodes[min_node_frame + 1][i];
                        temp_edge = lambda_s * sum(temp.mul(temp))[0] + lambda_d * euclidian_distance;
                        temp_node = lambda_f * distance_to_positive_patches(min_node_frame + 1, i) - lambda_b * distance_to_negative_patches(min_node_frame + 1, i) + const_shift;
                        
                        if (distance_from_source[min_node_frame + 1][i] > distance_from_source[min_node_frame][min_node_number] + temp_edge + temp_node) {
                            distance_from_source[min_node_frame + 1][i] = distance_from_source[min_node_frame][min_node_number] + temp_edge + temp_node;
                            parent_pointers[min_node_frame + 1][i] = min_node_number;
                        }
                    }
                }
            }
            
            expanded[min_node_frame][min_node_frame] = 1;
        }
    }
}

void precomputations() {
    read_video();
    compress_video();
}

void compute_path() {
    select_candidates();
    djikstra();

    // retrace the path
    int next_pointer = sink_parent_pointer;
    for (int i = 0; i < last_frame_number; i++) {
        retraced_path[last_frame_number - i - 1] = next_pointer;
        next_pointer = parent_pointers[last_frame_number - i - 1][next_pointer];
    }
    
    // extract the coordinates from the retraced_path
    Point next_node_coordinate;
    int x;
    int y;
    ofstream myfile;
    myfile.open  ("/Users/kadyrakunovolzhas/Desktop/LiveSketch/Unity_C++_communication/computed_path.txt", std::ios::app);
    for (int i = 0; i < last_frame_number; i++) {
        next_node_coordinate = candidate_nodes_coordinates[i][retraced_path[i]];
        x = next_node_coordinate.x;
        y = next_node_coordinate.y;
        // write into a file
        myfile << i << " " << x << " " << y << endl;
    }
    myfile.close();
    
}

