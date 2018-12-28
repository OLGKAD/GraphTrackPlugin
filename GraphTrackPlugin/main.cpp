#include <opencv2/opencv.hpp>
#include <iostream>
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
int width = 720; // 256; // 720;
int height = 528; // 240; //528;
int patch_width = 15;  // becomes too slow at 20. LiveSketch used 9x9 patches
int patch_height = 15; // 15x15 is what they used in GraphTrack (page. 7 in the paper)
int patches_per_frame = 40000; // 1/8 of all patches, needed for PCA. COULD IT BE DECREASED?
int last_frame_number = 45;
int retraced_path[45];
int current_frame_number = 0;
float lambda_f = 1;
float lambda_b = 1;
float lambda_s = 10;
float lambda_d = 10;
int sigma_b = 4096;
Mat current_frame;
string window_name = "my_window";
Mat pca_patches[40000 * 45] = {};
Mat compressed_patches[45][(528 - 14)][(720 - 14)] = {};
Mat positive_patches[10] = {}; // one per frame in only some frames
int interest_points_per_video = 10; // in how many frames an interest point should be marked
int positive_patch_counter = 0; // the next positve patch will be saved at this index in positive_patches
Mat negative_patches[10 * 20] = {}; // in each frame where interest points were marked (5), we select a number of negative examples (20 - CHOSEN RANDOMLY)
int negative_patches_per_frame = 20;
int negative_patch_counter = 0; // the next negative patch will be saved at this index in negative_patches
Mat candidate_nodes[45][200] = {}; // 200 candidates patches, per each frame
Point candidate_nodes_coordinates[45][200] = {};
int nodes_per_frame = 200;
bool has_interest_point[45] = {}; // true if an interest point was marked in the frame
uchar expanded[45][200] = {}; // 1 - if expanded; 0 - not expanded; -1 - node doesn't exist
float distance_from_source[45][200] = {}; // stores distances to each node from the source. Initialized to +inf (HUGE_VALF).
int parent_pointers[45][200] = {}; // needed to recover the path after a run of Djikstra. Every element stores the index of parent node in the previous row.
int sink_parent_pointer;

// only vars that could've changed (and the change accumulates)
void initialize_global_vars_again() {
    
}
/* computes the average color of the video: sum all the pixel RGB values (over all pixels, over all frames) and divide by the #pixels (width * height * frames).  AND
 * Substracts the average color from every pixel in the video, before it's used in PCA.
 */
void compute_average_color() {
    cout << "Computing average color" << endl;
    Mat average_frame(height, width, CV_32FC3, Scalar(0));
    Scalar average_color;
    Mat temp1;
    for (int i = 0; i < last_frame_number; i++) {
        frames_unnormalized[i].convertTo(temp1, CV_32FC3);
        average_frame += temp1;
    }
    //    write_mat_to_file(average_frame, "sum_of_frames");
    average_frame /= last_frame_number;
    average_color = sum(average_frame) / (height * width);
    //    write_mat_to_file(average_frame, "average_frame");
    
    // now subtract average color from every pixel of every frame
    Mat temp2;
    for (int i = 0; i < last_frame_number; i++) {
        frames_unnormalized[i].convertTo(temp2, CV_32FC3);
        frames[i] = temp2 - average_color;
    }
    
}

void write_mat_to_file(Mat mat, string filename) {
    FileStorage file("debug_log/" + filename, FileStorage::WRITE);
    
    file << filename << mat; // the matrix in the file will be named the same as the filename here.
}

// another version of "flatten". As of now, it's faster
/////////////////////////////////////// SOLUTION: seems like the only way to speed things up is to understand that "CONVOLUTIONS" part
Mat flatten(Mat patch) {
    Mat bgr[3];   //destination array
    split(patch,bgr);//split source
    Mat result;
    // stack the channels into a new mat:
    // this loop really SLOWS things down
    for (int i=0; i<3; i++)
        result.push_back(bgr[i]);
    result = result.reshape(1,1);
    
    return result;
    
}

// fills out the "frames" array
void read_video() {
    cout << "Reading the video" << endl;
    VideoCapture cap("/Users/kadyrakunovolzhas/Desktop/LowLevelPlugin/image1.jpg /Users/kadyrakunovolzhas/Desktop/Spring\ 2018/Computer\ Graphics/CG\ final\ project/Assets/rabbit_fast.avi");
    
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video" << endl;
        cin.get(); //wait for any key press
    }
    
    int frameNumber = -1;
    Mat frame;
    while (true)
    {
        // skip the first frame, since it's blank
        if (frameNumber == -1) {
            frameNumber += 1;
            cap.read(frame);
            //            cout << frame.rows << endl;
            //            cout << frame.cols << endl;
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
    //    write_mat_to_file(frames[0], "first_video_frame");
}

PCA computePCA_basis() {
    cout << "Running PCA" << endl;
    
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
    cout << "Compressing the video" << endl;
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors;
    
    Mat frame_temp;
    Mat frame;
    Mat transposed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    Mat compressed_patch(patch_height * patch_width * 3, 1, CV_32FC1);
    for (int i = 0; i < last_frame_number; i++) {
        cout << " frame number: " << i << endl;
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
    
}

void mark_interest_point(int frame_number, int x, int y)
{
    // the point of click is the center of the rectangle. However patches are defined by their top-left corner => x and y should be adjusted.

    //// DO
    candidate_nodes_coordinates[frame_number][0] = Point(x, y);
    
    //// DO
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

void mark_all_interest_points() {
    // get data from the txt file
    
    // repeatedly call mark_interest_point()
}

//// Displays the frame (with a rectangle on it). Doesn't do anything else.
//void on_trackbar(int, void* args) {
//    //    cout << "current trackbar frame: " << current_frame_number << endl;
//    current_frame = frames_unnormalized[current_frame_number].clone();
//    // find where the interest point is, and put a rectangle there
//    Point next_node_coordinate = candidate_nodes_coordinates[current_frame_number][retraced_path[current_frame_number]];
//    int x = next_node_coordinate.x;
//    int y = next_node_coordinate.y;
//    rectangle(current_frame, Point(x,y), Point(x + patch_width, y + patch_height), Scalar(0,0,0), 3);
//
//    imshow(window_name, current_frame);
//    //    waitKey(0);
//
//}
//// I WANT NONE-NORMALIZED FRAMES TO BE DISPLAYED
//void mark_interest_points() {
//    cout << "Mark interest points" << endl;
//    namedWindow(window_name); //create a window
//
//    current_frame = frames_unnormalized[0]; /////////////////////////// ALWAYS CALLED WITH THE 1ST FRAME
//    setMouseCallback(window_name, on_mouse_click, &current_frame);
//    imshow(window_name, current_frame);
//    //// CURRENT_FRAME_NUMBER NEEDS TO BE UPDATED BEFORE HAND (OR MAYBE IT'S NOT EVEN NEEDED)
//    createTrackbar( "frames", "my_window", &current_frame_number, last_frame_number - 1, on_trackbar);
//
//    /// Wait until user press some key
//    while(waitKey() != 27); //27 is the keycode for ESC
//    //    destroyAllWindows();  // the same window will be used to play the final video
//}


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
        //        cout << "reached 1.6" << endl;
        this_distance = sum(temp.mul(temp))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    //    closest_distance *= lambda_f;
    
    return closest_distance;
}

// given a node, defined by its layer (frame) and number, determine its distance to the closest negative patch.
float distance_to_negative_patches(int frame, int node) {
    float closest_distance = HUGE_VALF;
    float this_distance;
    Mat temp;
    
    for (int i = 0; i < negative_patch_counter; i++) {
        temp = negative_patches[i] - candidate_nodes[frame][node];
        //        cout << "reached 1.7" << endl;
        this_distance = sum(temp.mul(temp))[0];
        if (this_distance < closest_distance) {
            closest_distance = this_distance;
        }
    }
    
    if (closest_distance > sigma_b) {
        closest_distance = sigma_b;
    }
    
    //    closest_distance *= lambda_b;
    
    return closest_distance;
}

// candidate selection::: UNTESTED
// selects 200-200 nodes (patches) in each frame, resembling the interest points the most.
void select_candidates() {
    cout << "Candidate nodes are being chosen" << endl;
    /* we'll have (w-4) * (h-4) values, and we'll need the 200 smallest. Sorting will cost O(w*h * log(w*h)). Finding 200 smallest costs O(w*h * 200), which is faster.
     So, we'll first find the min, then the min of the rest, etc.
     */
    Mat distances(height - patch_height + 1, width - patch_width + 1, CV_32FC1);
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    /////////////////// ERROR: <= 72 check is not performed
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
    //    write_mat_to_file(pca.eigenvectors, "PCA_eigenvectors");
    compress_all_patches(pca); // runs at about 5-10 sec / frame, depending on a day... . Fine for now.
    //    write_mat_to_file(compressed_patches[0][0][0], "first_compressed_patch");
}

// assumes that candidate nodes have already been selected, and interest points marked
void djikstra() {
    cout << "Running Djikstra Algorithm" << endl;
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
    
    //    cout << "reached 1" << endl;
    float distance_from_source_to_sink = HUGE_VALF;
    //    bool sink_expanded = false;
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
                //                cout << "reached 1.5" << endl;
            } else {
                for (int i = 0; i < nodes_per_frame; i++) {
                    distance_from_source[0][i] = lambda_f * distance_to_positive_patches(0, i) - lambda_b * distance_to_negative_patches(0, i) + const_shift;
                }
            }
            last_layer_expanded += 1; // meaningless variable
            source_expanded = true;
            //            cout << "reached 2" << endl;
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
            //            cout << "reached 3" << endl;
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
                //                cout << "reached 4" << endl;
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
                //                cout << "reached 5" << endl;
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
                //                cout << "reached 6" << endl;
            }
            
            expanded[min_node_frame][min_node_frame] = 1;
        }
    }
}

/*
 * retraces the path, and displays the resulting video
 * assumes that Djikstra has been run
 */
void play_final_video() {
    cout << "Displaying the result" << endl;
    // retrace the path
    int next_pointer = sink_parent_pointer;
    for (int i = 0; i < last_frame_number; i++) {
        retraced_path[last_frame_number - i - 1] = next_pointer;
        next_pointer = parent_pointers[last_frame_number - i - 1][next_pointer];
    }
    
    // display the result
    namedWindow(window_name); // namedWindow("final video");
    Mat frame;
    Point next_node_coordinate;
    int x;
    int y;
    VideoWriter video("media/outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(width, height));
    
    for (int i = 0; i < last_frame_number; i++) {
        frame = frames_unnormalized[i].clone();
        
        next_node_coordinate = candidate_nodes_coordinates[i][retraced_path[i]];
        x = next_node_coordinate.x;
        y = next_node_coordinate.y;
        
        rectangle(frame, Point(x,y), Point(x + patch_width, y + patch_height), Scalar(0,0,0), 3);
        video.write(frame);
        imshow(window_name, frame); // imshow("final video", frame);
        if (waitKey(50) == 27)
        {
            cout << "Esc key is pressed by the user. Stopping the video" << endl;
            break;
        }
        
    }
    
    video.release();
    
    while (true) {
        // escape pressed => exit
        if (waitKey() == 27) {
            break;
            // enter pressed, recompute the path (happens after a new interest point was marked)
        } else if (waitKey() == 13) {
            select_candidates();
            djikstra(); // Won't help because by this time candidate_nodes have already been selected, and if the right patch wasn't included, running Djikstra again, won't help.
            play_final_video();
        }
    }
}

Mat image = imread("/Users/kadyrakunovolzhas/Desktop/LowLevelPlugin/image1.jpg");
int display_image() {
    if (image.empty())
    {
        return -1;
    }
    //    namedWindow("Display window");
    //    imshow( "Display window",  image);
    //    waitKey(0);
    return 7;
}

//int main(int argc, const char * argv[]) {
//    read_video(); // fast
//    compress_video(); // slow, but that's expected. "The feature extraction runs at about 1 frame in 3 sec on a commodity PC" - at 10x10 patcfhes. 1 frame in 4 sec for 15x15 patches.
//    
//    // only do it in the first frame. If the resulting path has errors, corrections will be made later (in play_final_video())
//    mark_interest_points(); // fast.  MULTIPLE INTEREST POINTS SHOULD BE MARKED IN THE FIRST FRAME
//    
//    // SHOULD GO IN A LOOP OVER ALL INTEREST POINTS
//    select_candidates(); // slow (when 5 points marked; pretty fast when 1 point marked => linear in points marked). Precision is such that I end up marking 5 interest points throughout the video anyway, but it's better to do it aftwerwards, when I see the exact frames where the point is hard to recognize.
//    djikstra(); // fast => no need to use the modified version of Djikstra.
//    // SHOULD GO IN A LOOP OVER ALL INTEREST POINTS
//    
//    /* it will display the result, which can be corrected (by replacing the rectangles in some frames). After that it go back to recomputing the final trajectory and play the final video.
//     * Press ENTER - to recompute the path. ESC - to exit
//     */
//    play_final_video(); // fast
//    //    cout << display_image() << endl;
//}

void precomputations() {
    read_video();
    compress_video();
    // no need to write anything into txt files, I think
}

void compute_path() {
    /* TODO
     * get the args passed to this function (or get them from a txt file)  in order to run
     * select_candidates() and djikstra() on updated args
     */
    
    select_candidates();
    djikstra();
    /* write the path into a txt file
     * 1. Point candidate_nodes_coordinates[45][200]
     * 2. int retraced_path[45]
     * 3. bool has_interest_point[45]  also need to be updated before hand
     * See play_final_video() on how to reconstruct the path using those two arrays.
     * OR why not do all the computations here and only pass one txt file.
     */
    
}

