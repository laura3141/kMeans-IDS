#include <fstream>
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std;

static const int k = 2;
static const int n_instances = 4000000;
static const int n_attributes = 22;

size_t N_inst = n_instances; 
size_t N_attr = n_attributes;

float *instances = new float[N_inst*N_attr];
int *results = new int[n_instances];
float *centers = new float[k*N_attr];

// Loads instances from CSV file into the 'instances' array
void load_instances(){
    int instance_count = 0;
    string line, aux;
    std::ifstream file("32_dados.csv");
    getline(file, line); // Skip header
    while (getline(file, line)  && instance_count < n_instances) {
        int p = 0;
        for (int j = 0; j < n_attributes; j++) {
            aux = "";
            while (p < line.size() && line[p] != ',') {
                aux += line[p];
                p++;
            }
            try {
                instances[instance_count * n_attributes + j] = std::stod(aux);
            } catch (const std::exception& e) {
                std::cerr << "Error converting: '" << aux << "'. " << e.what() << std::endl;
                instances[instance_count * n_attributes + j] = 0.0f;
            }
            p++;
        }
        instance_count++;
    }
    printf("Loaded instances: %d \n",instance_count);
}

// Initializes cluster centers randomly from the loaded instances
void initialize_centers(){
    srand(3);
    for (int i = 0; i < k; i++) {
        int index = rand()%n_instances;
        for (int j = 0;j < n_attributes;j++){
            centers[i*n_attributes+j] = instances[index*n_attributes+j];
        }
    }
}

// Counts the number of instances assigned to each cluster
void count_instances(int* c){
    for(int i=0;i<k;i++){
        c[i]=0;
    }
    for(int i=0;i<n_instances;i++){
        c[results[i]]++;
    }
}

// Assigns each instance to the nearest cluster center (parallelized for GPU)
void assign_cluster(){
    size_t Nia = N_inst*N_attr;
    size_t NaSete = k*N_attr;
    #pragma omp target data map(to: instances[0:Nia], centers[0:NaSete]) map(tofrom: results[0:n_instances])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < n_instances; i++){
            float min_dist = INFINITY;
            float distance;
            for(int j = 0; j < k;j++){
                float sum = 0;
                for(int n = 0; n<n_attributes;n++){
                    float diff = instances[i*n_attributes+n] - centers[j*n_attributes+n];
                    sum += diff * diff;
                }
                distance = sum;
                if(distance < min_dist){    
                    min_dist = distance;
                    results[i]=j;
                }
            }
        }
    }
}

// Updates cluster centers based on assigned instances
void update_centers(int* c) {
    count_instances(c);
    float sum[k][n_attributes];
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n_attributes; j++) {
            sum[i][j] = 0.0f;
        }
    }
    for(int i=0;i<n_instances;i++){
        for(int j=0;j<n_attributes;j++){
            sum[results[i]][j]+=instances[i*n_attributes+j];
        }
    }
    #pragma omp target teams distribute map(to: c[0:k]) map(tofrom: sum[0:k][0:n_attributes])
    for(int i = 0; i < k; i++) {
        #pragma omp parallel for
        for(int j = 0; j < n_attributes; j++) {
            if (c[i] != 0) {
                sum[i][j] = sum[i][j] / c[i];
            }
        }
    }
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n_attributes; j++) {
            centers[i * n_attributes + j] = sum[i][j];
        }
    }
}

// Main k-means loop: loads data, initializes centers, assigns clusters, updates centers
void kmeans(){
    int c[k];
    load_instances();
    initialize_centers();
    for(int i=0;i<30;i++){
        assign_cluster(); 
        update_centers(c); 
    }
}

int main (){
    kmeans();
    for (int i = 0; i < n_instances; ++i) {
        cout << results[i] << std::endl;
    }
    delete[] instances;
    delete[] results;
    delete[] centers;
    return 0;
}