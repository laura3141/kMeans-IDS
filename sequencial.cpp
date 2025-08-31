#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ittnotify.h>
using namespace std;

static const int k = 2;
static const int n_instances = 4000000;
static const int n_attributes = 22;

size_t N_inst = n_instances;
size_t N_attr = n_attributes;

double assign_time = 0;
double update_time = 0;

vector<double> instances(N_inst * N_attr);
vector<int> results(n_instances, -1);
vector<double> centers(k * N_attr);


// Loads instances from CSV file into the 'instances' array
void load_instances() {
    int instance_count = 0;
    string line, aux;

    std::ifstream file("32_dados.csv");

    getline(file, line); // Skip header

    while (getline(file, line) && instance_count < n_instances) {
        int p = 0;
        for (int j = 0; j < n_attributes; j++) {
            aux = "";
            while (p < line.size() && line[p] != ',') {
                aux += line[p];
                p++;
            }

            try {
                instances[instance_count * n_attributes + j] = stof(aux);
            } catch (const exception& e) {
                cerr << "Error converting: '" << aux << "'. " << e.what() << endl;
                instances[instance_count * n_attributes + j] = 0.0f;
            }
            p++;
        }
        instance_count++;
    }

    printf("Loaded instances: %d \n", instance_count);
}

// Initializes cluster centers randomly from the loaded instances
void initialize_centers() {
    srand(3);
    for (int i = 0; i < k; i++) {
        int index = rand() % n_instances;
        for (int j = 0; j < n_attributes; j++) {
            centers[i * n_attributes + j] = instances[index * n_attributes + j];
        }
    }
}

// Counts the number of instances assigned to each cluster
void count_instances(int* c) {
    for (int i = 0; i < k; i++)
        c[i] = 0;
    for (int i = 0; i < n_instances; i++)
        c[results[i]]++;
}

// Assigns each instance to the nearest cluster center (sequential version)
void assign_cluster() {
    for (int i = 0; i < n_instances; i++) {
        double min_dist = INFINITY;
        int best_cluster = -1;

        for (int j = 0; j < k; j++) {
            double sum = 0;
            for (int n = 0; n < n_attributes; n++) {
                double diff = instances[i * n_attributes + n] - centers[j * n_attributes + n];
                sum += diff * diff;
            }

            if (sum < min_dist) {
                min_dist = sum;
                best_cluster = j;
            }
        }

        results[i] = best_cluster;
    }
}

// Updates cluster centers based on assigned instances
void update_centers(int* c) {
    count_instances(c);

    vector<double> sum(k * n_attributes, 0.0f);

    for (int j = 0; j < n_instances; j++) {
        int cluster = results[j];
        for (int a = 0; a < n_attributes; a++) {
            sum[cluster * n_attributes + a] += instances[j * n_attributes + a];
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n_attributes; j++) {
            int idx = i * n_attributes + j;
            centers[idx] = (c[i] != 0) ? (sum[idx] / c[i]) : 0.0f;
        }
    }
}

// Main k-means loop: loads data, initializes centers, assigns clusters, updates centers
void kmeans() {
    load_instances();
    initialize_centers();
    int c[k];

    for (int i = 0; i < 30; i++) {
        assign_cluster();
        update_centers(c);
    }
}

int main() {
    kmeans();
    for (int i = 0; i < n_instances; ++i) {
        cout<<results[i] << std::endl;
    }
    return 0;
}