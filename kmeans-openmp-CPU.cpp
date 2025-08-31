#include <omp.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>

using namespace std;

static const int k = 2;
static const int n_instances = 4000000;
static const int n_attributes  = 22;

vector<float> instances((size_t)n_instances * n_attributes);
vector<int>   results(n_instances, -1);
vector<float> centers((size_t)k * n_attributes);

// Loads instances from CSV file into the 'instances' array
void load_instances(){
    int instance_count = 0;
    string line;
    string aux;
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

// Assigns each instance to the nearest cluster center (parallelized for CPU)
void assign_cluster() {
    const float INF = numeric_limits<float>::infinity();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_instances; ++i) {
        const float* xi = &instances[(size_t)i * n_attributes];
        float best = INF; int bestk = 0;

        for (int j = 0; j < k; ++j) {
            const float* cj = &centers[(size_t)j * n_attributes];
            float d2 = 0.0f;
            for (int a = 0; a < n_attributes; ++a) {
                float d = xi[a] - cj[a];
                d2 += d * d;
            }
            if (d2 < best) { best = d2; bestk = j; }
        }
        results[i] = bestk;
    }
}

// Counts the number of instances assigned to each cluster
void count_instances(int* c) {
    for (int i = 0; i < k; ++i) c[i] = 0;
    for (int i = 0; i < n_instances; ++i) ++c[results[i]];
}

// Updates cluster centers based on assigned instances
void update_centers(int seed_iter) {
    vector<float> sum((size_t)k * n_attributes, 0.0f);
    int c[k]; count_instances(c);

    for (int j = 0; j < n_instances; ++j) {
        int cl = results[j];
        const float* xj = &instances[(size_t)j * n_attributes];
        float* s = &sum[(size_t)cl * n_attributes];
        for (int a = 0; a < n_attributes; ++a) s[a] += xj[a];
    }

    const int N = k * n_attributes;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; ++t) {
        int cl = t / n_attributes;
        int a  = t % n_attributes;
        int cnt = c[cl];

        if (cnt != 0) {
            centers[t] = sum[t] / (float)cnt;
        } else {
            int idx = (seed_iter + cl) % n_instances;
            centers[t] = instances[(size_t)idx * n_attributes + a];
        }
    }
}

// Main k-means loop: loads data, initializes centers, assigns clusters, updates centers
void kmeans(){
    load_instances();
    initialize_centers();
    for (int it = 0; it < 30; ++it) {
        assign_cluster();
        update_centers(it);
    }
}

int main() {
    kmeans();
    for (int i = 0; i < n_instances; ++i) {
        cout << results[i] << std::endl;
    }
}
