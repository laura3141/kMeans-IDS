#include <CL/sycl.hpp>
#include <fstream>
#include <cmath>
#include <chrono>
#include <nvToolsExt.h>

using namespace std;
using namespace sycl;

static const int k = 2;
static const int n_instances = 4000000;
static const int n_attributes = 22;

gpu_selector selector;
queue q(selector, property::queue::enable_profiling{});

double sum_times1;
double sum_times2;

size_t N_inst = n_instances;
size_t N_attr = n_attributes;

std::vector<float> instances(N_inst * N_attr);
std::vector<int> results(n_instances, -1);
std::vector<float> centers(k * N_attr);


// Loads instances from CSV file into the 'instances' array
void load_instances() {
    int instance_count = 0;
    string line;
    string aux;
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
                instances[instance_count * n_attributes + j] = std::stod(aux);
            } catch (const std::exception& e) {
                std::cerr << "Error converting: '" << aux << "'. " << e.what() << std::endl;
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
    for (int i = 0; i < k; i++) {
        c[i] = 0;
    }
    for (int i = 0; i < n_instances; i++) {
        c[results[i]]++;
    }
}

// Assigns each instance to the nearest cluster center (runs on GPU using SYCL)
void assign_cluster(buffer<float, 1>& instances_buf, buffer<float, 1>& centers_buf) {
    buffer results_buf(results);
    event e = q.submit([&](handler& h) {
        accessor A(instances_buf, h, read_only);
        accessor B(centers_buf, h, read_only);
        accessor C(results_buf, h, write_only);
        size_t N = (size_t)n_instances;
        size_t L = 256;
        size_t G = (size_t)n_instances;
        h.parallel_for(sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_id(0);
            if (i >= N) return;
            float min_dist = INFINITY;
            float distance;
            for (int j = 0; j < k; j++) {
                float sum = 0;
                for (int n = 0; n < n_attributes; n++) {
                    float diff = A[i * n_attributes + n] - B[j * n_attributes + n];
                    sum += diff * diff;
                }
                distance = sum;
                if (distance < min_dist) {
                    min_dist = distance;
                    C[i] = j;
                }
            }
        });
    });
}

// Updates cluster centers based on assigned instances (runs on GPU using SYCL)
void update_centers(buffer<float, 1>& center_buf, int* c, buffer<float, 1>& instance_buf, int seed) {
    count_instances(c);
    std::vector<float> sum(k * n_attributes, 0.0f);
    int cluster;
    for (int j = 0; j < n_instances; j++) {
        cluster = results[j];
        for (int a = 0; a < n_attributes; a++) {
            sum[cluster * n_attributes + a] += instances[j * n_attributes + a];
        }
    }
    buffer sum_buf(sum);
    buffer<int> buf_c(c, range<1>(k));
    event e = q.submit([&](handler& h) {
        accessor A(sum_buf, h, read_only);
        accessor B(center_buf, h, write_only, sycl::noinit);
        accessor C(buf_c, h, read_only);
        accessor D(instance_buf, h, read_only);
        size_t N = (size_t)k * (size_t)n_attributes;
        size_t L = 32;
        size_t G = 64;
        h.parallel_for(sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_id(0);
            if (i >= N) return;
            int cluster = i / n_attributes;
            int attribute = i % n_attributes;
            int count = C[cluster];
            // If cluster has instances, compute mean; otherwise, reinitialize center
            if (count != 0) {
                B[i] = A[i] / (float)count;
            } else {
                int idx = (seed + cluster) % n_instances;
                B[i] = D[idx * n_attributes + attribute];
            }
        });
    });
}

// Main k-means loop: loads data, initializes centers, assigns clusters, updates centers
void kmeans() {
    load_instances();
    initialize_centers();
    int c[k];
    buffer instance_buf(instances);
    buffer center_buf(centers);
    for (int i = 0; i < 30; i++) {
        assign_cluster(instance_buf, center_buf);
        update_centers(center_buf, c, instance_buf, i);
    }
}

int main() {
    kmeans();
    for (int i = 0; i < n_instances; ++i) {
        cout << results[i] << std::endl;
    }
}
