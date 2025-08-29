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
static const int n_instancias = 4000000;
static const int n_atributos  = 22;

vector<float> instancias((size_t)n_instancias * n_atributos);
vector<int>   resultados(n_instancias, -1);
vector<float> centros((size_t)k * n_atributos);

void cria_instancias(){
    int c_instancias = 0;
    string linha, linha2;
    string aux;
    std::ifstream arquivo("./preprocessamento/32_dados.csv");
    getline(arquivo, linha);
    while (getline(arquivo, linha)  && c_instancias < n_instancias) {
        int p = 0;
        for (int j = 0; j < n_atributos; j++) {
            aux = "";
            while (p < linha.size() && linha[p] != ',') {
                aux += linha[p];
                p++;
            }
            try {
                instancias[c_instancias * n_atributos + j] = std::stod(aux);
            } catch (const std::exception& e) {
                std::cerr << "Erro ao converter: '" << aux << "'. " << e.what() << std::endl;
                instancias[c_instancias * n_atributos + j] = 0.0f;
            }
            p++;
        }
        c_instancias++;
    }
    printf("Instâncias carregadas: %d \n",c_instancias);
}

void inicializa_centros(){
    srand(3);
    for (int i = 0; i < k; i++) {
        int indicie = rand()%n_instancias;
        for (int j = 0;j < n_atributos;j++){
            centros[i*n_atributos+j] = instancias[indicie*n_atributos+j];
        }
    }
}

void atribui_cluster() {
    const float INF = numeric_limits<float>::infinity();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_instancias; ++i) {
        const float* xi = &instancias[(size_t)i * n_atributos];
        float best = INF; int bestk = 0;

        for (int j = 0; j < k; ++j) {
            const float* cj = &centros[(size_t)j * n_atributos];
            float d2 = 0.0f;
            for (int a = 0; a < n_atributos; ++a) {
                float d = xi[a] - cj[a];
                d2 += d * d;
            }
            if (d2 < best) { best = d2; bestk = j; }
        }
        resultados[i] = bestk;
    }
}

void conta_instancias(int* c) {
    for (int i = 0; i < k; ++i) c[i] = 0;
    for (int i = 0; i < n_instancias; ++i) ++c[resultados[i]];
}

void atualiza_centros(int seed_iter) {
    vector<float> soma((size_t)k * n_atributos, 0.0f);
    int c[k]; conta_instancias(c);

    for (int j = 0; j < n_instancias; ++j) {
        int cl = resultados[j];
        const float* xj = &instancias[(size_t)j * n_atributos];
        float* s = &soma[(size_t)cl * n_atributos];
        for (int a = 0; a < n_atributos; ++a) s[a] += xj[a];
    }

    const int N = k * n_atributos;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; ++t) {
        int cl = t / n_atributos;
        int a  = t % n_atributos;
        int cnt = c[cl];

        if (cnt != 0) {
            centros[t] = soma[t] / (float)cnt;
        } else {
            int idx = (seed_iter + cl) % n_instancias;
            centros[t] = instancias[(size_t)idx * n_atributos + a];
        }
    }
}
void kmeans(){
    cria_instancias();
    inicializa_centros();
    for (int it = 0; it < 30; ++it) {
        atribui_cluster();
        atualiza_centros(it);
    }
}


int main() {
    kmeans();
    for (int i = 0; i < n_instancias; ++i) {
        cout << "Instância " << i << " atribuída ao cluster " << resultados[i] << endl;
    }
    
    
}