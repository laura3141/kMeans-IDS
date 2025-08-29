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
static const int n_instancias = 4000000;
static const int n_atributos = 22;

size_t N_inst = n_instancias;
size_t N_atr = n_atributos;

double tempo_atribui =0;
double tempo_atualiza=0;

vector<double> instancias(N_inst * N_atr);
vector<int> resultados(n_instancias, -1);
vector<double> centros(k * N_atr);
vector<int> labels(n_instancias);

void cria_instancias() {
    int c_instancias = 0;
    string linha, linha2, aux;

    std::ifstream arquivo("./preprocessamento/32_dados.csv");

    getline(arquivo, linha);

    while (getline(arquivo, linha)&&c_instancias < n_instancias) {

        int p = 0;
        for (int j = 0; j < n_atributos; j++) {
            aux = "";
            while (p < linha.size() && linha[p] != ',') {
                aux += linha[p];
                p++;
            }

            try {
                instancias[c_instancias * n_atributos + j] = stof(aux);
            } catch (const exception& e) {
                cerr << "Erro ao converter: '" << aux << "'. " << e.what() << endl;
                instancias[c_instancias * n_atributos + j] = 0.0f;
            }
            p++;
        }

        c_instancias++;
    }

    printf("Instâncias carregadas: %d \n", c_instancias);
}

void inicializa_centros() {
    srand(3);
    for (int i = 0; i < k; i++) {
        int indice = rand() % n_instancias;
        for (int j = 0; j < n_atributos; j++) {
            centros[i * n_atributos + j] = instancias[indice * n_atributos + j];
        }
    }
}

void conta_instancias(int* c) {
    for (int i = 0; i < k; i++)
        c[i] = 0;
    for (int i = 0; i < n_instancias; i++)
        c[resultados[i]]++;
}

void atribui_cluster() {
    for (int i = 0; i < n_instancias; i++) {
        double menor = INFINITY;
        int melhor_cluster = -1;

        for (int j = 0; j < k; j++) {
            double soma = 0;
            for (int n = 0; n < n_atributos; n++) {
                double diff = instancias[i * n_atributos + n] - centros[j * n_atributos + n];
                soma += diff * diff;
            }

            if (soma < menor) {
                menor = soma;
                melhor_cluster = j;
            }
        }

        resultados[i] = melhor_cluster;
    }
}

void atualiza_centros(int* c) {
    conta_instancias(c);

    vector<double> soma(k * n_atributos, 0.0f);

    for (int j = 0; j < n_instancias; j++) {
        int cluster = resultados[j];
        for (int a = 0; a < n_atributos; a++) {
            soma[cluster * n_atributos + a] += instancias[j * n_atributos + a];
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n_atributos; j++) {
            int idx = i * n_atributos + j;
            centros[idx] = (c[i] != 0) ? (soma[idx] / c[i]) : 0.0f;
        }
    }
}
int avaliar_por_permutacao() {
    vector<int> perm(k);
    iota(perm.begin(), perm.end(), 0);

    int max_acertos = 0;

    do {
        int acertos = 0;
        for (int i = 0; i < n_instancias; ++i) {
            int rotulo_predito = perm[resultados[i]];
            if (rotulo_predito == labels[i])
                acertos++;
        }
        if (acertos > max_acertos)
            max_acertos = acertos;

    } while (next_permutation(perm.begin(), perm.end()));

    return max_acertos; 
}

void kmeans() {
    cria_instancias();
    inicializa_centros();
    int c[k];

    for (int i = 0; i < 30; i++) {
        atribui_cluster();
        atualiza_centros(c);
    }
}

int main() {
    kmeans();
    for (int i = 0; i < n_instancias; ++i) {
        cout << "Instância " << i << " atribuída ao cluster " << resultados[i] << std::endl;
    }
    return 0;
}