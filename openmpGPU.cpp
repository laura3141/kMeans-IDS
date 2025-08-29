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
static const int n_instancias = 4000000;
static const int n_atributos = 22;

size_t N_inst = n_instancias; 
size_t N_atr = n_atributos;

std::vector<int> labels(n_instancias);

float *instancias = new float[N_inst*N_atr];
int *resultados = new int[n_instancias];
float *centros = new float[k*N_atr];


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

void conta_instancias(int* c){
    for(int i=0;i<k;i++){
        c[i]=0;
    }
    for(int i=0;i<n_instancias;i++){
        c[resultados[i]]++;
    }
}

void atribui_cluster(){
    size_t Nia = N_inst*N_atr;
    size_t NaSete = k*N_atr;
    #pragma omp target data map(to: instancias[0:Nia], centros[0:NaSete]) map(tofrom: resultados[0:n_instancias])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < n_instancias; i++){
            float menor = INFINITY;
            float distancia;
            for(int j = 0; j < k;j++){
                float soma = 0;
                for(int n = 0; n<n_atributos;n++){
                    float diff = instancias[i*n_atributos+n] - centros[j*n_atributos+n];
                    soma += diff * diff;
                }
                distancia = soma;
                if(distancia < menor){    
                    menor = distancia;
                    resultados[i]=j;
                }
            }
        }
    }
}

void atualiza_centros(int* c) {
    conta_instancias(c);
    float soma[k][n_atributos];
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n_atributos; j++) {
            soma[i][j] = 0.0f;
        }
    }
    for(int i=0;i<n_instancias;i++){
        for(int j=0;j<n_atributos;j++){
            soma[resultados[i]][j]+=instancias[i*n_atributos+j];
        }
    }
    #pragma omp target teams distribute map(to: c[0:k]) map(tofrom: soma[0:k][0:n_atributos])
    for(int i = 0; i < k; i++) {
        #pragma omp parallel for
        for(int j = 0; j < n_atributos; j++) {
            if (c[i] != 0) {
                soma[i][j] = soma[i][j] / c[i];
            }
        }
    }
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n_atributos; j++) {
            centros[i * n_atributos + j] = soma[i][j];
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

void kmeans(){
    int c[k];
    cria_instancias();
    inicializa_centros();
    for(int i=0;i<30;i++){
        atribui_cluster(); 
        atualiza_centros(c); 
    }
}

int main (){
    kmeans();
    for (int i = 0; i < n_instancias; ++i) {
        cout << "Instância " << i << " atribuída ao cluster " << resultados[i] << endl;
    }
    delete[] instancias;
    delete[] resultados;
    delete[] centros;
    return 0;
}