#include  <CL/sycl.hpp>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ittnotify.h>
#include <oneapi/tbb/global_control.h>

using namespace std;
using namespace sycl;

static const int k = 2;
static const int n_instancias = 4000000;
static const int n_atributos = 22;

cpu_selector seletor;
queue q(seletor, property::queue::enable_profiling{});

size_t N_inst = n_instancias;
size_t N_atr = n_atributos;

std::vector<float> instancias(N_inst*N_atr);
std::vector<int> resultados(n_instancias,-1);
std::vector<float> centros(k*N_atr);

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

void atribui_cluster(buffer<float, 1>& instancias, buffer<float, 1>& centros){
    buffer resultado(resultados);
    range<1> global_size(n_instancias);
    event e = q.submit([&](handler & h ){
        accessor A(instancias,h,read_only);
        accessor B(centros,h,read_only);
        accessor C(resultado,h,write_only);
        h.parallel_for(range<1>{global_size},[=](id <1>i){
            float menor = INFINITY;
            float distancia;
            for(int j = 0; j < k;j++){
                float soma = 0;
                for(int n = 0;n<n_atributos;n++){
                    soma += ((A[i*n_atributos+n]-B[j*n_atributos+n]) * (A[i*n_atributos+n]-B[j*n_atributos+n]));
                }
                distancia = soma;
                if(distancia < menor){
                    menor = distancia;
                    C[i]=j;
                }
            }
        });
    });
    e.wait();
}

void atualiza_centros(buffer<float,1>& centro, int* c,buffer<float,1>& instancia,int seed) {
    conta_instancias(c);
    std::vector<float> soma(k * n_atributos, 0.0f);
    int cluster;
    for (int j = 0; j < n_instancias; j++) {
        cluster = resultados[j];
        for (int a = 0; a < n_atributos; a++) {
            soma[cluster * n_atributos + a] += instancias[j * n_atributos + a];
        }
    }
    buffer somaB(soma);
    buffer<int> buf_c(c, range<1>(k));
    range<1> global_size(k*n_atributos);
    event e = q.submit([&](handler& h) {
        accessor A(somaB, h, read_only);
        accessor B(centro, h, write_only, sycl::noinit);
        accessor C(buf_c, h, read_only);
        accessor D(instancia,h,read_only);
        h.parallel_for(range<1>{global_size},[=](id <1>i){
            int cluster  = i / n_atributos;
            int atributo = i % n_atributos;
            int count    = C[cluster];
            if (count != 0) {
                B[i] = A[i] / (float)count;
            } else {
                int idx = (seed + cluster) % n_instancias;
                B[i] = D[idx * n_atributos + atributo];
            }
        });
    });
    e.wait();
}

void kmeans(){
    cria_instancias();

    inicializa_centros();
    int c[k];
    buffer instancia(instancias);
    buffer centro (centros);
    for(int i=0;i<30;i++){
        atribui_cluster(instancia, centro);
        atualiza_centros(centro,c,instancia,i);
    }

}

int main(){
    kmeans();
    for (int i = 0; i < n_instancias; ++i) {
        cout << "Instância " << i << " atribuída ao cluster " << resultados[i] << std::endl;
    }
}