// memnoise_stats — muestrea N veces (programación + lectura) un conjunto de pesos con los modelos de memristor de
// memristor_noise.hpp y escribe media y desviación estándar POR PESO. Es la contraparte C++ de
// eval_results/compare_with_aihwkit.py (que compara estos números con los de aihwkit real). No simula redes ni entrena.
//
//   ./memnoise_stats --model reramcmo|pcm --weights w1,w2,... --w-max V --t T --n N
//       [--mapping legacy|pair|single] [--g-max V] [--g-min V] [--prog S] [--read S] [--drift S] [--seed K]
// Salida: una línea "mean sd" por peso. Un chip nuevo (programación) por muestra, como apply_noise de aihwkit.
#include "../memristor_noise.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model = "reramcmo", mapping = "single", wcsv;
    double gMax = 88.199997, gMin = 9.0, wMax = 1.0, t = 1.0, prog = 1.0, read = 1.0, drift = 1.0;
    long n = 1000, seed = 1;
    bool haveGMax = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::cerr << a << " necesita un valor\n"; std::exit(1); } return argv[++i]; };
        if      (a == "--model")   model = next();
        else if (a == "--mapping") mapping = next();
        else if (a == "--weights") wcsv = next();
        else if (a == "--w-max")   wMax = std::stod(next());
        else if (a == "--t")       t = std::stod(next());
        else if (a == "--n")       n = std::stol(next());
        else if (a == "--g-max")   { gMax = std::stod(next()); haveGMax = true; }
        else if (a == "--g-min")   gMin = std::stod(next());
        else if (a == "--prog")    prog = std::stod(next());
        else if (a == "--read")    read = std::stod(next());
        else if (a == "--drift")   drift = std::stod(next());
        else if (a == "--seed")    seed = std::stol(next());
        else { std::cerr << "argumento desconocido: " << a << "\n"; return 1; }
    }
    std::vector<double> w;
    { std::stringstream ss(wcsv); std::string tok; while (std::getline(ss, tok, ',')) if (!tok.empty()) w.push_back(std::stod(tok)); }
    if (w.empty() || n < 2) { std::cerr << "faltan --weights o --n >= 2\n"; return 1; }
    if (model == "pcm" && !haveGMax) gMax = 25.0;

    memnoise::CmoMapping mp = memnoise::CmoMapping::Single;
    if      (mapping == "legacy") mp = memnoise::CmoMapping::Legacy;
    else if (mapping == "pair")   mp = memnoise::CmoMapping::Pair;
    else if (mapping == "single") mp = memnoise::CmoMapping::Single;
    else { std::cerr << "--mapping debe ser legacy, pair o single\n"; return 1; }

    std::vector<double> s(w.size(), 0.0), s2(w.size(), 0.0), o;
    for (long k = 0; k < n; ++k) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed) * 1000003ULL + static_cast<uint64_t>(k));
        if (model == "reramcmo") {
            memnoise::ReRamCMODeviceModel d;
            d.g_max = gMax; d.g_min = gMin; d.mapping = mp;
            d.prog_noise_scale = prog; d.read_noise_scale = read; d.drift_scale = drift;
            memnoise::ReRamCMONoisyCrossbar x(w, wMax, d);
            x.program(rng); x.read_weights(rng, t, o);
        } else if (model == "pcm") {
            memnoise::PCMDeviceModel d;
            d.g_max = gMax; d.prog_noise_scale = prog; d.read_noise_scale = read; d.drift_scale = drift;
            memnoise::PCMNoisyCrossbar x(w, wMax, d);
            x.program(rng); x.read_weights(rng, t, o);
        } else { std::cerr << "--model debe ser reramcmo o pcm\n"; return 1; }
        for (size_t i = 0; i < w.size(); ++i) { s[i] += o[i]; s2[i] += o[i] * o[i]; }
    }
    std::printf("%s\n", "mean sd");
    for (size_t i = 0; i < w.size(); ++i) {
        const double m = s[i] / n, var = std::max(0.0, (s2[i] - n * m * m) / (n - 1));
        std::printf("%.12g %.12g\n", m, std::sqrt(var));
    }
    return 0;
}
