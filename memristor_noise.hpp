// memristor_noise.hpp
// Modelo de ruido de memristores para evaluar redes (p. ej. WANN) en C++17.
//
// Dos fuentes de ruido, con distinta naturaleza temporal:
//   1) Ruido de ESCRITURA (programación): se muestrea UNA vez por "chip cargado"
//      y queda fijo durante toda la inferencia.
//   2) Ruido de LECTURA: se re-muestrea en CADA inferencia (cada lectura del dispositivo).
//
// Mapeo de pesos con par diferencial: w = (G+ - G-) * (w_max / g_max).
//
// Este archivo tiene cuatro modelos independientes:
//   - DeviceModel / NoisyCrossbar: modelo simple de dos coeficientes (placeholder original,
//     útil para barridos de sensibilidad genéricos; sus coeficientes NO están calibrados
//     contra ningún dispositivo real — ver TODO).
//   - PCMDeviceModel / PCMNoisyCrossbar: puerto literal de
//     `aihwkit.inference.noise.pcm.PCMLikeNoiseModel` (IBM aihwkit, licencia MIT),
//     ajustado a mediciones reales de PCM — Nandakumar et al., ICECS (2019). Incluye
//     ruido de programación, deriva (drift) por ley de potencia y ruido de lectura
//     acumulado (1/f).
//   - ReRamWan2022DeviceModel / ReRamWan2022NoisyCrossbar: puerto literal de
//     `aihwkit.inference.noise.reram.ReRamWan2022NoiseModel` (misma librería/licencia),
//     ajustado a datos publicados por Wan et al., Nature (2022) para ReRAM — sin ley de
//     deriva continua, solo tres instantes de tiempo medidos (1 s / 1 día / 2 días).
//   - ReRamCMODeviceModel / ReRamCMONoisyCrossbar: puerto literal de
//     `aihwkit.inference.noise.reram.ReRamCMONoiseModel` (misma librería/licencia),
//     ajustado a dispositivos ReRAM filamentarios CMO/HfOx (Falcone et al., IBM Research
//     Europa-Zúrich). A diferencia de los otros dos: SÍ tiene deriva continua en log(t)
//     (como PCM), pero `g_max`/`g_min` no tienen valor por defecto en aihwkit (hay que
//     fijarlos con mediciones propias) y el polinomio de programación no se normaliza por
//     g_max — ver las notas junto a la struct.
//
// Fuente de los modelos portados (verificada línea a línea contra el código fuente,
// revisión "master" leída el 2026-10-08/2026-10-14):
//   https://github.com/IBM/aihwkit/blob/master/src/aihwkit/inference/noise/pcm.py
//   https://github.com/IBM/aihwkit/blob/master/src/aihwkit/inference/noise/reram.py
//   https://github.com/IBM/aihwkit/blob/master/src/aihwkit/inference/noise/base.py
// aihwkit es MIT license (IBM); este puerto reimplementa solo las fórmulas escalares
// (sin Tensor/batch, sin PyTorch) para encajar con el resto de este header.
//
// TODO: los coeficientes de DeviceModel (el modelo simple) son PLACEHOLDERS. Los de
//       PCMDeviceModel/ReRamWan2022DeviceModel/ReRamCMODeviceModel SÍ están ajustados a
//       dispositivos reales (ver las citas arriba), pero siguen siendo genéricos de la
//       literatura, no calibrados al chip/dispositivo específico de este proyecto —
//       reemplazar si se dispone de esas mediciones.

#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace memnoise {

// Niveles finitos de conductancia: un dispositivo real solo puede asentarse en uno
// de N estados distinguibles entre 0 y g_max (resolución finita de programación/
// verificación), no en cualquier valor continuo. n_levels <= 1 = sin cuantizar
// (comportamiento de siempre, preservado exactamente). Se aplica solo al estado
// YA ESCRITO (en program(), después del ruido de programación) — no en cada
// lectura, que sigue siendo una medición continua de ese estado físico fijo.
namespace detail {
inline double quantizeConductance(double g, double g_max, int n_levels) {
    if (n_levels <= 1) return g;
    const double step = g_max / static_cast<double>(n_levels - 1);
    return std::clamp(std::round(g / step) * step, 0.0, g_max);
}
}  // namespace detail

struct DeviceModel {
    double g_max = 1.0;                 // conductancia máxima (unidad arbitraria, p. ej. µS)

    // sigma_prog(g) = g_max * max(c0 + c1*u + c2*u^2, 0),  u = g / g_max  (dependiente del estado)
    double c0 = 0.03, c1 = 0.0, c2 = 0.0;

    // sigma_read(g) = g_max * r0 * u^read_exp
    //   read_exp = 0   -> ruido constante
    //   read_exp = 0.5 -> dependencia ~sqrt(g)
    double r0 = 0.01;
    double read_exp = 0.5;

    // Niveles de conductancia distinguibles al programar (<=1 = continuo).
    int n_levels = 0;

    double sigma_prog(double g) const {
        const double u = std::clamp(g / g_max, 0.0, 1.0);
        return g_max * std::max(c0 + c1 * u + c2 * u * u, 0.0);
    }
    double sigma_read(double g) const {
        const double u = std::clamp(g / g_max, 0.0, 1.0);
        return g_max * r0 * (read_exp == 0.0 ? 1.0 : std::pow(u, read_exp));
    }
};

class NoisyCrossbar {
public:
    // weights: pesos lógicos (una entrada por conexión activa). w_max: |w| que mapea a g_max.
    NoisyCrossbar(std::vector<double> weights, double w_max, DeviceModel dev)
        : dev_(dev), w_max_(w_max), n_(weights.size()),
          g_target_pos_(n_), g_target_neg_(n_), g_prog_pos_(n_), g_prog_neg_(n_) {
        for (size_t i = 0; i < n_; ++i) {
            const double g = std::clamp(std::abs(weights[i]) / w_max_, 0.0, 1.0) * dev_.g_max;
            g_target_pos_[i] = weights[i] > 0 ? g : 0.0;
            g_target_neg_[i] = weights[i] < 0 ? g : 0.0;
        }
        g_prog_pos_ = g_target_pos_;
        g_prog_neg_ = g_target_neg_;
    }

    // (1) Ruido de carga: llamar UNA vez por instancia de chip simulada.
    template <class RNG>
    void program(RNG& rng) {
        std::normal_distribution<double> Z(0.0, 1.0);
        auto draw = [&](double g) {
            const double gp = std::clamp(g + dev_.sigma_prog(g) * Z(rng), 0.0, dev_.g_max);
            return detail::quantizeConductance(gp, dev_.g_max, dev_.n_levels);
        };
        for (size_t i = 0; i < n_; ++i) {
            g_prog_pos_[i] = draw(g_target_pos_[i]);
            g_prog_neg_[i] = draw(g_target_neg_[i]);
        }
    }

    // (2) Ruido de lectura: llamar en CADA inferencia. Devuelve los pesos efectivos.
    template <class RNG>
    void read_weights(RNG& rng, std::vector<double>& out) const {
        out.resize(n_);
        std::normal_distribution<double> Z(0.0, 1.0);
        auto rd = [&](double g) {
            return std::clamp(g + dev_.sigma_read(g) * Z(rng), 0.0, dev_.g_max);
        };
        const double scale = w_max_ / dev_.g_max;
        for (size_t i = 0; i < n_; ++i)
            out[i] = (rd(g_prog_pos_[i]) - rd(g_prog_neg_[i])) * scale;
    }

    size_t size() const { return n_; }

private:
    DeviceModel dev_;
    double w_max_;
    size_t n_;
    std::vector<double> g_target_pos_, g_target_neg_, g_prog_pos_, g_prog_neg_;
};

// ===========================================================================
// Puerto de aihwkit: ruido de programación polinómico compartido
// ===========================================================================
//
// Tanto PCMLikeNoiseModel.apply_programming_noise_to_conductance como
// ReRamWan2022NoiseModel._apply_poly en aihwkit implementan exactamente el mismo cuerpo
// (están duplicados en el Python original, no comparten función): un polinomio en
// potencias ascendentes de u = g_target/g_max, reescalado por g_max/g_max_ref, usado como
// desviación estándar de un ruido gaussiano aditivo sobre g_target, recortado a >= 0 (sin
// cota superior — así está en aihwkit, no es un descuido de este puerto).
namespace detail {

template <class RNG>
double applyPolyProgrammingNoise(double g_target, const std::vector<double>& coeff,
                                  double g_max, double g_max_ref, double noise_scale,
                                  RNG& rng) {
    double mat = 1.0;
    double sig = coeff.empty() ? 0.0 : coeff[0];
    for (size_t i = 1; i < coeff.size(); ++i) {
        mat *= g_target / g_max;
        sig += mat * coeff[i];
    }
    sig *= g_max / g_max_ref;
    std::normal_distribution<double> Z(0.0, 1.0);
    return std::max(g_target + noise_scale * sig * Z(rng), 0.0);
}

}  // namespace detail

// ===========================================================================
// Puerto de PCMLikeNoiseModel (aihwkit.inference.noise.pcm), ajustado a PCM real —
// Nandakumar et al., ICECS (2019): https://ieeexplore.ieee.org/abstract/document/8964852
// ===========================================================================
struct PCMDeviceModel {
    double g_max = 25.0;  // µS; |w_max| del genoma se mapea a este valor.

    // Coeficientes del polinomio de ruido de programación (ascendentes: c0 + c1*u + c2*u^2,
    // u = g/g_max), ajustados por IBM a g_max de referencia = 25 µS.
    std::vector<double> prog_coeff = {0.26348, 1.9650, -1.1731};
    double prog_coeff_g_max_reference = 25.0;
    double prog_noise_scale = 1.0;
    double read_noise_scale = 1.0;
    double drift_scale = 1.0;

    double t_read = 250.0e-9;  // s; parámetro del ajuste de ruido 1/f.
    double t_0 = 20.0;         // s; "primer instante de lectura" del ajuste de deriva.

    // Niveles de conductancia distinguibles al programar (<=1 = continuo, por
    // defecto — así venía el modelo original de aihwkit, que asume conductancia
    // continua). No representa ningún dato de Nandakumar et al.; es un mecanismo
    // adicional para estudiar el efecto de una resolución de programación finita.
    int n_levels = 0;
};

// Mismo par diferencial (G+/G-) que NoisyCrossbar, pero con deriva (drift) por ley de
// potencia y ruido de lectura acumulado dependiente del tiempo transcurrido desde la
// programación (t_inference, en segundos). El coeficiente de deriva nu se sortea UNA vez
// por rama en program() y se reutiliza en todas las lecturas de ese chip — igual que
// aihwkit separa apply_programming_noise() (una vez) de apply_drift_noise() (por lectura):
// la función de conveniencia apply_noise() de aihwkit, en cambio, vuelve a sortear nu en
// cada llamada, y por tanto NO sirve para modelar un mismo chip leído varias veces (así lo
// dice su propio docstring).
class PCMNoisyCrossbar {
public:
    PCMNoisyCrossbar(std::vector<double> weights, double w_max, PCMDeviceModel dev)
        : dev_(dev), w_max_(w_max), n_(weights.size()),
          g_target_pos_(n_), g_target_neg_(n_), g_prog_pos_(n_), g_prog_neg_(n_),
          nu_pos_(n_), nu_neg_(n_) {
        for (size_t i = 0; i < n_; ++i) {
            const double g = std::clamp(std::abs(weights[i]) / w_max_, 0.0, 1.0) * dev_.g_max;
            g_target_pos_[i] = weights[i] > 0 ? g : 0.0;
            g_target_neg_[i] = weights[i] < 0 ? g : 0.0;
        }
        g_prog_pos_ = g_target_pos_;
        g_prog_neg_ = g_target_neg_;
    }

    // (1) Ruido de escritura + sorteo de nu (deriva): llamar UNA vez por chip simulado.
    template <class RNG>
    void program(RNG& rng) {
        for (size_t i = 0; i < n_; ++i) {
            g_prog_pos_[i] = detail::quantizeConductance(detail::applyPolyProgrammingNoise(
                g_target_pos_[i], dev_.prog_coeff, dev_.g_max, dev_.prog_coeff_g_max_reference,
                dev_.prog_noise_scale, rng), dev_.g_max, dev_.n_levels);
            g_prog_neg_[i] = detail::quantizeConductance(detail::applyPolyProgrammingNoise(
                g_target_neg_[i], dev_.prog_coeff, dev_.g_max, dev_.prog_coeff_g_max_reference,
                dev_.prog_noise_scale, rng), dev_.g_max, dev_.n_levels);
            // nu se sortea a partir del target ORIGINAL (pre-ruido), igual que aihwkit
            // (generate_drift_coefficients se llama con g_target, no con g_prog).
            nu_pos_[i] = sampleDriftNu(g_target_pos_[i], rng);
            nu_neg_[i] = sampleDriftNu(g_target_neg_[i], rng);
        }
    }

    // (2) Deriva + ruido de lectura acumulado (1/f) a t_inference segundos de la
    // programación. Llamar en CADA inferencia; nu es fijo, el ruido de lectura es nuevo
    // cada vez.
    template <class RNG>
    void read_weights(RNG& rng, double t_inference, std::vector<double>& out) const {
        out.resize(n_);
        const double scale = w_max_ / dev_.g_max;
        for (size_t i = 0; i < n_; ++i) {
            const double gp = driftAndRead(g_prog_pos_[i], nu_pos_[i], t_inference, rng);
            const double gn = driftAndRead(g_prog_neg_[i], nu_neg_[i], t_inference, rng);
            out[i] = (gp - gn) * scale;
        }
    }

    size_t size() const { return n_; }

private:
    // generate_drift_coefficients: nu = |mu(u) + sigma(u)*Z|, u = |g_target|/g_max,
    // con mu/sigma ajustados por tramos (en log(u)) y recortados a los rangos medidos.
    template <class RNG>
    double sampleDriftNu(double g_target, RNG& rng) const {
        constexpr double kZeroClip = 1e-7;  // evita log(0) cuando g_target == 0
        const double g_rel = std::max(std::fabs(g_target) / dev_.g_max, kZeroClip);
        const double lg = std::log(g_rel);
        const double mu = std::clamp(-0.0155 * lg + 0.0244, 0.049, 0.1);
        const double sig = std::clamp(-0.0125 * lg - 0.0059, 0.008, 0.045);
        std::normal_distribution<double> Z(0.0, 1.0);
        return dev_.drift_scale * std::fabs(mu + sig * Z(rng));
    }

    // apply_drift_noise_to_conductance: deriva por ley de potencia desde g_prog, más ruido
    // 1/f acumulado cuya varianza crece con log(tiempo transcurrido desde la programación).
    template <class RNG>
    double driftAndRead(double g_prog, double nu, double t_inference, RNG& rng) const {
        const double t = t_inference + dev_.t_0;
        const double g_drift = (t > dev_.t_0) ? g_prog * std::pow(t / dev_.t_0, -nu) : g_prog;
        const double q_s = std::min(
            0.0088 / std::max(std::pow(std::fabs(g_prog) / dev_.g_max, 0.65), 1e-3), 0.2);
        const double sig_noise = q_s * std::sqrt(std::log((t + dev_.t_read) / (2.0 * dev_.t_read)));
        std::normal_distribution<double> Z(0.0, 1.0);
        const double g_final =
            g_drift + std::fabs(g_drift) * dev_.read_noise_scale * sig_noise * Z(rng);
        return std::max(g_final, 0.0);
    }

    PCMDeviceModel dev_;
    double w_max_;
    size_t n_;
    std::vector<double> g_target_pos_, g_target_neg_, g_prog_pos_, g_prog_neg_, nu_pos_, nu_neg_;
};

// ===========================================================================
// Puerto de ReRamWan2022NoiseModel (aihwkit.inference.noise.reram), ajustado a datos
// publicados de ReRAM — Wan et al., Nature (2022): https://www.nature.com/articles/s41586-022-04992-8
// ===========================================================================
//
// A diferencia de PCM, este modelo NO tiene una ley de deriva continua: los autores solo
// disponían de mediciones en tres instantes fijos (1 s, 1 día, 2 días), cada uno con su
// propio ajuste polinómico de 4º grado. read_weights() debe llamarse con t_inference
// EXACTAMENTE igual a una de esas claves (igual que aihwkit, que lanza ArgumentError en
// caso contrario) — no hay forma de interpolar a un instante intermedio con este modelo.
//
// Nota sobre un valor por defecto de aihwkit: su __init__, cuando no se pasa coeff_dic,
// solo asigna el atributo `prog_coeff_g_max_reference = 40.0`, pero `_apply_poly` en
// realidad lee `self.coeff_g_max_reference` (fijado antes a `self.g_max`, no a 40.0) —
// un desajuste de nombres que, en la configuración por defecto de aihwkit, deja el
// reescalado g_max/g_max_ref en 1.0 salvo que el g_max usado ya sea 40 µS o el usuario
// pase coeff_g_max_reference=40.0 explícitamente. Aquí se corrige a propósito: g_max y
// coeff_g_max_reference quedan en 40.0 por defecto (el valor de referencia real del
// ajuste, según el propio docstring de aihwkit), para que el reescalado haga lo que el
// paper documenta en vez de reproducir ese bug.
struct ReRamWan2022DeviceModel {
    double g_max = 40.0;  // µS; |w_max| del genoma se mapea a este valor.
    double coeff_g_max_reference = 40.0;
    double noise_scale = 1.0;

    // Niveles de conductancia distinguibles al programar (<=1 = continuo, por
    // defecto). No representa ningún dato de Wan et al.; mecanismo adicional
    // para estudiar el efecto de una resolución de programación finita.
    int n_levels = 0;

    // t_inference (s) -> coeficientes ascendentes [c0..c4] en µS (Wan et al., Nature 2022,
    // fig. ajustada por IBM). std::map ordena por clave ascendente: program() usa siempre
    // la clave más chica (igual que aihwkit: min(coeff_dic.keys())).
    std::map<double, std::vector<double>> coeff_dic = {
        {1.0, {0.348, 16.030, -43.853, 45.393, -16.815}},          // 1 s
        {86400.0, {0.701, 22.086, -50.773, 47.095, -16.458}},      // 1 día
        {172800.0, {0.782, 20.274, -43.507, 37.062, -11.934}},     // 2 días
    };
};

class ReRamWan2022NoisyCrossbar {
public:
    ReRamWan2022NoisyCrossbar(std::vector<double> weights, double w_max,
                               ReRamWan2022DeviceModel dev)
        : dev_(std::move(dev)), w_max_(w_max), n_(weights.size()),
          g_target_pos_(n_), g_target_neg_(n_), g_prog_pos_(n_), g_prog_neg_(n_) {
        if (dev_.coeff_dic.empty())
            throw std::invalid_argument("ReRamWan2022NoisyCrossbar: coeff_dic vacío");
        for (size_t i = 0; i < n_; ++i) {
            const double g = std::clamp(std::abs(weights[i]) / w_max_, 0.0, 1.0) * dev_.g_max;
            g_target_pos_[i] = weights[i] > 0 ? g : 0.0;
            g_target_neg_[i] = weights[i] < 0 ? g : 0.0;
        }
    }

    // (1) Ruido de escritura: llamar UNA vez por chip simulado. Usa el polinomio de la
    // clave de tiempo más baja de coeff_dic (típicamente "1 s"), igual que aihwkit.
    template <class RNG>
    void program(RNG& rng) {
        const auto& coeff0 = dev_.coeff_dic.begin()->second;
        for (size_t i = 0; i < n_; ++i) {
            g_prog_pos_[i] = detail::quantizeConductance(detail::applyPolyProgrammingNoise(
                g_target_pos_[i], coeff0, dev_.g_max, dev_.coeff_g_max_reference,
                dev_.noise_scale, rng), dev_.g_max, dev_.n_levels);
            g_prog_neg_[i] = detail::quantizeConductance(detail::applyPolyProgrammingNoise(
                g_target_neg_[i], coeff0, dev_.g_max, dev_.coeff_g_max_reference,
                dev_.noise_scale, rng), dev_.g_max, dev_.n_levels);
        }
    }

    // (2) Conductancia "a t_inference segundos": re-aplica el polinomio de ruido de la
    // clave de tiempo pedida sobre g_prog (el valor YA escrito en program(), no el target
    // original) — así lo hace aihwkit: no hay una trayectoria desde g_prog, cada instante
    // medido tiene su propia distribución ajustada por separado.
    template <class RNG>
    void read_weights(RNG& rng, double t_inference, std::vector<double>& out) const {
        auto it = dev_.coeff_dic.find(t_inference);
        if (it == dev_.coeff_dic.end())
            throw std::invalid_argument(
                "ReRamWan2022NoisyCrossbar: t_inference debe ser una clave exacta de "
                "coeff_dic (por defecto: 1.0 / 86400.0 / 172800.0 segundos)");
        out.resize(n_);
        const double scale = w_max_ / dev_.g_max;
        for (size_t i = 0; i < n_; ++i) {
            const double gp = detail::applyPolyProgrammingNoise(
                g_prog_pos_[i], it->second, dev_.g_max, dev_.coeff_g_max_reference,
                dev_.noise_scale, rng);
            const double gn = detail::applyPolyProgrammingNoise(
                g_prog_neg_[i], it->second, dev_.g_max, dev_.coeff_g_max_reference,
                dev_.noise_scale, rng);
            out[i] = (gp - gn) * scale;
        }
    }

    size_t size() const { return n_; }

private:
    ReRamWan2022DeviceModel dev_;
    double w_max_;
    size_t n_;
    std::vector<double> g_target_pos_, g_target_neg_, g_prog_pos_, g_prog_neg_;
};

// ===========================================================================
// Puerto de ReRamCMONoiseModel (aihwkit.inference.noise.reram), ajustado a
// dispositivos ReRAM filamentarios CMO/HfOx — Falcone et al., IBM Research
// Europa-Zúrich (citado por aihwkit como "In Review" al momento de portar
// esto, sin DOI público confirmado).
// ===========================================================================
//
// Diferencias reales frente a PCM/ReRamWan2022 (no son descuidos de este puerto,
// así está en aihwkit):
//   - El polinomio de ruido de programación NO se normaliza por g_max — usa
//     g_target crudo, no u=g_target/g_max. El propio código de aihwkit lo deja
//     explícito con un comentario: `mat *= g_target  # / self.g_max`.
//     `coeff_g_max_reference` tampoco existe como parámetro separado: el
//     __init__ de aihwkit lo fija siempre en self.g_max, así que el reescalado
//     g_max/coeff_g_max_reference es SIEMPRE 1 (a diferencia del bug de
//     nombres de ReRamWan2022) — por eso acá ni se calcula, es un no-op.
//   - generate_drift_coefficients() de aihwkit es un no-op ("la relajación de
//     conductancia es independiente del nivel de conductancia"): no hay ningún
//     coeficiente que sortear una vez por chip en program(), a diferencia del
//     nu de PCM.
//   - A t_inference == 0, aihwkit devuelve g_prog SIN NINGÚN cambio (bypass
//     literal) — no "deriva nula + algo de ruido de lectura" como en PCM.
//   - aihwkit usa SingleDeviceConductanceConverter (un dispositivo por peso,
//     con g_min propio) en vez de SinglePairConductanceConverter. Acá se aplica
//     igual por separado a cada rama del par diferencial (g_target_pos_/neg_)
//     para encajar con el resto de este header — adaptación nuestra, no literal.
//   - g_max/g_min NO tienen valor por defecto en aihwkit (su __init__ exige que
//     se pasen, o lanza ValueError) — a diferencia de PCM (25 µS) y ReRamWan2022
//     (40 µS), acá no hay ningún número de referencia que copiar. Se dejan en
//     NaN y el constructor de la crossbar lanza si no se fijan explícitamente,
//     en vez de inventar un valor no verificado.
//   - aihwkit NO recorta g_prog a >= 0 en su _apply_poly (sin clamp inferior),
//     lo que puede dar conductancias negativas que después rompen el
//     log10(g_drift) de apply_drift_noise_to_conductance con NaN. Acá SÍ se
//     recorta (programación a [0, g_max]; drift a un epsilon chico antes del
//     log10) — una adición defensiva nuestra, necesaria en C++ para no
//     propagar NaN silenciosamente; no cambia el resultado en el rango normal
//     de operación, solo evita el caso límite.
struct ReRamCMODeviceModel {
    // Sin valor por defecto confiable — aihwkit tampoco trae uno para este
    // modelo (ValueError si no se pasan). Fijarlos a mediciones reales del
    // propio dispositivo CMO/HfOx antes de usar; NaN hace fallar rápido si se
    // te olvida, en vez de simular con un número inventado.
    double g_max = std::numeric_limits<double>::quiet_NaN();  // µS
    double g_min = std::numeric_limits<double>::quiet_NaN();  // µS

    double prog_noise_scale = 1.0;
    double read_noise_scale = 1.0;
    double drift_scale = 1.0;

    // Ventana de aceptación del lazo de programar-y-verificar (0.2% = más
    // estricta/menos ruido; 2% = más laxa/más ruido) — debe ser una clave
    // exacta de prog_coeff; si no, se usa la más chica disponible (igual que
    // aihwkit: min(coeff_dict.keys())).
    double acceptance_range = 2e-2;

    // acceptance_range -> coeficientes ascendentes [c0,c1] en µS, LINEALES en
    // g_target crudo (no normalizado). Ajustados por IBM a caracterización de
    // dispositivos CMO/HfOx.
    std::map<double, std::vector<double>> prog_coeff = {
        {0.2e-2, {0.00081107, 0.00106879}},
        {2.0e-2, {0.0112185391, 0.01129027418}},
    };

    // Deriva: desplazamiento de la media y ensanchamiento del desvío con
    // log(t_inference), independiente del nivel de conductancia.
    double decay_mean = -0.08900206;
    double decay_std_slope = 0.04201137, decay_std_intercept = 0.41183342;

    // Ruido de lectura acumulado.
    double read_K = 0.0277316483;
    double t_read = 1.0e-6;  // s

    // Niveles de conductancia distinguibles al programar (<=1 = continuo).
    int n_levels = 0;
};

class ReRamCMONoisyCrossbar {
public:
    ReRamCMONoisyCrossbar(std::vector<double> weights, double w_max, ReRamCMODeviceModel dev)
        : dev_(dev), w_max_(w_max), n_(weights.size()),
          g_target_pos_(n_), g_target_neg_(n_), g_prog_pos_(n_), g_prog_neg_(n_) {
        if (!(dev_.g_max > 0.0))
            throw std::invalid_argument(
                "ReRamCMONoisyCrossbar: hay que fijar g_max > 0 — este modelo no trae un "
                "valor por defecto en aihwkit, no inventamos uno acá");
        if (!(dev_.g_min >= 0.0) || !(dev_.g_min < dev_.g_max))
            throw std::invalid_argument("ReRamCMONoisyCrossbar: hay que fijar 0 <= g_min < g_max");
        for (size_t i = 0; i < n_; ++i) {
            const double g = std::clamp(std::abs(weights[i]) / w_max_, 0.0, 1.0) * dev_.g_max;
            g_target_pos_[i] = weights[i] > 0 ? g : 0.0;
            g_target_neg_[i] = weights[i] < 0 ? g : 0.0;
        }
        g_prog_pos_ = g_target_pos_;
        g_prog_neg_ = g_target_neg_;
    }

    // (1) Ruido de escritura: llamar UNA vez por chip simulado. Sin coeficiente
    // de deriva que sortear (generate_drift_coefficients es un no-op en este modelo).
    template <class RNG>
    void program(RNG& rng) {
        const auto& coeff = progCoeff();
        for (size_t i = 0; i < n_; ++i) {
            g_prog_pos_[i] = progNoise(g_target_pos_[i], coeff, rng);
            g_prog_neg_[i] = progNoise(g_target_neg_[i], coeff, rng);
        }
    }

    // (2) Deriva (log-lineal, independiente del estado) + ruido de lectura
    // acumulado, a t_inference segundos de la programación. A t_inference==0
    // devuelve g_prog sin cambios (igual que aihwkit).
    template <class RNG>
    void read_weights(RNG& rng, double t_inference, std::vector<double>& out) const {
        out.resize(n_);
        const double scale = w_max_ / dev_.g_max;
        for (size_t i = 0; i < n_; ++i) {
            const double gp = driftAndRead(g_prog_pos_[i], t_inference, rng);
            const double gn = driftAndRead(g_prog_neg_[i], t_inference, rng);
            out[i] = (gp - gn) * scale;
        }
    }

    size_t size() const { return n_; }

private:
    const std::vector<double>& progCoeff() const {
        auto it = dev_.prog_coeff.find(dev_.acceptance_range);
        if (it == dev_.prog_coeff.end()) it = dev_.prog_coeff.begin();  // min key, igual que aihwkit
        return it->second;
    }

    // _apply_poly: polinomio lineal en g_target CRUDO (sin normalizar por
    // g_max — ver nota de la sección). Recorte a [0, g_max] agregado acá
    // (aihwkit no lo tiene en este modelo).
    template <class RNG>
    double progNoise(double g_target, const std::vector<double>& coeff, RNG& rng) const {
        double mat = 1.0;
        double sig = coeff.empty() ? 0.0 : coeff[0];
        for (size_t i = 1; i < coeff.size(); ++i) {
            mat *= g_target;
            sig += mat * coeff[i];
        }
        std::normal_distribution<double> Z(0.0, 1.0);
        const double g_prog = g_target + dev_.prog_noise_scale * sig * Z(rng);
        return detail::quantizeConductance(std::clamp(g_prog, 0.0, dev_.g_max),
                                            dev_.g_max, dev_.n_levels);
    }

    // apply_drift_noise_to_conductance: desplaza media y desvío con
    // log(t_inference) (sin depender del nivel de conductancia), más ruido de
    // lectura vía log10(g_drift). Recorte a un epsilon chico antes del log10
    // agregado acá para no propagar NaN si el drift empujó g_drift a <= 0
    // (aihwkit no lo tiene).
    template <class RNG>
    double driftAndRead(double g_prog, double t_inference, RNG& rng) const {
        if (t_inference == 0.0) return g_prog;
        std::normal_distribution<double> Z(0.0, 1.0);
        const double lt = std::log(t_inference);
        const double g_mean = g_prog + dev_.decay_mean * lt * dev_.drift_scale;
        const double sigma_relax = dev_.decay_std_slope * lt + dev_.decay_std_intercept;
        const double g_drift = g_mean + Z(rng) * sigma_relax * dev_.drift_scale;
        constexpr double kFloor = 1e-6;  // evita log10(<=0); no está en aihwkit
        const double sigma_read = dev_.read_K * std::log10(std::max(g_drift, kFloor)) *
            std::sqrt(std::log((t_inference + dev_.t_read) / (2.0 * dev_.t_read)));
        const double g_final = g_drift + Z(rng) * sigma_read * dev_.read_noise_scale;
        return std::clamp(g_final, dev_.g_min, dev_.g_max);
    }

    ReRamCMODeviceModel dev_;
    double w_max_;
    size_t n_;
    std::vector<double> g_target_pos_, g_target_neg_, g_prog_pos_, g_prog_neg_;
};

}  // namespace memnoise

#ifdef MEMNOISE_DEMO
#include <cstdio>
int main() {
    using namespace memnoise;
    // WANN: todas las conexiones comparten el mismo peso w (aquí 0.5, con signos distintos).
    std::vector<double> w = {0.5, -0.5, 0.5, 0.5, -0.5};
    std::mt19937_64 rng(42);

    std::printf("-- modelo simple (placeholder) --\n");
    {
        NoisyCrossbar xb(w, /*w_max=*/2.0, DeviceModel{});
        for (int chip = 0; chip < 3; ++chip) {
            xb.program(rng);
            for (int inf = 0; inf < 2; ++inf) {
                std::vector<double> we;
                xb.read_weights(rng, we);
                std::printf("chip %d inf %d:", chip, inf);
                for (double x : we) std::printf(" %+.3f", x);
                std::printf("\n");
            }
        }
    }

    std::printf("-- PCM (aihwkit PCMLikeNoiseModel) --\n");
    {
        PCMNoisyCrossbar xb(w, /*w_max=*/2.0, PCMDeviceModel{});
        for (int chip = 0; chip < 3; ++chip) {
            xb.program(rng);
            for (double t_inference : {0.0, 3600.0, 86400.0}) {  // t=0, 1h, 1 día
                std::vector<double> we;
                xb.read_weights(rng, t_inference, we);
                std::printf("chip %d t=%.0fs:", chip, t_inference);
                for (double x : we) std::printf(" %+.3f", x);
                std::printf("\n");
            }
        }
    }

    std::printf("-- ReRAM (aihwkit ReRamWan2022NoiseModel) --\n");
    {
        ReRamWan2022NoisyCrossbar xb(w, /*w_max=*/2.0, ReRamWan2022DeviceModel{});
        for (int chip = 0; chip < 3; ++chip) {
            xb.program(rng);
            for (double t_inference : {1.0, 86400.0, 172800.0}) {
                std::vector<double> we;
                xb.read_weights(rng, t_inference, we);
                std::printf("chip %d t=%.0fs:", chip, t_inference);
                for (double x : we) std::printf(" %+.3f", x);
                std::printf("\n");
            }
        }
    }

    std::printf("-- ReRAM CMO/HfOx (aihwkit ReRamCMONoiseModel) --\n");
    {
        // g_max/g_min: sin default en aihwkit (ver nota de la struct) — estos
        // dos números son solo ilustrativos para que el demo corra, NO una
        // medición de Falcone et al.
        ReRamCMODeviceModel dev;
        dev.g_max = 25.0;
        dev.g_min = 1.0;
        ReRamCMONoisyCrossbar xb(w, /*w_max=*/2.0, dev);
        for (int chip = 0; chip < 3; ++chip) {
            xb.program(rng);
            for (double t_inference : {0.0, 3600.0, 86400.0}) {  // t=0, 1h, 1 día
                std::vector<double> we;
                xb.read_weights(rng, t_inference, we);
                std::printf("chip %d t=%.0fs:", chip, t_inference);
                for (double x : we) std::printf(" %+.3f", x);
                std::printf("\n");
            }
        }
    }
}
#endif
