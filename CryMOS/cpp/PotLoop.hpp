#define BOOST_NO_RTTI
#include <boost/math/tools/roots.hpp>

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <sstream>

constexpr long double Q_E = 1.6021766208e-19;

struct PotLoopBase {
    Float
        phi_t, g_A, psi_a, psi_b, n_i,
        N_A, eps_si, phi_ms, Q_0, g_t, cox;
    std::vector<Float> psi_t{}, N_t{};

    unsigned tol_bits{31}; //result will be exact to 2^(1-tol_bits)
    unsigned long max_iter{2000};

    inline Float exp_phi_t(Float x) const {
        return std::exp(x / phi_t);
    }

    inline Float fs_ea(Float psi_s, Float v_ch) const {
        return 1./(1. + g_A * exp_phi_t(psi_a - psi_s + v_ch));
    }

    inline Float fb_ea() const {
        return 1./(1. + g_A * exp_phi_t(psi_a - psi_b));
    }

    inline Float fs_Et(Float psi_t, Float psi_s, Float v_ch) const {
        return 1. / (1. + g_t * exp_phi_t(+psi_t - psi_s + v_ch));
    }

    Float Q_it(Float psi_s, Float v_ch) const {
        Float ret = 0;

        for(size_t i=0; i < psi_t.size(); ++i) {
            if(g_t == 0)
                ret += (-Q_E) * N_t[i];
            else
                ret += (-Q_E) * N_t[i] * fs_Et(psi_t[i], psi_s, v_ch);
        }
        return ret;
    }

    Float v_fb(Float psi_s, Float v_ch) const {
        return phi_ms + (Q_0 - Q_it(psi_s, v_ch)) / cox;
    }
};


struct PotLoop : public PotLoopBase {
    inline Float Es_x(Float psi_s, Float v_ch, Float fb_ea_) const {
        auto fs_ea_ = fs_ea(psi_s, v_ch);

        auto fac1 = 2. * Q_E / eps_si;
        auto fac2 = exp_phi_t(psi_s - v_ch) + exp_phi_t(-psi_s) - exp_phi_t(psi_b - v_ch) - exp_phi_t(-psi_b);
        auto fac3 = psi_s - psi_b - phi_t * std::log(fs_ea_ / fb_ea_);
        auto es2 = n_i * phi_t * fac1 * fac2 + fac1 * N_A * fac3;
        if(psi_s >= psi_b)
            return std::sqrt(es2);
        else
            return -std::sqrt(es2);
    }

    Float Es(Float psi_s, Float v_ch) const {
        auto fb_ea_ = fb_ea();
        return Es_x(psi_s, v_ch, fb_ea_);
    }

    std::pair<Float, Float> psi_s_x(Float v_ch, Float v_gb, boost::uintmax_t& iter, Float start=1.0) const {
        using boost::math::tools::bracket_and_solve_root;
        using boost::math::tools::eps_tolerance;

        const Float fb_ea_ = fb_ea();  //precalculate

        auto rootfun = [=](Float psi_s) {
            psi_s -= 1.;
            return v_fb(psi_s, v_ch) + eps_si * Es_x(psi_s, v_ch, fb_ea_) / cox + psi_s - psi_b - v_gb;
        };

        //TOMS748 instead?
        return bracket_and_solve_root(rootfun, start, 1.2, true, eps_tolerance<Float>(tol_bits), iter);
    }
};

// fermi-dirac solution
struct PotLoopFD : public PotLoopBase {
    // TODO: are these really necessary?
    Float E_i, E_v, E_c, N_c, N_v;

    inline Float fdint(Float e) const {
        // return fdk(k=0.5, phi=E / (k * T))
        static fdint_method<> fd1h("fd1h");
        return fd1h(e / (Q_E * phi_t));
    }

    inline Float Es(Float psi_s, Float v_ch) const {
        using boost::math::quadrature::gauss_kronrod;
        gauss_kronrod<double, 15> integrator;

        auto fac = 2. * Q_E / eps_si;

        const auto S2PI = 2. * boost::math::constants::one_div_root_pi<Float>();

        auto int_fun = [&](Float psi) {
            auto n_fd = N_c * S2PI * fdint(Q_E * (psi - v_ch) + E_i - E_c);
            auto p_fd = N_v * S2PI * fdint(E_v - Q_E * psi - E_i);
            auto na_min = N_A / (1. + g_A * exp_phi_t(psi_a - psi + v_ch));
            return n_fd - p_fd + na_min;
        };

        Float error, ret;
        try {
            if(psi_s >= psi_b)
                ret = sqrt(fac * integrator.integrate(int_fun, psi_b, psi_s, 5, 1e-18, &error));
            else
                ret = -sqrt(-fac * integrator.integrate(int_fun, psi_s, psi_b, 5, 1e-18, &error));
        } catch(std::domain_error const& ex) {
            std::stringstream msg;
            msg << ex.what() << " psi_s = " << psi_s;
            throw std::domain_error(msg.str());
        }

        return ret;
    }

    std::pair<Float, Float> psi_s_x(Float v_ch, Float v_gb, boost::uintmax_t& iter, Float start=1.0) const {
        using boost::math::tools::bracket_and_solve_root;
        using boost::math::tools::eps_tolerance;

        auto rootfun = [=](Float psi_s) {
            psi_s -= 1.;
            return v_fb(psi_s, v_ch) + eps_si * Es(psi_s, v_ch) / cox + psi_s - psi_b - v_gb;
        };

        //TOMS748 instead?
        return bracket_and_solve_root(rootfun, start, 1.2, true, eps_tolerance<Float>(tol_bits), iter);
    }
};

struct PotLoopGildenblat : public PotLoopBase {
    Float lam_bulk, bulk_n, bulk_p;

    inline Float Es(Float psi_s, Float v_ch) const {
        auto k_0 = std::exp(-v_ch / phi_t);
        auto phi_s = psi_s - psi_b;
        auto u = phi_s / phi_t;
        auto g_fun = 1. / lam_bulk * std::log(1. + lam_bulk * (std::exp(u) - 1));
        auto h2 = std::exp(-u) - 1 + g_fun + bulk_n / bulk_p * k_0 * (std::exp(u) - 1. - g_fun);

        auto es2 = 2 * Q_E * bulk_p * phi_t / eps_si * h2;

        if(psi_s >= psi_b)
            return std::sqrt(es2);
        else
            return -std::sqrt(es2);
    }

    std::pair<Float, Float> psi_s_x(Float v_ch, Float v_gb, boost::uintmax_t& iter, Float start=1.0) const {
        using boost::math::tools::bracket_and_solve_root;
        using boost::math::tools::eps_tolerance;

        const Float fb_ea_ = fb_ea();  //precalculate

        auto rootfun = [=](Float psi_s) {
            psi_s -= 1.;
            return v_fb(psi_s, v_ch) + eps_si * Es(psi_s, v_ch) / cox + psi_s - psi_b - v_gb;
        };

        //TOMS748 instead?
        return bracket_and_solve_root(rootfun, start, 1.2, true, eps_tolerance<Float>(tol_bits), iter);
    }
};

template<class L>
Float psi_s(L& l, Float v_ch, Float v_gb) {
    static Float last_root = 1.0;

    boost::uintmax_t iter = l.max_iter;
    auto root = l.psi_s_x(v_ch, v_gb, iter, last_root);

    if(iter == l.max_iter)
        throw std::runtime_error("no solution found in max_iter iterations");

    if(root.first + root.second < 1e-6)
        throw std::runtime_error("ran into psi_s < -1.0");

    if(!isfinite(root.first) || !isfinite(root.second))
        throw std::runtime_error("no solution found (nan or inf)");

    last_root = (root.first + root.second) / 2;
    return last_root - 1.;
}

template<class L>
py::tuple psi_s2(const L& l, Float v_ch, Float v_gb) {
    boost::uintmax_t iter = l.max_iter;
    auto root = l.psi_s_x(v_ch, v_gb, iter);
    return py::make_tuple(root.first, root.second, iter);
}
