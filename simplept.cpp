
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#include <algorithm>
#define PI 3.1415926535897932384626433832795
#define RRDEPTH 5
#define SURVIVALPROB 0.9
#define LIGHTIND 7

/*
 * Thread-safe random number generator
 */

struct RNG {
    RNG() : distrb(0.0, 1.0), engines() {}

    void init(int nworkers) {
        std::random_device rd;
        engines.resize(nworkers);
        for (int i = 0; i < nworkers; ++i)
            engines[i].seed(rd());
    }

    double operator()() {
        int id = omp_get_thread_num();
        return distrb(engines[id]);
    }

    std::uniform_real_distribution<double> distrb;
    std::vector<std::mt19937> engines;
} rng;


/*
 * Basic data types
 */

struct Vec {
    double x, y, z;

    Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }

    Vec operator+ (const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    Vec operator- (const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    Vec operator* (double b) const { return Vec(x * b, y * b, z * b); }

    Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
    Vec& normalize() { return *this = *this * (1.0 / std::sqrt(x * x + y * y + z * z)); }
    double dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; }
    Vec cross(const Vec& b) const { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Ray {
    Vec o, d;
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

struct BRDF {
    virtual Vec eval(const Vec& n, const Vec& o, const Vec& i) const = 0;
    virtual void sample(const Vec& n, const Vec& o, Vec& i, double& pdf) const = 0;
};


/*
 * Utility functions
 */

inline double clamp(double x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline int toInt(double x) {
    return static_cast<int>(std::pow(clamp(x), 1.0 / 2.2) * 255 + .5);
}



/*
 * Shapes
 */

struct Sphere {
    Vec p, e;           // position, emitted radiance
    double rad;         // radius
    const BRDF& brdf;   // BRDF

    Sphere(double rad_, Vec p_, Vec e_, const BRDF& brdf_) :
        rad(rad_), p(p_), e(e_), brdf(brdf_) {}

    double intersect(const Ray& r) const { // returns distance, 0 if nohit
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0) return 0; else det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};


/*
 * Sampling functions
 */

inline void createLocalCoord(const Vec& n, Vec& u, Vec& v, Vec& w) {
    w = n;
    u = ((std::abs(w.x) > .1 ? Vec(0, 1) : Vec(1)).cross(w)).normalize();
    v = w.cross(u);
}


/*
 * BRDFs
 */

 // Ideal diffuse BRDF
struct DiffuseBRDF : public BRDF {
    DiffuseBRDF(Vec kd_) : kd(kd_) {}

    Vec eval(const Vec& n, const Vec& o, const Vec& i) const {
        return kd * (1.0 / PI);
    }

    void sample(const Vec& n, const Vec& o, Vec& i, double& pdf) const {
        double z = std::sqrt(rng());
        double r = std::sqrt(1.0 - z * z);
        double phi = 2.0 * PI * rng();
        double x = r * std::cos(phi);
        double y = r * std::sin(phi);

        Vec u;
        Vec v;
        Vec w;
        createLocalCoord(n, u, v, w);

        i = (u * x + v * y + w * z).normalize();
        pdf = i.dot(n) * (1.0 / PI);
    }

    Vec kd;
};

// Ideal specular BRDF
struct SpecularBRDF : public BRDF {
    SpecularBRDF(Vec ks_) : ks(ks_) {}

    Vec mirroredDirection(const Vec& n, const Vec& o) const
    {
        return n * 2.0 * n.dot(o) - o;
    }

    Vec eval(const Vec& n, const Vec& o, const Vec& i) const
    {
        Vec md = mirroredDirection(n, o);
        md = md.normalize();
        if (i.x == md.x && i.y == md.y && i.z == md.z)
        {
            return ks * (1.0 / (n.dot(i)));
        }
        else
        {
            return Vec();
        }
    }

    void sample(const Vec& n, const Vec& o, Vec& i, double& pdf) const
    {
        i = mirroredDirection(n, o).normalize();
        pdf = 1.0;
    }

    Vec ks;
};

// Phong BRDF
struct PhongBRDF : public BRDF {
    PhongBRDF(Vec kd_, Vec ks_, int nExp_) : kd(kd_), ks(ks_), nExp(nExp_) {}

    Vec mirroredDirection(const Vec& n, const Vec& o) const
    {
        return n * 2.0 * n.dot(o) - o;
    }

    void diffSample(const Vec& n, const Vec& o, Vec& i, double& pdf) const {
        double z = std::sqrt(rng());
        double r = std::sqrt(1.0 - z * z);
        double phi = 2.0 * PI * rng();
        double x = r * std::cos(phi);
        double y = r * std::sin(phi);

        Vec u;
        Vec v;
        Vec w;
        createLocalCoord(n, u, v, w);

        i = (u * x + v * y + w * z).normalize();
        pdf = i.dot(n) * (1.0 / PI);
    }

    void specSample(const Vec& n, const Vec& o, Vec& i, double& pdf) const
    {
        i = mirroredDirection(n, o).normalize();
        pdf = 1.0;
    }

    Vec eval(const Vec& n, const Vec& o, const Vec& i) const {
        Vec mirror = mirroredDirection(n, o).normalize();
        double theta1 = std::acos(i.dot(mirror));

        return (kd * (1 / PI)) + (ks * ((nExp + 2) / (2 * PI)) * std::pow(std::max(0.0, std::cos(theta1)), nExp));
    }

    void sample(const Vec& n, const Vec& o, Vec& i, double& pdf) const {
        if (rng() < kd.x)
        {
            diffSample(n, o, i, pdf);
            return;
        }
        double theta = std::acos(std::pow(1 - rng(), 1 / (nExp + 2)));
        double phi = 2.0 * PI * rng();


        i = Vec(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta)).normalize();
        double c = (2.0 * PI) / (nExp + 1);
        pdf = (1/c) * std::pow(std::cos(theta), nExp + 1);
    }

    Vec kd;
    Vec ks;
    int nExp;
};


/*
 * Scene configuration
 */

 // Pre-defined BRDFs
const DiffuseBRDF leftWall(Vec(.75, .25, .25)),
rightWall(Vec(.25, .25, .75)),
otherWall(Vec(.75, .75, .75)),
blackSurf(Vec(0.0, 0.0, 0.0)),
brightSurf(Vec(0.9, 0.9, 0.9));

const SpecularBRDF spec(Vec(0.999, 0.999, 0.999));

const PhongBRDF phong(Vec(0.8, 0.8, 0.8), Vec(0.2, 0.2, 0.2), 100);

// Scene: list of spheres
const Sphere spheres[] = {
    Sphere(1e5,  Vec(1e5 + 1,40.8,81.6),   Vec(),         leftWall),   // Left
    Sphere(1e5,  Vec(-1e5 + 99,40.8,81.6), Vec(),         rightWall),  // Right
    Sphere(1e5,  Vec(50,40.8, 1e5),      Vec(),         otherWall),  // Back
    Sphere(1e5,  Vec(50, 1e5, 81.6),     Vec(),         otherWall),  // Bottom
    Sphere(1e5,  Vec(50,-1e5 + 81.6,81.6), Vec(),         otherWall),  // Top
    Sphere(16.5, Vec(27,16.5,47),        Vec(),         phong), // Ball 1
    Sphere(16.5, Vec(73,16.5,78),        Vec(),         spec), // Ball 2
    //Sphere(15.0, Vec(45.0,15.0,105.0),        Vec(),         phong), // Ball 3
    Sphere(5.0,  Vec(50,70.0,81.6),      Vec(50,50,50), blackSurf)   // Light
};

// Camera position & direction
const Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).normalize());


/*
 * Global functions
 */

bool intersect(const Ray& r, double& t, int& id) {
    double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
    return t < inf;
}


/*
 * KEY FUNCTION: radiance estimator
 */

Vec radiance(const Vec& x, const Vec& omega, const Vec& nx, int idx, int depth);
Vec reflectedRadiance(const Vec& x, const Vec& omega, const Vec& nx, int idx, int depth);
void luminaireSample(const Vec& p, const double& r, Vec& y, double& pdf, Vec& ny);
int visible(const Vec& x, const Vec& y, int idx);

Vec receivedRadiance(const Ray& r, int depth, bool flag) {
    double t;                                   // Distance to intersection
    int id = 0;                                 // id of intersected sphere

    if (!intersect(r, t, id)) return Vec();   // if miss, return black
    const Sphere& obj = spheres[id];            // the hit object

    Vec x = r.o + r.d * t;                        // The intersection point
    Vec o = (Vec() - r.d).normalize();          // The outgoing direction (= -r.d)

    Vec n = (x - obj.p).normalize();            // The normal direction
    if (n.dot(o) < 0) n = n * -1.0;

    /*
    Tips

    1. Other useful quantities/variables:
    Vec Le = obj.e;                             // Emitted radiance
    const BRDF &brdf = obj.brdf;                // Surface BRDF at x

    2. Call brdf.sample() to sample an incoming direction and continue the recursion
    */

    return radiance(x, o, n, id, depth);
}

Vec radiance(const Vec& x, const Vec& omega, const Vec& nx, int idx, int depth)
{
    return spheres[idx].e + reflectedRadiance(x, omega, nx, idx, depth);
}

Vec reflectedRadiance(const Vec& x, const Vec& omega, const Vec& nx, int idx, int depth)
{
    Sphere s = spheres[idx];
    const BRDF& brdf = s.brdf;

    Vec reflRad = Vec();

    Vec y1;
    double pdf1;
    Vec ny1;
    luminaireSample(spheres[LIGHTIND].p, spheres[LIGHTIND].rad, y1, pdf1, ny1);
    Vec i1 = (y1 - x).normalize();
    Vec i2;
    double pdf2;
    if (visible(x, y1, idx) == 1)
    {
        pdf1 *= (y1 - x).dot(y1 - x) / ny1.dot(i1 * -1);
        brdf.sample(ny1, i1, i2, pdf2);

        // Uncomment Line below for Balance Heuristic (light sample part)
        reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i1)) * nx.dot(i1) * (1 / (pdf1 + pdf2));

        // Uncomment Line below for Power Heuristic (light sample part)
        //reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i1)) * nx.dot(i1) * pdf1 * (1 / (std::pow(pdf1, 2) + std::pow(pdf2, 2)));

        // Uncomment Line below for Constant Heuristic (light sample part)
        //reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i1)) * nx.dot(i1) * (1 / pdf1) * 0.5;
    }
    
    brdf.sample(nx, omega, i2, pdf2);
    Ray rx(x, i2);
    double ty2;
    int idy2 = 0;
    intersect(rx, ty2, idy2);
    const Sphere& obj = spheres[idy2];
    Vec y2 = rx.o + rx.d * ty2;
    Vec o = (Vec() - rx.d).normalize();
    Vec ny2 = (y2 - obj.p).normalize();
    if (ny2.dot(o) < 0) ny2 = ny2 * -1.0;
    if (idy2 == LIGHTIND)
    {
        pdf1 = (1 / (4.0 * PI * spheres[LIGHTIND].rad * spheres[LIGHTIND].rad)) * (y2 - x).dot(y2 - x) / ny2.dot(i2 * -1);

        // Uncomment Line below for Balance Heuristic (brdf sample part)
        reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i2)) * nx.dot(i2) * (1 / (pdf1 + pdf2));

        // Uncomment Line below for Power Heuristic (brdf sample part)
        //reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i2)) * nx.dot(i2) * pdf2 * (1 / (std::pow(pdf1, 2) + std::pow(pdf2, 2)));

        // Uncomment Line below for Constant Heuristic (brdf sample part)
        //reflRad = reflRad + spheres[LIGHTIND].e.mult(brdf.eval(nx, omega, i2)) * nx.dot(i2) * (1 / pdf2) * 0.5;
    }

    double p;
    if (depth <= RRDEPTH)
    {
        p = 1.0;
    }
    else
    {
        p = SURVIVALPROB;
    }

    if (rng() < p)
    {
        reflRad = reflRad + reflectedRadiance(y2, o, ny2, idy2, depth + 1).mult(brdf.eval(nx, omega, i2)) * nx.dot(i2) * (1.0 / (pdf2 * p));
    }

    return reflRad;
}

void luminaireSample(const Vec& p, const double& r, Vec& y, double& pdf, Vec& ny)
{
    double xi1 = rng();
    double xi2 = rng();

    double z0 = 2.0 * xi1 - 1.0;
    double x0 = std::sqrt(1.0 - z0 * z0) * std::cos(2.0 * PI * xi2);
    double y0 = std::sqrt(1.0 - z0 * z0) * std::sin(2.0 * PI * xi2);

    y = p + (Vec(x0, y0, z0) * r);
    pdf = 1 / (4.0 * PI * r * r);
    ny = Vec(x0, y0, z0);
}

int visible(const Vec& x, const Vec& y, int idx)
{
    Vec i = (x - y).normalize();
    Ray r(y, i);

    double t;
    int id = 0;
    intersect(r, t, id);

    int vis1 = 0;
    if (id == idx)
    {
        vis1 = 1;
    }

    Vec i2 = (y - x).normalize();
    Ray r2(x, i2);

    double t2;
    int id2 = 0;
    intersect(r2, t2, id2);

    int vis2 = 0;
    if (id2 == LIGHTIND)
    {
        vis2 = 1;
    }

    if (vis1 == 1 && vis2 == 1)
    {
        return 1;
    }
    return 0;
}


/*
 * Main function (do not modify)
 */

int main(int argc, char* argv[]) {
    int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);
    rng.init(nworkers);

    int w = 480, h = 360, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    Vec cx = Vec(w * .5135 / h), cy = (cx.cross(cam.d)).normalize() * .5135;
    std::vector<Vec> c(w * h);

    int tot = 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int i = (h - y - 1) * w + x;

            for (int sy = 0; sy < 2; ++sy) {
                for (int sx = 0; sx < 2; ++sx) {
                    Vec r;
                    for (int s = 0; s < samps; s++) {
                        double r1 = 2 * rng(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * rng(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                            cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        r = r + receivedRadiance(Ray(cam.o, d.normalize()), 1, true) * (1. / samps);
                    }
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
                }
            }
        }
#pragma omp critical
        fprintf(stderr, "\rRendering (%d spp) %6.2f%%", samps * 4, 100. * (++tot) / h);
    }
    fprintf(stderr, "\n");

    // Write resulting image to a PPM file
    FILE* f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    fclose(f);

    return 0;
}
