//------------------------------------------------------------------------------
//  Command-line interface for ACD2D
//------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>

#include "acd2d.h"
#include "acd2d_concavity.h"
#include "acd2d_util.h"

using namespace std;
using namespace acd2d;

// Global variables - đổi tên để tránh xung đột
double g_tau = 0.0;
double g_alpha = 0.0;
double g_beta = 1.0;
string g_measure_type = "hybrid1";
string g_input_file;
string g_output_file;
bool g_verbose = false;
double g_scale_factor = 1.0;

void print_usage(const char* prog_name) {
    cout << "ACD2D - Approximate Convex Decomposition 2D (Command-line)\n\n";
    cout << "Usage: " << prog_name << " [options]\n\n";
    cout << "Options:\n";
    cout << "  -i <file>     Input .poly file (required)\n";
    cout << "  -o <file>     Output .poly file (default: output.poly)\n";
    cout << "  -t <value>    Tau - concavity tolerance [0.0-1.0] (default: 0.0)\n";
    cout << "  -a <value>    Alpha - weight for concavity in cut direction (default: 0.0)\n";
    cout << "  -b <value>    Beta - weight for distance in cut direction (default: 1.0)\n";
    cout << "  -m <type>     Measure type: hybrid1, hybrid2, sp, sl (default: hybrid1)\n";
    cout << "  -v            Verbose output\n";
    cout << "  -h            Show this help\n\n";
    cout << "Example:\n";
    cout << "  " << prog_name << " -i input.poly -o output.poly -t 0.05 -m hybrid1\n";
}

bool parse_arguments(int argc, char** argv) {
    if (argc < 2) {
        return false;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            g_input_file = argv[++i];
        } 
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            g_output_file = argv[++i];
        } 
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            g_tau = atof(argv[++i]);
            if (g_tau < 0.0 || g_tau > 1.0) {
                cerr << "Error: Tau must be in range [0.0, 1.0]\n";
                return false;
            }
        } 
        else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            g_alpha = atof(argv[++i]);
        } 
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            g_beta = atof(argv[++i]);
        } 
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_measure_type = argv[++i];
            if (g_measure_type != "hybrid1" && g_measure_type != "hybrid2" && 
                g_measure_type != "sp" && g_measure_type != "sl") {
                cerr << "Error: Invalid measure type. Use: hybrid1, hybrid2, sp, or sl\n";
                return false;
            }
        } 
        else if (strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        } 
        else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } 
        else {
            cerr << "Error: Unknown option " << argv[i] << "\n";
            return false;
        }
    }

    if (g_input_file.empty()) {
        cerr << "Error: Input file required (-i option)\n";
        return false;
    }

    if (g_output_file.empty()) {
        g_output_file = "output.poly";
    }

    return true;
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (!parse_arguments(argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }

    if (g_verbose) {
        cout << "=== ACD2D Configuration ===\n";
        cout << "Input file:  " << g_input_file << "\n";
        cout << "Output file: " << g_output_file << "\n";
        cout << "Tau:         " << g_tau << "\n";
        cout << "Alpha:       " << g_alpha << "\n";
        cout << "Beta:        " << g_beta << "\n";
        cout << "Measure:     " << g_measure_type << "\n";
        cout << "===========================\n\n";
    }

    // Load polygon
    cd_polygon poly;
    ifstream fin(g_input_file.c_str());
    
    if (!fin.good()) {
        cerr << "Error: Cannot open input file: " << g_input_file << endl;
        return 1;
    }

    fin >> poly;
    fin.close();

    if (!poly.valid()) {
        cerr << "Error: Invalid polygon in input file\n";
        return 1;
    }

    g_scale_factor = poly.front().getRadius();
    poly.normalize();

    if (g_verbose) {
        cout << "Loaded polygon with " << poly.size() << " chain(s)\n";
    }

    // Create decomposer (save diagonals = true)
    cd_2d cd(true);
    cd.addPolygon(poly);
    cd.updateCutDirParameters(g_alpha, g_beta);

    // Create concavity measure
    IConcavityMeasure* measure = ConcavityMeasureFac::createMeasure(g_measure_type);
    
    if (g_measure_type == "hybrid2") {
        ((HybridMeasurement2*)measure)->setTau(g_tau);
    }

    if (g_verbose) {
        cout << "Starting decomposition...\n";
    }

    // Perform decomposition
    cd.decomposeAll(g_tau, measure);
    delete measure;

    // Get results
    const list<cd_polygon>& result_polys = cd.getDoneList();
    const list<cd_diagonal>& diagonals = cd.getDiagonal();

    // Print statistics
    cout << "\n=== Decomposition Results ===\n";
    cout << "Number of output polygons: " << result_polys.size() << "\n";
    cout << "Number of cuts: " << diagonals.size() << "\n";

    if (g_verbose && !diagonals.empty()) {
        cout << "\nCut lines:\n";
        int cut_id = 1;
        for (const auto& diag : diagonals) {
            // Scale back diagonals
            cout << "  Cut " << cut_id++ << ": "
                 << "(" << diag.v[0][0] * g_scale_factor << ", " << diag.v[0][1] * g_scale_factor << ") -> "
                 << "(" << diag.v[1][0] * g_scale_factor << ", " << diag.v[1][1] * g_scale_factor << ")\n";
        }
    }

    // Denormalize polygons before saving
    cd_2d cd_denormalized(true);
    for (auto poly_iter = result_polys.begin(); poly_iter != result_polys.end(); ++poly_iter) {
        cd_polygon denorm_poly = *poly_iter;
        denorm_poly.scale(g_scale_factor);  // Scale back to original size
        cd_denormalized.addPolygon(denorm_poly);
    }

    // Save output
    save_polys(g_output_file, cd_denormalized);
    cout << "\nOutput saved to: " << g_output_file << "\n";

    return 0;
}