#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <experimental/filesystem>
#include <cmath>

namespace fs = std::experimental::filesystem;

struct Options {
    std::string input_file;
    std::string output_file;
    size_t num_frames = 0;
    bool limit_frames = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --input <path> [--output <path>] [--num-frames <n>]" << std::endl;
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input" || arg == "-i") && i+1 < argc) {
            opts.input_file = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i+1 < argc) {
            opts.output_file = argv[++i];
        } else if ((arg == "--num-frames" || arg == "-n") && i+1 < argc) {
            opts.num_frames = std::stoul(argv[++i]);
            opts.limit_frames = true;
        } else {
            print_usage(argv[0]);
            exit(1);
        }
    }
    if (opts.input_file.empty()) {
        print_usage(argv[0]);
    // Auto-generate output name if not provided
    if (opts.output_file.empty()) {
        auto base = fs::path(opts.input_file).stem().string();
        auto dir = fs::path(opts.input_file).parent_path();
        opts.output_file = (dir / ("fft_means_4bins_" + base + ".txt")).string();
    }
    }
    return opts;
}

// Missing the bins_per_frame constant
const size_t bins_per_frame = 1024;

int main(int argc, char* argv[]) {
    Options opts = parse_args(argc, argv);
    // Determine file size and frame count
    std::uintmax_t file_size = fs::file_size(opts.input_file);
    const size_t frame_bytes = bins_per_frame * sizeof(float);
    size_t total_frames = file_size / frame_bytes;
    if (opts.limit_frames && opts.num_frames < total_frames) {
        total_frames = opts.num_frames;
        std::cout << "Processing only " << total_frames << " frames as requested" << std::endl;
    }
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Frame size: " << frame_bytes << " bytes" << std::endl;
    std::cout << "Frames to process: " << total_frames << std::endl;

    std::ifstream in(opts.input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }
    std::ofstream out(opts.output_file);
    if (!out) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    std::vector<float> buffer(bins_per_frame);
    size_t chunks_per_frame = (bins_per_frame + 39) / 40;

    for (size_t frame = 0; frame < total_frames; ++frame) {
        in.read(reinterpret_cast<char*>(buffer.data()), frame_bytes);
        if (!in) break;

        for (size_t chunk = 0; chunk < chunks_per_frame; ++chunk) {
            size_t start = chunk * 40;
            size_t end = std::min(start + 40, bins_per_frame);
            double sum = 0.0;
            for (size_t i = start; i < end; ++i) {
                sum += buffer[i];
            }
            double mean = sum / static_cast<double>(end - start);
            out << mean << '\n';
        }

        if (frame % 100 == 0) {
            std::cout << "Processed " << frame << "/" << total_frames << " frames..." << std::endl;
        }
    }

    size_t lines = total_frames * chunks_per_frame;
    std::cout << "Conversion complete. Output saved to " << opts.output_file << std::endl;
    std::cout << "Total lines in output file: " << lines << std::endl;

    return 0;
}