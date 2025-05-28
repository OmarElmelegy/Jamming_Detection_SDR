#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <file_path>" << std::endl;
}

std::vector<double> read_rssi_values(const std::string& filepath) {
    std::vector<double> values;
    std::ifstream infile(filepath);
    if (!infile) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return values;
    }

    double val;
    while (infile >> val) {
        values.push_back(val);
    }

    return values;
}

struct Stats {
    size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double mode = NAN;
    double variance = 0.0;
    double stdev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double range = 0.0;
};

Stats compute_statistics(std::vector<double>& values) {
    Stats s;
    s.count = values.size();
    if (s.count == 0) return s;

    // Sort for median, min, max, mode
    std::sort(values.begin(), values.end());

    s.min = values.front();
    s.max = values.back();
    s.range = s.max - s.min;

    // Mean
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    s.mean = sum / static_cast<double>(s.count);

    // Median
    if (s.count % 2 == 0) {
        s.median = (values[s.count/2 - 1] + values[s.count/2]) / 2.0;
    } else {
        s.median = values[s.count/2];
    }

    // Mode (find most frequent)
    std::unordered_map<double, size_t> freq;
    for (double x : values) freq[x]++;
    size_t max_count = 0;
    bool unique_mode = true;
    for (const auto& [value, count] : freq) {
        if (count > max_count) {
            max_count = count;
            s.mode = value;
            unique_mode = true;
        } else if (count == max_count) {
            unique_mode = false;
        }
    }
    if (!unique_mode) s.mode = NAN;

    // Variance & Std Dev (population sample)
    double sq_sum = 0.0;
    for (double x : values) {
        sq_sum += (x - s.mean) * (x - s.mean);
    }
    if (s.count > 1) {
        s.variance = sq_sum / static_cast<double>(s.count - 1);
        s.stdev = std::sqrt(s.variance);
    }

    return s;
}

void print_statistics(const Stats& s) {
    if (s.count == 0) {
        std::cout << "No valid RSSI readings found." << std::endl;
        return;
    }

    std::cout << "Count   : " << s.count << std::endl;
    std::cout << "Mean    : " << s.mean << std::endl;
    std::cout << "Median  : " << s.median << std::endl;
    if (std::isnan(s.mode)) {
        std::cout << "Mode    : No unique mode" << std::endl;
    } else {
        std::cout << "Mode    : " << s.mode << std::endl;
    }
    std::cout << "Variance: " << s.variance << std::endl;
    std::cout << "Std Dev : " << s.stdev << std::endl;
    std::cout << "Min     : " << s.min << std::endl;
    std::cout << "Max     : " << s.max << std::endl;
    std::cout << "Range   : " << s.range << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filepath = argv[1];
    auto values = read_rssi_values(filepath);
    auto stats = compute_statistics(values);
    print_statistics(stats);

    return 0;
}
