#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <map>
#include <set>

class Bytes: public std::vector<uint8_t> {
public:
    bool operator<(const Bytes &other) const {
        for (size_t i = 0; i < std::min(this->size(), other.size()); i++) {
            if ((*this)[i] != other[i]) {
                return (*this)[i] < other[i];
            }
        }
        return this->size() < other.size();
    }

    bool operator==(const Bytes &other) const {
        if (this->size() != other.size()) {
            return false;
        }
        for (size_t i = 0; i < this->size(); i++) {
            if ((*this)[i] != other[i]) {
                return false;
            }
        }
        return true;
    }

};
std::ostream& operator<<(std::ostream &os, const Bytes &bytes) {
    for (auto b : bytes) {
        os << b;
    }
    return os;
}

extern "C"
int bpe_trainer(const char* input_file, const char * merges_file, int vocab_size) {
    std::fstream fin(input_file);
    int count = 0;
    fin >> count;
    // std::cout << count << "\n";
    std::vector<std::vector<Bytes>> token_bytes;
    std::vector<int> token_counts;
    for (int i = 0; i < count; i++) {
        std::vector<Bytes> bytes;
        int size = 0;
        fin >> size;
        for (int j = 0; j < size; j++) {
            int byte;
            fin >> byte;
            Bytes temp;
            temp.push_back(byte);
            // std::cout << temp << " ";
            bytes.push_back(temp);
        }
        // std::cout << "\n";
        token_bytes.push_back(bytes);
        int token_count;
        fin >> token_count;
        token_counts.push_back(token_count);
        // std::cout << size << " " << token_count << "\n";
    }
    fin.close();
    auto cpus = std::thread::hardware_concurrency() / 3 * 2;
    if (count < 10000) {
        cpus = 16;
    }
    // std::cout << "cpu " << cpus << "\n";
    auto per_partition = count / cpus;
    std::vector<std::pair<int, int>> partitions;
    std::mutex mu;
    for (unsigned int i = 0; i < cpus; i++) {
        int start = i * per_partition;
        int end = (i == cpus - 1) ? count : (i + 1) * per_partition;
        partitions.push_back({start, end});
    }
    // 1. count total pair count
    std::set<std::pair<int64_t, std::pair<Bytes, Bytes>>> to_merge;
    std::map<std::pair<Bytes, Bytes>, int64_t> pair_count;
    auto count_pair_partition = [&](int partition) {
        std::map<std::pair<Bytes, Bytes>, int64_t> local_pair_count;
        for (int i = partitions[partition].first; i < partitions[partition].second; i++) {
            auto &tokens = token_bytes[i];
            for (size_t j = 0; j + 1 < tokens.size(); j++) {
                local_pair_count[{tokens[j], tokens[j + 1]}] += token_counts[i];
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        for (auto &p : local_pair_count) {
            // to_merge.insert({p.second, p.first});
            pair_count[p.first] += p.second;
        }
    };
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < cpus; i++) {
        threads.emplace_back(count_pair_partition, i);
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();
    // std::cout << "count pair done\n";
    for (auto &p : pair_count) {
        // std::cout << "count " << p.first.first << " " << p.first.second << " " << p.second << "\n";
        to_merge.insert({p.second, p.first});
    }
    std::fstream fout(merges_file, std::ios::out | std::ios::binary);
    while (vocab_size--) {
        std::cout << "vocab size left " << vocab_size << "\n";
        if (to_merge.empty()) {
            break;
        }
        auto top = *to_merge.rbegin();
        to_merge.erase(top);
        Bytes first = top.second.first;
        Bytes second = top.second.second;
        // std::cout << "merge " << first << " " << second << " " << top.first << "\n";
        std::fflush(stdout);
        int32_t size;
        size = first.size();
        fout.write(reinterpret_cast<char*>(&size), sizeof(size));
        fout.write(reinterpret_cast<char*>(first.data()), first.size());
        size = second.size();
        fout.write(reinterpret_cast<char*>(&size), sizeof(size));
        fout.write(reinterpret_cast<char*>(second.data()), second.size());
        fout.flush();
        Bytes merged = first;
        merged.insert(merged.end(), second.begin(), second.end());
        auto count = top.first;
        std::map<std::pair<Bytes, Bytes>, int> pair_count_diff;
        auto update = [&](int partition) {
            std::map<std::pair<Bytes, Bytes>, int> local_diff;
            for (int i = partitions[partition].first; i < partitions[partition].second; i++) {
                auto &tokens = token_bytes[i];
                std::vector<Bytes> new_tokens;
                int64_t j = 0;
                while (j < tokens.size()) {
                    if (j + 1 < tokens.size() && tokens[j] == first && tokens[j + 1] == second) {
                        new_tokens.push_back(merged);
                        if (j - 1 >= 0) {
                            local_diff[{tokens[j - 1], tokens[j]}] -= token_counts[i];
                            local_diff[{tokens[j - 1], merged}] += token_counts[i];
                        }
                        if (j + 2 < tokens.size()) {
                            local_diff[{tokens[j + 1], tokens[j + 2]}] -= token_counts[i];
                            local_diff[{merged, tokens[j + 2]}] += token_counts[i];
                        }
                        local_diff[{tokens[j], tokens[j + 1]}] -= token_counts[i];
                        j += 2;
                    } else {
                        new_tokens.push_back(tokens[j]);
                        j += 1;
                    }
                }
                tokens = new_tokens;
            }
            std::lock_guard<std::mutex> lock(mu);
            for (auto &p : local_diff) {
                pair_count_diff[p.first] += p.second;
            }
        };
        for (unsigned int i = 0; i < cpus; i++) {
            threads.emplace_back(update, i);
        }
        for (auto &t : threads) {
            t.join();
        }
        threads.clear();
        for (auto &p : pair_count_diff) {
            auto key = p.first;
            auto diff = p.second;
            auto it = to_merge.find({pair_count[key], key});
            if (it != to_merge.end()) {
                to_merge.erase(it);
            }
            pair_count[key] += diff;
            if (pair_count[key] > 0) {
                to_merge.insert({pair_count[key], key});
            }
        }
    }
    fout.close();
    return 0;
}

int main() {
    std::string root = "/home/tyhcyq/cyq/assignment1-basics/tmp/";
    std::string input = root + "pretokened.txt";
    std::string merges = root + "merges.txt";
    bpe_trainer(input.c_str(), merges.c_str(), 1000);
    return 0;
}