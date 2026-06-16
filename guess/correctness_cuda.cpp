#include "md5.h"
#include "md5_cuda.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    const std::vector<std::string> passwords = {
        "123456",
        "password",
        "12345678",
        "qwerty",
        "123456789",
        "12345",
        "1234",
        "111111"
    };

    std::vector<std::array<bit32, 4>> cuda_results;

    try
    {
        MD5HashBatchCUDA(passwords, &cuda_results);
    }
    catch (const std::exception& error)
    {
        std::cerr << "[FAIL] CUDA MD5 error: " << error.what() << '\n';
        return 1;
    }

    for (std::size_t i = 0; i < passwords.size(); ++i)
    {
        bit32 cpu_result[4];
        MD5Hash(passwords[i], cpu_result);

        for (std::size_t word = 0; word < 4; ++word)
        {
            if (cuda_results[i][word] != cpu_result[word])
            {
                std::cerr << "[FAIL] Digest mismatch for "
                          << passwords[i] << " at word " << word << '\n';
                std::cerr << std::hex << std::setfill('0');
                std::cerr << "CPU : ";
                for (std::size_t j = 0; j < 4; ++j)
                {
                    std::cerr << std::setw(8) << cpu_result[j];
                }
                std::cerr << "\nCUDA: ";
                for (std::size_t j = 0; j < 4; ++j)
                {
                    std::cerr << std::setw(8) << cuda_results[i][j];
                }
                std::cerr << std::dec << '\n';
                return 1;
            }
        }

        std::cout << "[PASS] " << passwords[i] << '\n';
    }

    try
    {
        // HashVector() uses this no-result path for throughput measurements.
        MD5HashBatchCUDA(passwords, nullptr);
    }
    catch (const std::exception& error)
    {
        std::cerr << "[FAIL] CUDA no-result path: " << error.what() << '\n';
        return 1;
    }

    std::cout << "CUDA MD5 correctness PASS.\n";
    return 0;
}
